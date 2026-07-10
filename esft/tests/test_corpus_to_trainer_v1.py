#!/usr/bin/env python3
"""CPU tests for the Toucan/selfgen trainer converter and template contract."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[2]
CONVERTER_PATH = REPO / "esft" / "corpus_to_trainer_v1.py"
TRAINER_PATH = REPO / "esft" / "deploy" / "train_fullffn_dcp.py"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SELFGEN = REPO / "esft/data/selfgen_toolcall_v1/20260711_toolcall_v1_pilot500_r3/train.jsonl"
TOUCAN = Path("/mnt/vault/corpora/derived/qwen36-a6b-intake-20260711-v1/clean/toucan-1.5m")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_module("corpus_to_trainer_v1", CONVERTER_PATH)
trainer = load_module("train_fullffn_dcp_for_converter_test", TRAINER_PATH)


def load_tokenizer():
    if not SNAPSHOT.is_dir():
        raise unittest.SkipTest(f"missing required local tokenizer snapshot: {SNAPSHOT}")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)


TOOLS = [{"type": "function", "function": {
    "name": "weather", "description": "Lookup weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
TOOL_MESSAGES = [
    {"role": "user", "content": "Weather in Tokyo?"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Tokyo"}}}]},
    {"role": "tool", "name": "weather", "content": '{"temp": 24}'},
    {"role": "assistant", "content": "It is 24 C."},
]


class TemplateContractTest(unittest.TestCase):
    def test_tool_system_content_matches_tools_branch(self):
        tokenizer = load_tokenizer()
        expected = tokenizer.apply_chat_template(TOOL_MESSAGES, tools=TOOLS, tokenize=False,
                                                 add_generation_prompt=False)
        tool_content = expected.split("<|im_start|>system\n", 1)[1].split("<|im_end|>\n", 1)[0]
        self.assertEqual(converter._tool_system_content(TOOLS), tool_content)

    def test_user_preamble_is_renderable_by_incremental_trainer(self):
        tokenizer = load_tokenizer()
        converted = converter.normalize_messages(TOOL_MESSAGES, TOOLS)
        self.assertEqual(converted[0]["role"], "user")
        for index, message in enumerate(converted):
            tokenizer.apply_chat_template(converted[:index + 1], tokenize=False,
                                          add_generation_prompt=(message["role"] != "assistant"))

    def test_prefix_text_holds_but_token_ids_do_not_at_assistant_boundary(self):
        tokenizer = load_tokenizer()
        previous_text = previous_ids = None
        saw_token_failure = False
        for i, message in enumerate(TOOL_MESSAGES):
            text = tokenizer.apply_chat_template(
                TOOL_MESSAGES[:i + 1], tokenize=False,
                add_generation_prompt=(message["role"] != "assistant"),
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if previous_text is not None:
                self.assertTrue(text.startswith(previous_text), msg=f"text prefix at turn {i}")
                if not ids[:len(previous_ids)] == previous_ids:
                    saw_token_failure = True
            previous_text, previous_ids = text, ids
        self.assertTrue(saw_token_failure, "Qwen token IDs unexpectedly became prefix-stable")


class ConverterTest(unittest.TestCase):
    def test_selfgen_round_trip_and_trainer_labels_only_assistant_turn(self):
        tokenizer = load_tokenizer()
        with SELFGEN.open(encoding="utf-8") as handle:
            source = json.loads(next(handle))
        record = converter.to_trainer_record(source, source_tag="unit-selfgen", domain="toolcall")
        round_trip = json.loads(json.dumps(record, ensure_ascii=False))
        self.assertEqual(round_trip["_source"], "unit-selfgen")
        self.assertEqual(round_trip["_domain"], "toolcall")
        self.assertEqual(round_trip["messages"][0]["role"], "user")
        call = next(m for m in round_trip["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
        self.assertIsInstance(call["tool_calls"][0]["function"]["arguments"], dict)

        input_ids, labels = trainer.render_and_tokenize(tokenizer, round_trip["messages"])
        self.assertEqual(len(input_ids), len(labels))
        self.assertIn("<tool_call>", tokenizer.decode([x for x in labels if x != -100]))

        # Recreate the trainer's exact per-turn slices, then verify every tool
        # response token is ignored while the assistant tool-call turn is not.
        prev_len = 0
        assistant_has_labels = False
        offset = 0
        for message_index, message in enumerate(round_trip["messages"]):
            rendered = tokenizer.apply_chat_template(
                round_trip["messages"][:message_index + 1], tokenize=False,
                add_generation_prompt=(message["role"] != "assistant"),
            )
            full = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            width = len(full) - prev_len
            turn_labels = labels[offset:offset + width]
            if message["role"] == "assistant":
                assistant_has_labels |= any(token != -100 for token in turn_labels)
            else:
                self.assertTrue(all(token == -100 for token in turn_labels), message["role"])
            prev_len, offset = len(full), offset + width
        self.assertTrue(assistant_has_labels)

    def test_legacy_and_sft_shapes_normalize_to_qwen_roles(self):
        legacy = {
            "messages": json.dumps([
                {"role": "system", "content": '<|im_system|>tool_declare<|im_middle|>' + json.dumps(TOOLS) + '<|im_end|>'},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "", "function_call": {"name": "weather", "arguments": '{"city":"Tokyo"}'}},
                {"role": "function", "name": "weather", "content": "{}"},
            ]),
            "available_tools": json.dumps(TOOLS),
        }
        sft = {
            "messages": json.dumps([
                {"role": "user", "content": "x"},
                {"role": "tool_call", "content": '{"name":"weather","arguments":{"city":"Tokyo"}}'},
                {"role": "tool_response", "content": "{}"},
            ]),
            "tools": json.dumps(TOOLS),
        }
        for row in (legacy, sft):
            record = converter.to_trainer_record(row, source_tag="unit", domain="toolcall")
            roles = [message["role"] for message in record["messages"]]
            self.assertEqual(roles, ["user", "assistant", "tool"])
            call = record["messages"][1]["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "weather")
            self.assertEqual(call["function"]["arguments"], {"city": "Tokyo"})

    def test_existing_completed_toucan_parquet_round_trip_when_available(self):
        candidates = [path for path in TOUCAN.rglob("*.parquet")
                      if converter.parquet_is_complete(path, 300)] if TOUCAN.is_dir() else []
        if not candidates:
            self.skipTest("no complete Toucan clean parquet currently available")
        row = next(converter.iter_parquet_rows(candidates[0], batch_size=1))
        record = converter.to_trainer_record(row, source_tag="unit-toucan", domain="toolcall")
        self.assertEqual(json.loads(json.dumps(record, ensure_ascii=False)), record)
        self.assertTrue(any(message["role"] == "assistant" for message in record["messages"]))

    def test_streaming_convert_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            written, bad = converter.convert([str(SELFGEN)], str(output), source_tag="unit-stream",
                                             domain="toolcall", max_records=2)
            self.assertEqual((written, bad), (2, 0))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["_source"] for row in rows], ["unit-stream", "unit-stream"])


if __name__ == "__main__":
    unittest.main()
