#!/usr/bin/env python3
"""CPU tests for the Toucan/selfgen trainer converter and template contract."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[2]
CONVERTER_PATH = REPO / "esft" / "corpus_to_trainer_v1.py"
TRAINER_PATH = REPO / "esft" / "deploy" / "train_fullffn_dcp.py"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
SELFGEN = REPO / "esft/data/selfgen_toolcall_v1/20260711_toolcall_v1_pilot500_r3/train.jsonl"
# Optional local Toucan intake dir; the test using it skips when unset/missing.
TOUCAN = Path(os.environ.get("A6B_TOUCAN_DIR") or "/nonexistent/toucan-intake")


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

    def test_offsets_matches_one_shot_render_and_masks_non_assistant_turns(self):
        tokenizer = load_tokenizer()
        native = converter.to_trainer_record(
            {"messages": TOOL_MESSAGES, "tools": TOOLS},
            source_tag="unit-native",
            domain="toolcall",
            tools_mode="native",
        )
        self.assertEqual(native["tools"], TOOLS)
        self.assertNotIn("# Tools", native["messages"][0]["content"])

        rendered = tokenizer.apply_chat_template(
            native["messages"], tools=native["tools"], tokenize=False)
        self.assertTrue(rendered.startswith("<|im_start|>system\n# Tools\n"))
        self.assertIn('"name": "weather"', rendered)
        expected_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        input_ids, labels = trainer.render_and_tokenize(
            tokenizer,
            native["messages"],
            tools=native["tools"],
            tokenize_mode="offsets",
        )
        self.assertEqual(input_ids, expected_ids)  # one-shot tokenisation contract
        self.assertEqual(len(input_ids), len(labels))

        spans = trainer._assistant_turn_spans(rendered, native["messages"])
        offsets = tokenizer(rendered, add_special_tokens=False,
                            return_offsets_mapping=True)["offset_mapping"]
        for token_id, (start, end), label in zip(input_ids, offsets, labels):
            assistant_token = any(left <= start and end <= right and end > start
                                  for left, right in spans)
            self.assertEqual(label, token_id if assistant_token else -100)
        labelled = tokenizer.decode([token for token in labels if token != -100])
        self.assertIn("<tool_call>", labelled)
        self.assertIn("It is 24 C.", labelled)
        self.assertNotIn("Weather in Tokyo?", labelled)
        self.assertNotIn('{"temp": 24}', labelled)

        # pack_examples must forward both tools and the opt-in mode, not silently
        # fall back to the old incremental branch.
        packed_ids, packed_labels = trainer.pack_examples(
            [native], tokenizer, seq_length=512, random_concat_ratio=0.0,
            seed=7, tokenize_mode="offsets")
        self.assertEqual(packed_ids[0][:len(input_ids)], input_ids)
        self.assertEqual(packed_labels[0][:len(labels)], labels)

        # The worker receives the whole record (not only messages), preserving
        # native tools when preparation uses multiprocessing.
        trainer._WORKER_TOK = tokenizer
        trainer._WORKER_CAP = 0
        trainer._WORKER_TOKENIZE_MODE = "offsets"
        worker_ids, worker_labels = trainer._tok_worker_one(native)
        self.assertEqual(list(worker_ids), input_ids)
        self.assertEqual(list(worker_labels), labels)

    def test_offsets_v3_plain_messages_has_stable_label_count_and_positions(self):
        tokenizer = load_tokenizer()
        # v3-style records have only plain role/content messages and no tools.
        messages = [
            {"role": "user", "content": "Give alpha."},
            {"role": "assistant", "content": "alpha"},
            {"role": "user", "content": "Give beta."},
            {"role": "assistant", "content": "beta"},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids, labels = trainer.render_and_tokenize(
            tokenizer, messages, tokenize_mode="offsets")
        self.assertEqual(input_ids, tokenizer(rendered, add_special_tokens=False)["input_ids"])
        self.assertEqual(sum(label != -100 for label in labels), 16)
        labelled = tokenizer.decode([token for token in labels if token != -100])
        self.assertIn("alpha", labelled)
        self.assertIn("beta", labelled)
        self.assertNotIn("Give alpha.", labelled)
        self.assertNotIn("Give beta.", labelled)

        repeated = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "duplicate"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "duplicate"},
        ]
        repeated_rendered = tokenizer.apply_chat_template(repeated, tokenize=False)
        self.assertEqual(len(trainer._assistant_turn_spans(repeated_rendered, repeated)), 2)

    def test_incremental_default_is_legacy_bit_identical(self):
        tokenizer = load_tokenizer()
        messages = TOOL_MESSAGES

        # This is the pre-flag implementation written inline so the assertion
        # does not merely compare two calls through the new dispatcher.
        expected_ids, expected_labels, previous_length = [], [], 0
        for index, message in enumerate(messages):
            rendered = tokenizer.apply_chat_template(
                messages[:index + 1], tokenize=False,
                add_generation_prompt=(message["role"] != "assistant"),
            )
            full_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            new_ids = full_ids[previous_length:]
            expected_ids.extend(new_ids)
            expected_labels.extend(new_ids if message["role"] == "assistant"
                                   else [-100] * len(new_ids))
            previous_length = len(full_ids)

        default = trainer.render_and_tokenize(tokenizer, messages)
        explicit = trainer.render_and_tokenize(tokenizer, messages,
                                                tokenize_mode="incremental")
        self.assertEqual(default, (expected_ids, expected_labels))
        self.assertEqual(explicit, (expected_ids, expected_labels))

    def test_cache_key_separates_mode_and_native_tools_presence(self):
        args = SimpleNamespace(data_cache_dir="/tmp/cache", train_data="data.jsonl",
                               seq_length=4096, seed=7, random_concat_ratio=0.2,
                               max_records=0, tokenize_mode="incremental")
        incremental_plain = trainer.cache_path_for(args, tools_present=False)
        incremental_tools = trainer.cache_path_for(args, tools_present=True)
        args.tokenize_mode = "offsets"
        offsets_plain = trainer.cache_path_for(args, tools_present=False)
        self.assertNotEqual(incremental_plain, incremental_tools)
        self.assertNotEqual(incremental_plain, offsets_plain)


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
