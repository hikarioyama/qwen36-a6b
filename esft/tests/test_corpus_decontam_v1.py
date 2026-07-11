#!/usr/bin/env python3
"""CPU-only scan -> decontam smoke tests with tiny JSONL and parquet inputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq


ESFT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("corpus_decontam_v1", ESFT / "corpus_decontam_v1.py")
assert SPEC and SPEC.loader
corpus = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = corpus
SPEC.loader.exec_module(corpus)


# Nine words -> two distinct 8-grams: EvalIndex.match requires >=2 distinct
# matching grams (min_hits) so a single shared boilerplate gram cannot reject.
MATCH = "alpha beta gamma delta epsilon zeta eta theta iota"


class CorpusDecontamV1Test(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        source.mkdir()
        jsonl = source / "tiny.jsonl"
        jsonl.write_text(
            json.dumps({"conversations": [{"role": "user", "content": MATCH},
                                             {"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": {}}}]}],
                        "tools": [{"type": "function"}]}) + "\n" +
            json.dumps({"conversations": [{"role": "user", "content": "fully distinct safe sample"}]}) + "\n",
            encoding="utf-8")
        parquet = source / "tiny.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"messages": json.dumps([{"role": "user", "content": MATCH}]), "id": "remove"},
            {"messages": json.dumps([{"role": "user", "content": "parquet safe record"}]), "id": "keep"},
        ]), parquet)
        eval_file = root / "mmlu_eval.jsonl"
        eval_file.write_text(json.dumps({"question": MATCH}) + "\n", encoding="utf-8")
        return source, eval_file

    def test_scan_then_decontam_streaming_jsonl_and_parquet(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, eval_file = self.make_fixture(root)
            scan_json, scan_md = root / "scan.json", root / "scan.md"
            self.assertEqual(corpus.main([
                "scan", "--input", str(source), "--batch-size", "1",
                "--output-json", str(scan_json), "--output-md", str(scan_md),
            ]), 0)
            scanned = json.loads(scan_json.read_text())
            self.assertEqual(len(scanned["files"]), 2)
            json_report = next(x for x in scanned["files"] if x["format"] == "jsonl")
            self.assertEqual(json_report["actual_row_count"], 2)
            self.assertGreater(json_report["conversation_and_tool_signals"]["messages_array"], 0)
            self.assertIn("tool_calls_field", json_report["conversation_and_tool_signals"])
            self.assertIn("Per-file schema", scan_md.read_text())

            out = root / "clean"
            # The smoke fixture owns its complete eval universe; do not depend
            # on a machine's optional HF caches or external benchmark clones.
            with mock.patch.object(corpus, "local_eval_rows", side_effect=FileNotFoundError("test fixture")):
                self.assertEqual(corpus.main([
                    "decontam", "--input", str(source), "--batch-size", "1", "--output-dir", str(out),
                    "--eval-override", f"mmlu={eval_file}",
                ]), 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["eval_sources"]["mmlu"]["status"], "AVAILABLE")
            self.assertEqual(manifest["totals"], {"input_rows": 4, "kept_rows": 2, "removed_rows": 2})
            clean_jsonl = out / source.name / "tiny.jsonl"
            self.assertEqual(len(clean_jsonl.read_text().splitlines()), 1)
            clean_parquet = out / source.name / "tiny.parquet"
            self.assertEqual(pq.ParquetFile(clean_parquet).metadata.num_rows, 1)
            log = (out / "removals.jsonl").read_text().splitlines()
            self.assertEqual(len(log), 2)
            self.assertEqual(json.loads(log[0])["matched_eval_sets"], ["mmlu"])

    def test_cjk_normalization_matches_eight_characters(self):
        tokens = corpus.normalize_tokens("あいうえおかきく")
        self.assertEqual(tokens, list("あいうえおかきく"))
        self.assertEqual(len(list(corpus.record_gram_digests({"x": "日本語の評価汚染を検出する文章"}))),
                         len(corpus.normalize_tokens("日本語の評価汚染を検出する文章")) - 7)


if __name__ == "__main__":
    unittest.main()
