#!/usr/bin/env python3
"""CPU-only tests for Corpus Judge v1 prompt preparation and ledger handling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "esft" / "judge" / "corpus_judge_v1.py"
spec = importlib.util.spec_from_file_location("corpus_judge_v1", MODULE_PATH)
assert spec and spec.loader
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class CorpusJudgeTest(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "source.jsonl"
        rows = [
            {"metadata": {"id": f"seed-{i}"}, "messages": [{"role": "user", "content": f"request {i}"}, {"role": "assistant", "content": f"answer {i}"}]}
            for i in range(11)
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return source

    def test_prepare_resume_append_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root)
            rubric = root / "rubric.md"
            rubric.write_text("strict rubric\n", encoding="utf-8")
            out = root / "out"
            self.assertEqual(judge.main(["prepare", "--input", str(source), "--rubric", str(rubric),
                                         "--output-dir", str(out), "--batch-size", "10"]), 0)
            prompts = sorted((out / "prompts").glob("*.md"))
            self.assertEqual(len(prompts), 2)
            self.assertIn('"id": "seed-0"', prompts[0].read_text(encoding="utf-8"))
            sha = judge.sha256_file(rubric)
            incoming = root / "incoming.jsonl"
            first_batch = json.loads((out / "batch_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
            incoming.write_text(json.dumps({"id": "seed-0", "verdict": "reject", "reason": "Mechanical template.",
                                            "rubric_sha": sha, "batch_id": first_batch["batch_id"]}) + "\n", encoding="utf-8")
            self.assertEqual(judge.main(["append", "--input", str(incoming), "--rubric", str(rubric),
                                         "--output-dir", str(out)]), 0)
            self.assertEqual(judge.main(["prepare", "--input", str(source), "--rubric", str(rubric),
                                         "--output-dir", str(out), "--batch-size", "10"]), 0)
            pending = json.loads((out / "pending_batches.json").read_text(encoding="utf-8"))
            rendered = "".join((out / path).read_text(encoding="utf-8") for path in pending["prompt_files"])
            self.assertNotIn('"id": "seed-0"', rendered)
            self.assertEqual(judge.main(["summary", "--output-dir", str(out)]), 0)

    def test_toucan_shape_decodes_messages_and_uses_uuid(self):
        row = {"uuid": "u-1", "messages": json.dumps([{"role": "user", "content": "x"}]),
               "available_tools": json.dumps([{"name": "tool"}]), "question_quality_assessment": "ignored"}
        prepared = judge.prompt_record(row, Path("/tmp/source.parquet"), 3)
        self.assertEqual(prepared["id"], "u-1")
        self.assertEqual(prepared["messages"][0]["content"], "x")
        self.assertNotIn("question_quality_assessment", prepared)
if __name__ == "__main__":
    unittest.main()
