from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ESFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ESFT))

import selfgen_ja_verifiable_v2 as v2


class JaVerifiableRegistryTests(unittest.TestCase):
    def test_every_registry_type_has_positive_and_negative_case(self):
        cases = {
            "char_range": ("あいうえお", {"type": "char_range", "min": 4, "max": 6}, "あ"),
            "sentence_count": ("一文です。二文です。", {"type": "sentence_count", "count": 2}, "一文です。"),
            "paragraph_count": ("一段落です。\n\n二段落です。", {"type": "paragraph_count", "count": 2}, "一段落です。"),
            "keyword_count": ("確認して確認します。", {"type": "keyword_count", "keyword": "確認", "count": 2}, "確認します。"),
            "forbidden_word": ("安全です。", {"type": "forbidden_word", "word": "危険"}, "危険です。"),
            "script_only": ("あいうえお。", {"type": "script_only", "script": "hiragana"}, "アイウ"),
            "bullet_count": ("- 一つ\n- 二つ", {"type": "bullet_count", "count": 2}, "- 一つ"),
            "numbered_list_count": ("1. 一つ\n2. 二つ", {"type": "numbered_list_count", "count": 2}, "1. 一つ"),
            "heading": ("## 見出し\n本文", {"type": "heading"}, "本文"),
            "json_object": ('{"題名":"案内","状態":"完了"}', {"type": "json_object", "keys": ["題名", "状態"]}, "[]"),
            "markdown_table": ("|項目|内容|\n|---|---|\n|予定|確認|\n|方法|確認|", {"type": "markdown_table", "rows": 2}, "|項目|内容|\n|---|---|\n|予定|確認|"),
            "polite_style": ("案内します。確認します。", {"type": "polite_style"}, "案内する。"),
            "plain_style": ("案内する。確認する。", {"type": "plain_style"}, "案内します。"),
            "ending": ("これで終わりです。", {"type": "ending", "suffix": "終わりです。"}, "これで終わり。"),
        }
        self.assertEqual(set(cases), set(v2.CONSTRAINT_REGISTRY))
        for name, (good, constraint, bad) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(v2.validate_response(good, [constraint]), [])
                self.assertIn(name, v2.validate_response(bad, [constraint]))

    def test_bundles_cover_native_third_and_fixture_is_valid(self):
        seeds, rejected = v2.build_seeds(60, 7, set())
        self.assertFalse(rejected)
        self.assertGreaterEqual(sum(s["constraint_group"] == "native" for s in seeds), 20)
        self.assertGreaterEqual(len({s["topic"] for s in seeds}), len(v2.TOPICS))
        self.assertGreaterEqual(len({s["template_id"] for s in seeds}), len(v2.TEMPLATE_PREFIXES))
        for seed in seeds:
            self.assertEqual(v2.validate_response(v2.fixture_response(seed["fixture_kind"]), seed["constraints"]), [])
            self.assertNotIn(v2.fixture_response(seed["fixture_kind"]), seed["user_instruction"])

    def test_japanese_normalized_character_ngrams_detect_overlap(self):
        grams = v2.ngrams("ＡＢＣ　日本語の案内文です")
        self.assertTrue(grams & v2.ngrams("abc日本語の案内文です"))

    def test_contamination_includes_the_fixed_system_prompt(self):
        seed = v2.make_seed(0, __import__("random").Random(0))
        system_gram = next(iter(v2.ngrams(v2.SYSTEM)))
        self.assertEqual(v2.contaminated(seed, {system_gram}), "contamination_8gram")

    def test_training_metadata_preserves_configured_best_of(self):
        seed = v2.make_seed(0, __import__("random").Random(0))
        row = v2.render_training(seed, v2.fixture_response(seed["fixture_kind"]), 2, 7)
        self.assertEqual(row["metadata"]["best_of"], 7)
        self.assertEqual(row["metadata"]["candidate_index"], 2)

    def test_source_contract_has_six_protected_evaluations(self):
        self.assertEqual(set(v2.source_files()), {"mifeval_ja", "mmlu", "gsm8k", "humaneval", "jmmlu", "bfcl"})


class JaVerifiablePipelineTests(unittest.TestCase):
    def test_prepare_then_fixture_execute_never_writes_train_data(self):
        contamination = ({"method": "test", "protected_sets": ["mifeval_ja", "mmlu", "gsm8k", "humaneval", "jmmlu", "bfcl"]}, set())
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(v2, "OUT_ROOT", Path(tmp)), \
             patch.object(v2, "verified_stock_identity", return_value={"revision": "test"}), \
             patch.object(v2, "contamination_corpus", return_value=contamination):
            v2.prepare(v2.parser().parse_args(["prepare", "--run-id", "fixture", "--n", "20", "--seed", "5"]))
            v2.execute(v2.parser().parse_args(["execute", "--run-id", "fixture", "--fixture"]))
            target = Path(tmp) / "fixture"
            summary = json.loads((target / "fixture_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary, {"not_training_data": True, "accepted": 20, "rejected": 0, "reasons": {}})
            self.assertTrue((target / "fixture_validation.json").is_file())
            self.assertFalse((target / "train.jsonl").exists())
            self.assertEqual(manifest["state"], "prepared")


if __name__ == "__main__":
    unittest.main()
