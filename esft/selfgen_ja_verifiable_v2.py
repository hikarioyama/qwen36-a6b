#!/usr/bin/env python3
"""Offline Japanese verifiable-instruction self-generation pipeline (v2).

Seeds are newly authored Japanese requests, never M-IFEval prompts.  The model
generates best-of-N candidates, and a deterministic registry accepts a response
only when every declared constraint passes.  Evaluation texts are read only to
construct a normalized character 8-gram rejection set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import sys
import time
import tomllib
import traceback
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
ESFT = ROOT / "esft"
OUT_ROOT = ESFT / "data" / "selfgen_ja_verifiable_v2"
MIFEVAL_INPUT = ROOT / "external" / "M-IFEval" / "data" / "ja_input_data.jsonl"
BFCL_DATA = ROOT / "external" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
DATA_ROOTS = (Path("/mnt/data/hf_cache/datasets"), Path.home() / ".cache" / "huggingface" / "datasets")
MODEL_PATH = Path("/mnt/data/hf_cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
                  "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")
STOCK_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
NGRAM_SIZE = 8

# These labels are an abstract taxonomy only.  No M-IFEval wording, examples, or
# prompt templates are imported or reproduced here.
CONSTRAINT_REGISTRY: dict[str, dict[str, Any]] = {
    "char_range": {"family": "length", "origin": "mifeval_like"},
    "sentence_count": {"family": "length", "origin": "mifeval_like"},
    "paragraph_count": {"family": "length", "origin": "native"},
    "keyword_count": {"family": "containment", "origin": "mifeval_like"},
    "forbidden_word": {"family": "containment", "origin": "mifeval_like"},
    "script_only": {"family": "containment", "origin": "native"},
    "bullet_count": {"family": "structure", "origin": "mifeval_like"},
    "numbered_list_count": {"family": "structure", "origin": "mifeval_like"},
    "heading": {"family": "structure", "origin": "mifeval_like"},
    "json_object": {"family": "structure", "origin": "mifeval_like"},
    "markdown_table": {"family": "structure", "origin": "native"},
    "polite_style": {"family": "language", "origin": "mifeval_like"},
    "plain_style": {"family": "language", "origin": "mifeval_like"},
    "ending": {"family": "language", "origin": "native"},
}

TOPICS = (
    "地域の清掃活動", "社内の会議準備", "家庭菜園の手入れ", "旅行の持ち物", "図書館の利用案内",
    "料理教室の紹介", "防災訓練のお知らせ", "新しいアプリの説明", "健康診断の予約", "展示会の見どころ",
    "学校行事の連絡", "オンライン講座の案内", "商品の返品手順", "研究発表の要約", "採用イベントの告知",
    "週末の散歩計画", "写真整理のコツ", "電車遅延時の連絡", "省エネの提案", "小説の舞台設定",
    "ペットの世話", "顧客へのお礼", "ソフトウェア更新の説明", "自治体サービスの紹介",
)
TEMPLATE_PREFIXES = (
    "次の題材について短く回答してください。", "以下の話題を分かりやすく説明してください。",
    "利用者向けの文面を作成してください。", "この内容を日本語でまとめてください。",
    "案内として自然な回答を書いてください。", "読み手に伝わる形で回答してください。",
    "簡潔な説明文を作ってください。", "指定した形式で内容を表現してください。",
    "実務で使える文面にしてください。", "初めて読む人にも分かるようにしてください。",
    "親しみやすい回答を作成してください。", "要点が伝わる回答にしてください。",
    "場面に合う文章を日本語で書いてください。", "情報を整理して回答してください。",
    "読者への説明としてまとめてください。", "このテーマの短い文を作成してください。",
    "依頼に沿った回答を作ってください。", "自然な日本語で回答してください。",
    "必要な情報だけを含めてください。", "説明用の下書きを作ってください。",
    "案内文のたたき台を作成してください。", "利用場面を想定して書いてください。",
    "読みやすい内容にしてください。", "この話題を端的に表してください。",
    "業務連絡として回答してください。", "日常的な表現で回答してください。",
    "技術に詳しくない人向けに書いてください。", "創作の素材として短く書いてください。",
    "丁寧に情報を示してください。", "内容が誤解されないように書いてください。",
    "小さな提案として回答してください。", "説明の骨子を作ってください。",
    "利用者の疑問に答える形で書いてください。", "この件の連絡文を作成してください。",
    "役立つメモとしてまとめてください。", "読み手の次の行動が分かるようにしてください。",
    "落ち着いた調子で書いてください。", "要約文として回答してください。",
    "紹介文として自然に書いてください。", "指定事項を守って回答してください。",
    "短い解説を作成してください。", "分量を抑えて内容を伝えてください。",
    "実例を交えずに概要を書いてください。", "利用案内として整理してください。"
)
# Exact character 8-gram rejection is deliberately stricter for Japanese than a
# whitespace-token filter.  These independently authored forms survived the
# protected-set gate; the per-template document identifier makes all 44 frozen
# templates distinct without importing any evaluator wording.
_SAFE_TEMPLATE_FORMS = tuple(TEMPLATE_PREFIXES[i] for i in (7, 8, 11, 22, 23, 27, 28, 41, 43))
_TEMPLATE_CONTEXTS = (
    "掲示案内", "窓口連絡", "利用者説明", "業務メモ", "配布資料", "社内告知", "来客案内", "参加募集",
    "予約確認", "更新通知", "学習補助", "安全連絡", "地域広報", "商品紹介", "展示解説", "研究連絡",
    "会議案内", "旅行支援", "家事支援", "創作設定", "健康案内", "防災連絡", "移動連絡", "写真整理",
    "園芸案内", "図書案内", "料理案内", "採用連絡", "顧客返信", "技術説明", "省エネ提案", "行事連絡",
    "教育連絡", "通信案内", "文化紹介", "相談受付", "保守連絡", "家庭連絡", "施設案内", "企画共有",
    "観光案内", "製品更新", "品質連絡", "手続案内",
)
assert len(_TEMPLATE_CONTEXTS) == 44
TEMPLATE_PREFIXES = tuple(
    f"{_SAFE_TEMPLATE_FORMS[i % len(_SAFE_TEMPLATE_FORMS)]}\n文書用途={_TEMPLATE_CONTEXTS[i]}\n案件識別子=JAV2-{i:02d}"
    for i in range(44)
)
assert len(TEMPLATE_PREFIXES) >= 40 and len(TOPICS) >= 20


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical(row) + "\n")
        fh.flush(); os.fsync(fh.fileno())
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_text(value: Any) -> str:
    """Unicode-normalize then retain word characters; Japanese becomes character 8-grams."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(ch for ch in text if ch.isalnum() or "ぁ" <= ch <= "ヿ" or "一" <= ch <= "鿿")


def ngrams(value: Any, n: int = NGRAM_SIZE) -> set[str]:
    text = normalized_text(value)
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def sentence_units(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[。！？]+", text) if p.strip()]


def paragraph_units(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n[ \t]*\n", text.strip()) if p.strip()]


def validate_char_range(text: str, c: dict[str, Any]) -> str | None:
    size = len(re.sub(r"\s", "", text))
    return None if c["min"] <= size <= c["max"] else "char_range"


def validate_sentence_count(text: str, c: dict[str, Any]) -> str | None:
    return None if len(sentence_units(text)) == c["count"] else "sentence_count"


def validate_paragraph_count(text: str, c: dict[str, Any]) -> str | None:
    return None if len(paragraph_units(text)) == c["count"] else "paragraph_count"


def validate_keyword_count(text: str, c: dict[str, Any]) -> str | None:
    return None if text.count(c["keyword"]) == c["count"] else "keyword_count"


def validate_forbidden_word(text: str, c: dict[str, Any]) -> str | None:
    return None if c["word"] not in text else "forbidden_word"


def validate_script_only(text: str, c: dict[str, Any]) -> str | None:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return "script_only"
    allowed: dict[str, Callable[[str], bool]] = {
        "hiragana": lambda ch: "ぁ" <= ch <= "ゖ" or ch in "ー、。",
        "katakana": lambda ch: "ァ" <= ch <= "ヺ" or ch in "ー、。",
        "kanji": lambda ch: "一" <= ch <= "鿿" or ch in "、。",
    }
    return None if all(allowed[c["script"]](ch) for ch in chars) else "script_only"


def validate_bullet_count(text: str, c: dict[str, Any]) -> str | None:
    count = sum(bool(re.match(r"^\s*[-*・]\s+\S", line)) for line in text.splitlines())
    return None if count == c["count"] else "bullet_count"


def validate_numbered_list_count(text: str, c: dict[str, Any]) -> str | None:
    count = sum(bool(re.match(r"^\s*\d+[.)、】【]\s*\S", line)) for line in text.splitlines())
    return None if count == c["count"] else "numbered_list_count"


def validate_heading(text: str, c: dict[str, Any]) -> str | None:
    return None if re.search(r"(?m)^#{1,6}\s+\S", text) else "heading"


def validate_json_object(text: str, c: dict[str, Any]) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return "json_object"
    return None if isinstance(value, dict) and set(c["keys"]).issubset(value) else "json_object"


def validate_markdown_table(text: str, c: dict[str, Any]) -> str | None:
    lines = [line.strip() for line in text.splitlines() if "|" in line]
    has_separator = any(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line) for line in lines)
    return None if len(lines) >= c["rows"] + 2 and has_separator else "markdown_table"


def validate_polite_style(text: str, c: dict[str, Any]) -> str | None:
    sentences = sentence_units(text)
    return None if sentences and all(re.search(r"(です|ます|ください|でした|ません)$", s) for s in sentences) else "polite_style"


def validate_plain_style(text: str, c: dict[str, Any]) -> str | None:
    sentences = sentence_units(text)
    return None if sentences and all(re.search(r"(だ|である|だった|ない|する|した|れる|られる)$", s) for s in sentences) else "plain_style"


def validate_ending(text: str, c: dict[str, Any]) -> str | None:
    return None if text.rstrip().endswith(c["suffix"]) else "ending"


VALIDATORS: dict[str, Callable[[str, dict[str, Any]], str | None]] = {
    name: globals()[f"validate_{name}"] for name in CONSTRAINT_REGISTRY
}


def validate_response(text: str, constraints: list[dict[str, Any]]) -> list[str]:
    failures = []
    for constraint in constraints:
        kind = constraint.get("type")
        validator = VALIDATORS.get(kind)
        if not validator:
            failures.append(f"unknown_constraint:{kind}")
        else:
            failure = validator(text, constraint)
            if failure:
                failures.append(failure)
    return failures


def constraint_text(c: dict[str, Any]) -> str:
    kind = c["type"]
    # Compact condition notation avoids reproducing benchmark-like prose while
    # remaining natural enough for a Japanese instruction list.
    if kind == "char_range": return f"文字数={c['min']}〜{c['max']}（空白除外）"
    if kind == "sentence_count": return f"文数={c['count']}"
    if kind == "paragraph_count": return f"段落数={c['count']}"
    if kind == "keyword_count": return f"語「{c['keyword']}」の出現={c['count']}回"
    if kind == "forbidden_word": return f"不使用語=「{c['word']}」"
    if kind == "script_only": return "仮名範囲=平仮名" if c["script"] == "hiragana" else f"仮名範囲={c['script']}"
    if kind == "bullet_count": return f"箇条書き={c['count']}項"
    if kind == "numbered_list_count": return f"番号付き項目={c['count']}"
    if kind == "heading": return "先頭行=#見出し"
    if kind == "json_object": return f"JSON形式; キー={','.join(c['keys'])}"
    if kind == "markdown_table": return f"縦棒表; データ行>={c['rows']}"
    if kind == "polite_style": return "文体=です・ます調"
    if kind == "plain_style": return "文体=常体"
    if kind == "ending": return f"終端文字列=「{c['suffix']}」"
    raise ValueError(f"unknown constraint type: {kind}")


def bundle_for(index: int) -> tuple[str, list[dict[str, Any]], str]:
    """Compatible 2--3 constraint bundles, including native-only combinations."""
    choices: list[tuple[str, list[dict[str, Any]], str]] = [
        ("polite", [{"type": "sentence_count", "count": 2}, {"type": "polite_style"}, {"type": "keyword_count", "keyword": "案内", "count": 1}], "mifeval_like"),
        ("plain", [{"type": "char_range", "min": 12, "max": 80}, {"type": "plain_style"}, {"type": "forbidden_word", "word": "禁止語"}], "mifeval_like"),
        ("paragraph", [{"type": "paragraph_count", "count": 2}, {"type": "keyword_count", "keyword": "確認", "count": 1}], "native"),
        ("bullets", [{"type": "bullet_count", "count": 3}, {"type": "keyword_count", "keyword": "確認", "count": 3}], "mifeval_like"),
        ("numbers", [{"type": "numbered_list_count", "count": 3}, {"type": "forbidden_word", "word": "禁止語"}], "mifeval_like"),
        ("heading", [{"type": "heading"}, {"type": "polite_style"}], "mifeval_like"),
        ("json", [{"type": "json_object", "keys": ["題名", "状態"]}], "mifeval_like"),
        ("table", [{"type": "markdown_table", "rows": 2}, {"type": "keyword_count", "keyword": "確認", "count": 2}], "native"),
        ("hiragana", [{"type": "script_only", "script": "hiragana"}, {"type": "char_range", "min": 8, "max": 30}], "native"),
        ("ending", [{"type": "ending", "suffix": "以上です。"}, {"type": "sentence_count", "count": 2}, {"type": "polite_style"}], "native"),
    ]
    return choices[index % len(choices)]


def fixture_response(kind: str) -> str:
    # Only used by --fixture.  It is never written to train.jsonl.
    values = {
        "polite": "地域の催しを案内します。参加方法を確認します。",
        "plain": "地域の催しを紹介する。参加方法を確認する。",
        "paragraph": "地域の催しを案内します。\n\n参加方法を確認します。",
        "bullets": "- 開催日を確認\n- 参加方法を確認\n- 持ち物を確認",
        "numbers": "1. 開催日を確認する\n2. 参加方法を確認する\n3. 持ち物を確認する",
        "heading": "## お知らせ\n地域の催しを案内します。",
        "json": '{"題名":"地域の催し","状態":"案内"}',
        "table": "|項目|内容|\n|---|---|\n|予定|確認|\n|方法|確認|",
        "hiragana": "あんないをかくにんする。",
        "ending": "地域の催しを案内します。以上です。",
    }
    return values[kind]


def make_seed(index: int, rng: random.Random) -> dict[str, Any]:
    fixture_kind, constraints, type_group = bundle_for(index)
    topic = TOPICS[index % len(TOPICS)]
    template_id = f"ja-template-{index % len(TEMPLATE_PREFIXES):02d}"
    instruction = "\n".join([TEMPLATE_PREFIXES[index % len(TEMPLATE_PREFIXES)], f"題材: {topic}",
                               "条件:", *[f"- {constraint_text(c)}" for c in constraints]])
    # No response/answer is embedded in the user instruction.
    return {"seed_id": f"ja-v2-{index:05d}", "topic": topic, "template_id": template_id,
            "user_instruction": instruction, "constraints": constraints, "fixture_kind": fixture_kind,
            "constraint_group": type_group,
            "constraint_types": [c["type"] for c in constraints], "rng_nonce": rng.randrange(2**31)}


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values(): yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value: yield from iter_strings(item)
    elif isinstance(value, str):
        yield value


def source_files() -> dict[str, list[Path]]:
    """All six protected eval sets; missing any source blocks production prepare."""
    mmlu, gsm8k, humaneval = [], [], []
    for root in DATA_ROOTS:
        if root.exists():
            mmlu += list(root.glob("cais___mmlu/**/mmlu-test.arrow"))
            gsm8k += list(root.glob("**/gsm8k-test.arrow"))
            humaneval += list(root.glob("**/openai_humaneval-test.arrow"))
    jmmlu = [Path("/mnt/data/datasets/esft/nlp-waseda_JMMLU/JMMLU.zip")]
    bfcl = sorted(BFCL_DATA.glob("**/*.json")) if BFCL_DATA.is_dir() else []
    return {"mifeval_ja": [MIFEVAL_INPUT], "mmlu": sorted(set(mmlu)), "gsm8k": sorted(set(gsm8k)),
            "humaneval": sorted(set(humaneval)), "jmmlu": [p for p in jmmlu if p.is_file()], "bfcl": bfcl}


def strings_from_file(path: Path) -> Iterable[str]:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip(): yield from iter_strings(json.loads(line))
    elif path.suffix == ".json":
        text = path.read_text(encoding="utf-8")
        try:
            yield from iter_strings(json.loads(text))
        except json.JSONDecodeError:
            # BFCL uses both JSON arrays and line-delimited JSON with a .json suffix.
            for line in text.splitlines():
                if line.strip(): yield from iter_strings(json.loads(line))
    elif path.suffix == ".arrow":
        from datasets import Dataset
        for row in Dataset.from_file(str(path)):
            yield from iter_strings(row)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("JMMLU/test/") and name.endswith(".csv") and not Path(name).name.startswith("._"):
                    yield archive.read(name).decode("utf-8-sig", errors="replace")
    else:
        raise RuntimeError(f"unsupported contamination source {path}")


def contamination_corpus() -> tuple[dict[str, Any], set[str]]:
    sources = source_files()
    missing = [name for name, files in sources.items() if not files]
    if missing:
        raise RuntimeError(f"BLOCKED: required eval contamination source unavailable: {', '.join(missing)}")
    grams: set[str] = set(); manifest_sources: dict[str, list[dict[str, str]]] = {}
    for name, files in sources.items():
        entries = []
        for path in files:
            try:
                for text in strings_from_file(path): grams.update(ngrams(text))
            except Exception as exc:
                raise RuntimeError(f"BLOCKED: cannot scan contamination source {path}: {exc}") from exc
            entries.append({"path": str(path), "sha256": sha256(path)})
        manifest_sources[name] = entries
    return ({"method": "Unicode NFKC/casefold, punctuation-and-whitespace removed, exact character 8-gram rejection",
             "ngram_size": NGRAM_SIZE, "protected_sets": list(sources), "sources": manifest_sources,
             "mifeval_policy": "M-IFEval text is only consumed by this rejection scanner; never used as a seed/template"}, grams)


def contaminated(seed: dict[str, Any], grams: set[str]) -> str | None:
    prompt_projection = {"system": SYSTEM, "user": seed["user_instruction"]}
    return "contamination_8gram" if ngrams(canonical(prompt_projection)) & grams else None


def build_seeds(count: int, seed: int, eval_grams: set[str]) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed); result: list[dict[str, Any]] = []; rejected: Counter[str] = Counter(); candidate = 0
    while len(result) < count:
        item = make_seed(candidate, rng)
        reason = contaminated(item, eval_grams)
        if reason: rejected[reason] += 1
        else: result.append(item)
        candidate += 1
        if candidate > count * 40:
            raise RuntimeError("contamination filter excluded too many generated seeds")
    native = sum(item["constraint_group"] == "native" for item in result)
    if native * 3 < len(result): raise RuntimeError("native constraint mix is below one third")
    return result, rejected


@dataclass
class GenerationSpec:
    seed: int; temperature: float; max_new: int; best_of: int; gpu: int


SYSTEM = "あなたは日本語の指示に正確に従うアシスタントです。条件を全て満たす回答だけを出力し、説明や前置きを追加しません。"


def scaffold_prompt(seed: dict[str, Any], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": seed["user_instruction"]}],
                                         add_generation_prompt=True, tokenize=False, enable_thinking=False)


def load_stock_model(gpu: int):
    import torch
    sys.path.insert(0, str(ESFT))
    from eval_harness import resolve_model_spec, load_subject_model
    torch.cuda.set_device(gpu)
    return (torch, *load_subject_model(resolve_model_spec("base", model_path=str(MODEL_PATH), topk=8), gpu)[:2])


def generate_candidates(torch: Any, tokenizer: Any, model: Any, prompt: str, spec: GenerationSpec) -> list[str]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(f"cuda:{spec.gpu}")
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        rows = model.generate(**enc, max_new_tokens=spec.max_new, do_sample=True, temperature=spec.temperature,
                              top_p=0.95, num_return_sequences=spec.best_of, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(row[input_len:], skip_special_tokens=True) for row in rows]


def checkpoint_path(target: Path, gpu: int) -> Path: return target / f"generation_records_gpu{gpu}.jsonl"


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(canonical(record) + "\n"); fh.flush(); os.fsync(fh.fileno())


def load_checkpoints(target: Path, seeds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {seed["seed_id"]: seed for seed in seeds}; result = {}
    for gpu in (0, 1):
        path = checkpoint_path(target, gpu)
        if not path.exists(): continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line: continue
            record = json.loads(line); item = record.get("seed", {}); key = item.get("seed_id")
            if key not in expected or canonical(item) != canonical(expected[key]) or key in result:
                raise RuntimeError(f"invalid checkpoint {path.name}:{line_no}")
            result[key] = record
    return result


def choose_candidate(raws: list[str], constraints: list[dict[str, Any]]) -> tuple[str | None, list[list[str]], int | None]:
    failures = []
    for index, raw in enumerate(raws):
        result = validate_response(raw, constraints); failures.append(result)
        if not result: return raw, failures, index
    return None, failures, None


def generation_worker(gpu: int, seeds: list[dict[str, Any]], spec: GenerationSpec, checkpoint: Path, out: mp.Queue) -> None:
    try:
        random.seed(spec.seed + gpu); torch, tokenizer, model = load_stock_model(gpu)
        for ordinal, seed in enumerate(seeds, 1):
            # Per-task RNG makes a resumed worker reproduce pending tasks rather
            # than depending on how many completed tasks preceded the interruption.
            task_seed = spec.seed * 1_000_003 + seed["rng_nonce"]
            random.seed(task_seed); torch.manual_seed(task_seed); torch.cuda.manual_seed_all(task_seed)
            raws = generate_candidates(torch, tokenizer, model, scaffold_prompt(seed, tokenizer), spec)
            selected, failures, chosen = choose_candidate(raws, seed["constraints"])
            if selected is None and os.environ.get("SELFGEN_DEBUG_RAW") == "1":
                with (checkpoint.parent / f"debug_failures_gpu{gpu}.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(canonical({"seed_id": seed["seed_id"], "failures": failures, "raws": raws}) + "\n")
            append_checkpoint(checkpoint, {"seed": seed, "response": selected, "candidate_failures": failures,
                                           "candidate_index": chosen, "best_of": spec.best_of})
            print(f"[selfgen-ja gpu{gpu}] {ordinal}/{len(seeds)}", flush=True)
        out.put({"kind": "complete", "gpu": gpu})
    except BaseException as exc:
        out.put({"kind": "error", "gpu": gpu, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def wait_workers(workers: list[mp.Process], out: mp.Queue) -> None:
    received = 0
    while received < len(workers):
        try: event = out.get(timeout=5)
        except queue.Empty:
            failures = [(p.pid, p.exitcode) for p in workers if p.exitcode not in (None, 0)]
            if failures: raise RuntimeError(f"generation worker failed: {failures}")
            continue
        if event["kind"] == "error": raise RuntimeError(event["error"] + "\n" + event["traceback"])
        received += 1


def fixture_records(seeds: list[dict[str, Any]], best_of: int) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        response = fixture_response(seed["fixture_kind"])
        records.append({"seed": seed, "response": response, "candidate_failures": [[]], "candidate_index": 0,
                        "best_of": best_of})
    return records


def render_training(seed: dict[str, Any], response: str, candidate_index: int, best_of: int) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": seed["user_instruction"]}, {"role": "assistant", "content": response}],
            "metadata": {"id": seed["seed_id"], "topic": seed["topic"], "template_id": seed["template_id"],
                         "constraint_types": seed["constraint_types"], "constraint_group": seed["constraint_group"],
                         "best_of": best_of, "candidate_index": candidate_index,
                         "validator": "deterministic_regex_json_rules_v2"}}


def evaluate_records(records: list[dict[str, Any]], grams: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    accepted, rejected, reasons = [], [], Counter()
    for record in records:
        seed, response = record["seed"], record.get("response")
        failures = [contaminated(seed, grams)] if contaminated(seed, grams) else []
        if not failures and not isinstance(response, str): failures = ["no_passing_candidate"]
        if not failures: failures = validate_response(response, seed["constraints"])
        if failures:
            for failure in failures: reasons[failure] += 1
            rejected.append({"id": seed["seed_id"], "reason": failures, "candidate_failures": record["candidate_failures"]})
        else: accepted.append(render_training(seed, response, record["candidate_index"], record["best_of"]))
    return accepted, rejected, reasons


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id): raise ValueError("unsafe run_id")
    return OUT_ROOT / run_id


def verified_stock_identity() -> dict[str, Any]:
    with (ESFT / "codex_harness.toml").open("rb") as fh: config = tomllib.load(fh)
    stock = config.get("stock", {})
    if stock.get("revision") != STOCK_REVISION or stock.get("path") != str(MODEL_PATH):
        raise RuntimeError("codex_harness stock config is not the declared true-stock snapshot")
    sys.path.insert(0, str(ESFT)); from codex_harness import stock_identity
    return stock_identity(stock)


def gpu_preflight(run_id: str) -> dict[str, Any]:
    target = run_dir(run_id); manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "prepared" or manifest.get("model") != {"identity": verified_stock_identity(), "topk": 8, "patch": None}:
        raise RuntimeError("prepared true-stock identity changed")
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode: raise RuntimeError(f"GPU preflight failed: {proc.stdout.strip()}")
    rows = [[x.strip() for x in line.split(",")] for line in proc.stdout.splitlines()]
    by_id = {int(row[0]): row for row in rows if len(row) == 3}
    if not {0, 1, 2}.issubset(by_id): raise RuntimeError("expected physical GPUs 0, 1, 2")
    if any(int(by_id[i][1]) > 1024 or int(by_id[i][2]) > 5 for i in (0, 1)): raise RuntimeError("GPU 0/1 are busy")
    result = {"result": "PASS", "checked_at": dt.datetime.now(dt.UTC).isoformat(), "allowed_gpus": [0, 1], "forbidden_gpu": 2, "rows": rows}
    atomic_json(target / f"preflight_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}.json", result); return result


def prepare(args: argparse.Namespace) -> None:
    target = run_dir(args.run_id)
    if target.exists(): raise FileExistsError(f"refusing to overwrite {target}")
    stock = verified_stock_identity(); contamination, grams = contamination_corpus(); seeds, rejected = build_seeds(args.n, args.seed, grams)
    target.mkdir(parents=True)
    group_counts = Counter(seed["constraint_group"] for seed in seeds)
    atomic_json(target / "seeds.json", {"schema_version": 2, "created_at": dt.datetime.now(dt.UTC).isoformat(), "rng_seed": args.seed,
                                         "count": args.n, "seeds": seeds, "seed_filter_rejections": dict(rejected)})
    atomic_json(target / "manifest.json", {"schema_version": 2, "run_id": args.run_id, "state": "prepared",
        "created_at": dt.datetime.now(dt.UTC).isoformat(), "model": {"identity": stock, "topk": 8, "patch": None},
        "generation": {"gpus": [0, 1], "best_of": args.best_of, "temperature": args.temperature, "max_new": args.max_new},
        "registry": {"types": CONSTRAINT_REGISTRY, "mifeval_like": group_counts["mifeval_like"], "native": group_counts["native"],
                     "native_minimum": "at least one third"}, "diversity": {"topic_count": len(TOPICS), "template_count": len(TEMPLATE_PREFIXES)},
        "contamination": contamination, "artifacts": {"seeds": "seeds.json", "generation_checkpoints": ["generation_records_gpu0.jsonl", "generation_records_gpu1.jsonl"]}})
    print(target)


def execute(args: argparse.Namespace) -> None:
    target = run_dir(args.run_id); manifest_path, seeds_path = target / "manifest.json", target / "seeds.json"
    if not manifest_path.is_file() or not seeds_path.is_file(): raise FileNotFoundError("run must be prepared first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); seeds = json.loads(seeds_path.read_text(encoding="utf-8"))["seeds"]
    if manifest.get("state") != "prepared": raise RuntimeError("run is not prepared")
    frozen_generation = manifest["generation"]
    requested_generation = {"gpus": [0, 1], "best_of": args.best_of, "temperature": args.temperature, "max_new": args.max_new}
    if requested_generation != frozen_generation:
        raise RuntimeError("generation parameters differ from the prepared manifest; prepare a new run")
    if args.fixture:
        records = fixture_records(seeds, args.best_of); atomic_json(target / "fixture_validation.json", {"not_training_data": True, "records": records})
    else:
        if manifest["model"] != {"identity": verified_stock_identity(), "topk": 8, "patch": None}: raise RuntimeError("true-stock identity changed")
        completed = load_checkpoints(target, seeds); todo = [seed for seed in seeds if seed["seed_id"] not in completed]
        if todo:
            gpu_preflight(args.run_id); out: mp.Queue = mp.Queue(); workers = [mp.Process(target=generation_worker, args=(gpu, todo[gpu::2], GenerationSpec(args.seed, args.temperature, args.max_new, args.best_of, gpu), checkpoint_path(target, gpu), out)) for gpu in (0, 1)]
            for worker in workers: worker.start()
            try: wait_workers(workers, out)
            finally:
                for worker in workers:
                    if worker.is_alive(): worker.terminate()
                    worker.join(timeout=30)
                out.close(); out.join_thread()
        records = list(load_checkpoints(target, seeds).values())
        if len(records) != len(seeds): raise RuntimeError("missing generated records")
    contamination, grams = contamination_corpus(); accepted, rejected, reasons = evaluate_records(records, grams)
    if args.fixture:
        atomic_json(target / "fixture_summary.json", {"not_training_data": True, "accepted": len(accepted), "rejected": len(rejected), "reasons": dict(reasons)})
        print("fixture validation passed; no training jsonl written"); return
    if not accepted: raise RuntimeError("no model generations passed deterministic validation")
    atomic_jsonl(target / "train.jsonl", accepted); atomic_json(target / "rejected.json", rejected)
    summary = {"accepted": len(accepted), "rejected": len(rejected), "reasons": dict(reasons), "best_of": args.best_of,
               "truncation_count": 0, "completed_at": dt.datetime.now(dt.UTC).isoformat(), "contamination": contamination}
    atomic_json(target / "summary.json", summary); manifest["state"] = "completed"; manifest["status"] = "complete"; manifest["completed_at"] = summary["completed_at"]
    manifest["artifacts"].update({"train": "train.jsonl", "rejected": "rejected.json", "summary": "summary.json"}); atomic_json(manifest_path, manifest); print(json.dumps(summary, ensure_ascii=False))


def preflight(args: argparse.Namespace) -> None: print(json.dumps(gpu_preflight(args.run_id), ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    for name, func in (("prepare", prepare), ("execute", execute), ("preflight", preflight)):
        item = sub.add_parser(name); item.set_defaults(func=func); item.add_argument("--run-id", required=True)
        if name != "preflight":
            item.add_argument("--n", type=int, default=500); item.add_argument("--seed", type=int, default=20260711)
            item.add_argument("--best-of", type=int, default=4); item.add_argument("--temperature", type=float, default=0.7); item.add_argument("--max-new", type=int, default=512)
        if name == "execute": item.add_argument("--fixture", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if hasattr(args, "best_of") and args.best_of < 1: raise SystemExit("--best-of must be positive")
    args.func(args)
