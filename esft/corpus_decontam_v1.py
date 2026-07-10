#!/usr/bin/env python3
"""Offline streaming corpus intake and eval decontamination (v1).

The matcher rejects a record when any normalized eight-token gram in its text
equals one from a locally available evaluation set.  It is deliberately a
high-recall, exact-overlap screen, not a claim that every train/eval relation
has been detected.  All Hugging Face access is forced offline; an unavailable
local evaluation source is recorded as ``SKIPPED`` rather than downloaded.

Examples (do not run the full corpus interactively):

  python3 esft/corpus_decontam_v1.py scan \
    --input /mnt/vault/corpora/toucan-1.5m --limit-per-file 2048 \
    --output-json esft/data/corpus_scan.json --output-md reports/corpus_scan.md
  python3 esft/corpus_decontam_v1.py decontam \
    --input /mnt/vault/corpora/toucan-1.5m --output-dir esft/data/toucan_clean
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable, Iterator
import zipfile

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only on misconfigured hosts
    raise SystemExit("pyarrow is required for parquet corpus streaming") from exc


ROOT = Path(__file__).resolve().parents[1]
ESFT = ROOT / "esft"
DEFAULT_INPUTS = (
    Path("/mnt/vault/corpora/toucan-1.5m"),
    Path("/mnt/vault/corpora/toolmind"),
)
EVAL_NAMES = ("mmlu", "gsm8k", "humaneval", "jmmlu", "bfcl", "mifeval")
NGRAM_SIZE = 8
HASH_BYTES = 16
JSONISH = re.compile(r"^\s*[\[{]")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_license_label(path: Path) -> dict[str, str]:
    """Conservative local-only license classification; never infer upstream terms."""
    name = path.name
    if "toucan-1.5m" in path.parts:
        return {
            "group": "A_declared",
            "status": "measured_local_readme",
            "label": "Toucan local README declares Apache-2.0.",
        }
    if name in {"APIGen-MT-5k-query.jsonl", "tau-train-query.jsonl"}:
        return {
            "group": "B_suspected_noncommercial_or_review_required",
            "status": "user_directed_suspicion_not_independently_verified",
            "label": "Do not use commercially until upstream terms are reviewed.",
        }
    if "toolmind" in path.parts:
        return {
            "group": "UNVERIFIED_upstream",
            "status": "hypothesized",
            "label": "ToolMind local README declares Apache-2.0; upstream component terms were not verified offline.",
        }
    return {"group": "UNKNOWN", "status": "unknown", "label": "No local license mapping."}


def input_files(inputs: Iterable[Path]) -> list[tuple[Path, Path]]:
    """Return (source-root, file) pairs in a deterministic order."""
    found: list[tuple[Path, Path]] = []
    for raw in inputs:
        path = raw.resolve()
        if path.is_file():
            if path.suffix.lower() in {".jsonl", ".parquet"}:
                found.append((path.parent, path))
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in {".jsonl", ".parquet"}:
                    found.append((path, child))
    if not found:
        raise FileNotFoundError("no .jsonl or .parquet corpus files found")
    return found


def rows_from_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: corpus JSONL row is not an object")
            yield row


def rows_from_parquet(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from rows_from_jsonl(path)
    elif path.suffix.lower() == ".parquet":
        yield from rows_from_parquet(path, batch_size)
    else:  # pragma: no cover - input_files filters this
        raise ValueError(path)


def schema_for(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".parquet":
        pf = pq.ParquetFile(path)
        return {
            "format": "parquet",
            "arrow_schema": str(pf.schema_arrow),
            "metadata_row_count": pf.metadata.num_rows,
        }
    return {"format": "jsonl", "arrow_schema": None, "metadata_row_count": None}


def parse_embedded_json(value: Any) -> Any:
    if isinstance(value, str) and JSONISH.match(value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def walk_text(value: Any) -> Iterator[str]:
    """Yield textual leaves, including JSON encoded in common string fields."""
    value = parse_embedded_json(value)
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_text(child)
    elif isinstance(value, str):
        yield value


def normalize_tokens(value: str) -> list[str]:
    """NFKC/casefold tokens; CJK ideographs become one token each.

    This retains the existing BFCL-style word overlap for Latin text while
    making an eight-gram meaningful for Japanese M-IFEval/JMMLU material.
    """
    text = unicodedata.normalize("NFKC", value).casefold().replace("_", " ")
    tokens: list[str] = []
    word: list[str] = []
    for char in text:
        code = ord(char)
        cjk = (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or
               0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF)
        if cjk:
            if word:
                tokens.append("".join(word))
                word = []
            tokens.append(char)
        elif char.isalnum():
            word.append(char)
        elif word:
            tokens.append("".join(word))
            word = []
    if word:
        tokens.append("".join(word))
    return tokens


def gram_digest(tokens: list[str], start: int) -> bytes:
    return hashlib.blake2b("\x1f".join(tokens[start:start + NGRAM_SIZE]).encode("utf-8"),
                           digest_size=HASH_BYTES).digest()


def record_gram_digests(row: dict[str, Any]) -> Iterator[bytes]:
    for text in walk_text(row):
        tokens = normalize_tokens(text)
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            yield gram_digest(tokens, i)


def structural_signals(row: dict[str, Any]) -> Counter[str]:
    signals: Counter[str] = Counter()
    for key in row:
        signals[f"top_level:{key}"] += 1

    def visit(value: Any) -> None:
        value = parse_embedded_json(value)
        if isinstance(value, dict):
            if "role" in value:
                signals["message_object"] += 1
                if isinstance(value.get("role"), str):
                    signals[f"role:{value['role']}"] += 1
            if "tool_calls" in value:
                signals["tool_calls_field"] += 1
            if "function_call" in value:
                signals["function_call_field"] += 1
            if "function" in value and isinstance(value.get("function"), dict):
                signals["nested_function_object"] += 1
            if "name" in value and "arguments" in value:
                signals["name_arguments_object"] += 1
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) and "role" in item for item in value):
                signals["messages_array"] += 1
            for item in value:
                visit(item)

    visit(row)
    return signals


def scan_file(path: Path, *, batch_size: int, limit: int) -> dict[str, Any]:
    meta = schema_for(path)
    chars = 0
    signals: Counter[str] = Counter()
    observed_keys: Counter[str] = Counter()
    scanned = 0
    for row in iter_rows(path, batch_size):
        if limit and scanned >= limit:
            break
        scanned += 1
        chars += len(canonical(row))
        observed_keys.update(row.keys())
        signals.update(structural_signals(row))
    complete = not limit or scanned < limit
    # For parquet, metadata row count is exact without materializing the file.
    actual_count = meta["metadata_row_count"] if meta["metadata_row_count"] is not None else (scanned if complete else None)
    return {
        "path": str(path),
        "license": source_license_label(path),
        **meta,
        "actual_row_count": actual_count,
        "measurement": {
            "rows_scanned": scanned,
            "complete": complete,
            "average_serialized_chars": (chars / scanned if scanned else 0.0),
            "average_tokens_approx_chars_div_4": (chars / scanned / 4 if scanned else 0.0),
            "length_metric": "canonical JSON serialized character count",
        },
        "observed_top_level_fields": dict(sorted(observed_keys.items())),
        "conversation_and_tool_signals": dict(sorted(signals.items())),
    }


def scan(inputs: list[Path], *, batch_size: int, limit: int) -> dict[str, Any]:
    files = input_files(inputs)
    report_files = [scan_file(path, batch_size=batch_size, limit=limit) for _, path in files]
    groups: dict[str, dict[str, Any]] = {}
    for root, path in files:
        key = str(path.parent.relative_to(root)) if path.parent != root else "."
        group = groups.setdefault(key, {"files": 0, "metadata_rows": 0, "scanned_rows": 0,
                                        "scanned_chars": 0.0, "complete": True})
        file_report = next(item for item in report_files if item["path"] == str(path))
        group["files"] += 1
        group["metadata_rows"] += file_report["actual_row_count"] or 0
        group["scanned_rows"] += file_report["measurement"]["rows_scanned"]
        group["scanned_chars"] += (file_report["measurement"]["average_serialized_chars"] *
                                   file_report["measurement"]["rows_scanned"])
        group["complete"] = group["complete"] and file_report["measurement"]["complete"]
    for group in groups.values():
        n = group["scanned_rows"]
        group["average_serialized_chars"] = group.pop("scanned_chars") / n if n else 0.0
        group["average_tokens_approx_chars_div_4"] = group["average_serialized_chars"] / 4
    return {
        "schema_version": 1,
        "command": "scan",
        "created_at_utc": utc_now(),
        "normalization_note": "No tokenizer used; token approximation is serialized characters / 4.",
        "limit_per_file": limit,
        "files": report_files,
        "groups": groups,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def scan_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Corpus intake scan", "",
        f"Created: `{report['created_at_utc']}`", "",
        f"Per-file row sampling limit: `{report['limit_per_file']}` (0 means complete streaming scan).", "",
        "## Group summary", "",
        "| Group | Files | metadata/exact rows when available | scanned rows | complete | avg chars (n=scanned) | approx tokens |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, item in sorted(report["groups"].items()):
        lines.append("| {} | {} | {} | {} | {} | {:.1f} | {:.1f} |".format(
            name, item["files"], item["metadata_rows"], item["scanned_rows"], item["complete"],
            item["average_serialized_chars"], item["average_tokens_approx_chars_div_4"]))
    lines.extend(["", "## Per-file schema and conversation representation", ""])
    for item in report["files"]:
        measurement = item["measurement"]
        lines.extend([
            f"### `{item['path']}`", "",
            f"- Format/schema: `{item['format']}`; fields: {', '.join(item['observed_top_level_fields'])}",
            f"- Rows: actual={item['actual_row_count']!r}; measured n={measurement['rows_scanned']}; complete={measurement['complete']}.",
            f"- Length: {measurement['average_serialized_chars']:.1f} chars/sample; {measurement['average_tokens_approx_chars_div_4']:.1f} chars/4 token approximation.",
            f"- Conversation/tool signals (n=rows/messages occurrences): `{canonical(item['conversation_and_tool_signals'])}`",
            f"- License manifest: `{canonical(item['license'])}`", "",
        ])
    return "\n".join(lines) + "\n"


def load_json_or_jsonl(path: Path) -> Any:
    """Load arrays/objects and BFCL's line-delimited ``.json`` files."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as array_error:
        rows: list[Any] = []
        try:
            for line in text.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise array_error
        if not rows:
            raise array_error
        return rows


def eval_override_map(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--eval-override must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if name not in EVAL_NAMES:
            raise ValueError(f"unknown eval source {name!r}; expected one of {EVAL_NAMES}")
        out[name] = Path(raw_path)
    return out


def hf_datasets_cache() -> Path:
    return Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache/huggingface/datasets"))


def hf_hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))


def arrow_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
    """Read immutable HF Arrow cache files without datasets' write-side locks."""
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            try:
                reader = pa.ipc.open_stream(source)
            except pa.ArrowInvalid:
                reader = pa.ipc.open_file(source)
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches)) \
                if hasattr(reader, "num_record_batches") else reader
            for batch in batches:
                yield from batch.to_pylist()


def jmmlu_rows() -> tuple[Iterable[dict[str, Any]], list[Path]]:
    """Read the exact JMMLU test CSVs used by eval_harness without extraction."""
    snapshots = hf_hub_cache() / "datasets--nlp-waseda--JMMLU" / "snapshots"
    extracted = sorted(snapshots.glob("*/JMMLU.zip.extracted/JMMLU/test/*.csv"))
    if extracted:
        def from_csv() -> Iterator[dict[str, Any]]:
            for path in extracted:
                with path.open(encoding="utf-8-sig", newline="") as fh:
                    yield from csv.DictReader(fh)
        return from_csv(), extracted
    zips = sorted(snapshots.glob("*/JMMLU.zip"))
    if not zips:
        raise FileNotFoundError("local JMMLU.zip / extracted test CSVs")
    zip_path = zips[-1]
    def from_zip() -> Iterator[dict[str, Any]]:
        with zipfile.ZipFile(zip_path) as archive:
            names = sorted(name for name in archive.namelist()
                           if name.startswith("JMMLU/test/") and name.endswith(".csv")
                           and not Path(name).name.startswith("._"))
            for name in names:
                with archive.open(name) as raw:
                    yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
    return from_zip(), [zip_path]


def local_eval_rows(name: str) -> tuple[Iterable[dict[str, Any]], list[Path], str]:
    """Use the local material named by the current eval/pilot scripts only.

    ``eval_harness`` normally reaches these via ``datasets.load_dataset``.  This
    reader consumes its immutable local Arrow cache directly, avoiding cache
    lock/temp writes and guaranteeing no network fallback in a read-only run.
    """
    cache = hf_datasets_cache()
    if name == "mmlu":
        # eval_harness explicitly requests the aggregate ``all`` config; do
        # not accidentally add separately cached per-subject configurations.
        paths = sorted(cache.glob("cais___mmlu/all/*/*/mmlu-test.arrow"))
        if not paths:
            raise FileNotFoundError("local cais/mmlu all/test Arrow cache")
        return arrow_rows(paths), paths, "esft/eval_harness.py:Mmlu.load -> local cais/mmlu all/test Arrow cache"
    if name == "gsm8k":
        paths = sorted(cache.glob("gsm8k/main/*/*/gsm8k-test.arrow"))
        if not paths:
            # Older datasets cache naming, retained solely as an offline path.
            paths = sorted(cache.glob("openai___gsm8k/main/*/*/gsm8k-test.arrow"))
        if not paths:
            raise FileNotFoundError("local openai/gsm8k main/test Arrow cache")
        return arrow_rows(paths), paths, "esft/eval_harness.py:Gsm8k.load -> local openai/gsm8k main/test Arrow cache"
    if name == "humaneval":
        paths = sorted(cache.glob("openai___openai_humaneval/*/*/*/openai_humaneval-test.arrow"))
        if not paths:
            raise FileNotFoundError("local openai/openai_humaneval test Arrow cache")
        return arrow_rows(paths), paths, "esft/eval_harness.py:HumanEval.load -> local openai/openai_humaneval test Arrow cache"
    if name == "jmmlu":
        rows, paths = jmmlu_rows()
        return rows, paths, "esft/eval_harness.py:Jmmlu.load -> local JMMLU test CSVs"
    if name == "mifeval":
        path = ROOT / "external" / "M-IFEval" / "data" / "ja_input_data.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return rows_from_jsonl(path), [path], "esft/mifeval_pilot.py:INPUT"
    if name == "bfcl":
        root = ROOT / "external" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
        paths = sorted(root.glob("BFCL_v4_*.json"))
        if not paths:
            raise FileNotFoundError(root)
        rows: list[dict[str, Any]] = []
        for path in paths:
            loaded = load_json_or_jsonl(path)
            if isinstance(loaded, list):
                rows.extend(item for item in loaded if isinstance(item, dict))
        return rows, paths, "esft/bfcl_pilot.py:BFCL upstream data path"
    raise AssertionError(name)


@dataclass
class EvalIndex:
    masks: dict[bytes, int]
    statuses: dict[str, dict[str, Any]]

    def match(self, row: dict[str, Any]) -> list[str]:
        mask = 0
        for digest in record_gram_digests(row):
            mask |= self.masks.get(digest, 0)
        return [name for bit, name in enumerate(EVAL_NAMES) if mask & (1 << bit)]


def build_eval_index(overrides: dict[str, Path] | None = None) -> EvalIndex:
    overrides = overrides or {}
    masks: dict[bytes, int] = {}
    statuses: dict[str, dict[str, Any]] = {}
    for bit, name in enumerate(EVAL_NAMES):
        try:
            if name in overrides:
                path = overrides[name]
                rows = load_json_or_jsonl(path)
                if isinstance(rows, dict):
                    rows = [rows]
                if not isinstance(rows, list):
                    raise ValueError("override must contain JSON object(s)")
                source_paths, origin = [path], "explicit --eval-override"
            else:
                rows, source_paths, origin = local_eval_rows(name)
            row_count = 0
            grams = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_count += 1
                for digest in record_gram_digests(row):
                    masks[digest] = masks.get(digest, 0) | (1 << bit)
                    grams += 1
            statuses[name] = {
                "status": "AVAILABLE", "origin": origin, "rows_loaded": row_count,
                "ngram_occurrences": grams,
                "source_sha256": {str(path): sha256_file(path) for path in source_paths if path.is_file()},
            }
        except Exception as exc:  # local cache absence must be explicit, not repaired by network
            statuses[name] = {"status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}
    return EvalIndex(masks=masks, statuses=statuses)


def output_path(output_dir: Path, root: Path, source: Path) -> Path:
    return output_dir / root.name / source.relative_to(root)


def process_jsonl(source: Path, dest: Path, index: EvalIndex, log_fh: Any, batch_size: int) -> dict[str, int]:
    del batch_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    total = kept = removed = 0
    with tmp.open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(rows_from_jsonl(source)):
            total += 1
            matched = index.match(row)
            if matched:
                removed += 1
                log_fh.write(canonical({"source": str(source), "row_index": row_no,
                                        "row_sha256": hashlib.sha256(canonical(row).encode()).hexdigest(),
                                        "matched_eval_sets": matched}) + "\n")
            else:
                kept += 1
                out.write(canonical(row) + "\n")
    os.replace(tmp, dest)
    return {"input_rows": total, "kept_rows": kept, "removed_rows": removed}


def process_parquet(source: Path, dest: Path, index: EvalIndex, log_fh: Any, batch_size: int) -> dict[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    pf = pq.ParquetFile(source)
    writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd")
    total = kept = removed = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            clean: list[dict[str, Any]] = []
            for row in batch.to_pylist():
                matched = index.match(row)
                if matched:
                    removed += 1
                    log_fh.write(canonical({"source": str(source), "row_index": total,
                                            "row_sha256": hashlib.sha256(canonical(row).encode()).hexdigest(),
                                            "matched_eval_sets": matched}) + "\n")
                else:
                    clean.append(row)
                    kept += 1
                total += 1
            if clean:
                writer.write_batch(pa.RecordBatch.from_pylist(clean, schema=pf.schema_arrow))
    finally:
        writer.close()
    os.replace(tmp, dest)
    return {"input_rows": total, "kept_rows": kept, "removed_rows": removed}


def decontam(inputs: list[Path], output_dir: Path, *, batch_size: int,
              overrides: dict[str, Path], allow_no_eval: bool) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix output into non-empty {output_dir}")
    index = build_eval_index(overrides)
    available = [name for name, value in index.statuses.items() if value["status"] == "AVAILABLE"]
    if not available and not allow_no_eval:
        raise RuntimeError("all eval sources are SKIPPED; refusing to create an unverified clean corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "removals.jsonl"
    files: list[dict[str, Any]] = []
    with log_path.open("x", encoding="utf-8") as log_fh:
        for root, source in input_files(inputs):
            dest = output_path(output_dir, root, source)
            if source.suffix.lower() == ".jsonl":
                counts = process_jsonl(source, dest, index, log_fh, batch_size)
            else:
                counts = process_parquet(source, dest, index, log_fh, batch_size)
            files.append({"source": str(source), "clean_output": str(dest),
                          "license": source_license_label(source), **counts})
    return {
        "schema_version": 1,
        "command": "decontam",
        "created_at_utc": utc_now(),
        "method": {
            "match": "exact normalized 8-gram overlap; NFKC/casefold; Latin word tokens and CJK character tokens",
            "ngram_size": NGRAM_SIZE,
            "digest": "BLAKE2b-128 of token sequence (collision risk is negligible but nonzero)",
            "scope": "all textual leaves, including JSON-encoded strings; record removed on any eval-set match",
            "memory": "streaming corpus batches; eval 8-gram digest index retained in RAM",
        },
        "eval_sources": index.statuses,
        "available_eval_sets": available,
        "skipped_eval_sets": [name for name in EVAL_NAMES if name not in available],
        "removal_log": str(log_path),
        "files": files,
        "totals": {key: sum(item[key] for item in files) for key in ("input_rows", "kept_rows", "removed_rows")},
        "clean_output_root": str(output_dir),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan_p = sub.add_parser("scan", help="stream schema/length/conversation inspection")
    dec_p = sub.add_parser("decontam", help="stream exact normalized 8-gram rejection")
    for item in (scan_p, dec_p):
        item.add_argument("--input", action="append", type=Path, default=[], help="corpus file or directory (repeatable)")
        item.add_argument("--batch-size", type=int, default=512)
    scan_p.add_argument("--limit-per-file", type=int, default=0, help="0=complete; otherwise sample rows per file")
    scan_p.add_argument("--output-json", required=True, type=Path)
    scan_p.add_argument("--output-md", required=True, type=Path)
    dec_p.add_argument("--output-dir", required=True, type=Path)
    dec_p.add_argument("--manifest", type=Path, help="default: OUTPUT_DIR/manifest.json")
    dec_p.add_argument("--eval-override", action="append", default=[], metavar="NAME=PATH",
                       help="test/offline override; valid names: " + ", ".join(EVAL_NAMES))
    dec_p.add_argument("--allow-no-eval", action="store_true", help="unsafe: permit output when every eval source is SKIPPED")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.command == "scan" and args.limit_per_file < 0:
        parser.error("--limit-per-file must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    inputs = args.input or list(DEFAULT_INPUTS)
    if args.command == "scan":
        report = scan(inputs, batch_size=args.batch_size, limit=args.limit_per_file)
        write_json(args.output_json, report)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(scan_markdown(report), encoding="utf-8")
        print(canonical({"status": "OK", "json": str(args.output_json), "markdown": str(args.output_md),
                         "files": len(report["files"])}))
        return 0
    manifest = args.manifest or (args.output_dir / "manifest.json")
    report = decontam(inputs, args.output_dir, batch_size=args.batch_size,
                      overrides=eval_override_map(args.eval_override), allow_no_eval=args.allow_no_eval)
    write_json(manifest, report)
    print(canonical({"status": "OK", "manifest": str(manifest), "clean_output_root": str(args.output_dir),
                     "removed_rows": report["totals"]["removed_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
