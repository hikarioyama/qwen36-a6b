#!/usr/bin/env python3
"""Prepare, append, and summarize strict Corpus Judge v1 decisions.

This is deliberately an offline formatter and ledger utility.  ``prepare``
does not call an LLM: it writes one self-contained prompt file per 10--20
record batch for a later cx/Codex job.  That job must emit JSONL decisions, and
``append`` validates and appends them to the durable ``judgements.jsonl``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator


VERDICTS = {"accept", "reject", "borderline"}


class JudgeDataError(ValueError):
    """An input or judgement cannot be safely processed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_json(value: Any, *, expected: type | None = None) -> Any:
    """Decode JSON-in-string fields used by Toucan without over-decoding text."""
    decoded = value
    for _ in range(2):
        if not isinstance(decoded, str):
            break
        text = decoded.strip()
        if not text or text[0] not in "[{":
            break
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            break
    if expected is not None and not isinstance(decoded, expected):
        raise JudgeDataError(f"expected {expected.__name__}, got {type(decoded).__name__}")
    return decoded


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JudgeDataError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise JudgeDataError(f"{path}:{line_no}: record is not an object")
            yield row


def iter_parquet(path: Path, read_batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional reader
        raise JudgeDataError("pyarrow is required for parquet input") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=read_batch_size):
        yield from batch.to_pylist()


def iter_rows(path: Path, read_batch_size: int) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise JudgeDataError(f"input is not a file: {path}")
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif path.suffix == ".parquet":
        yield from iter_parquet(path, read_batch_size)
    else:
        raise JudgeDataError("input must be .jsonl or .parquet")


def record_id(row: dict[str, Any], source: Path, row_index: int) -> str:
    metadata = decode_json(row.get("metadata"), expected=dict) if row.get("metadata") is not None else {}
    for candidate in (row.get("id"), row.get("uuid"), metadata.get("id"), metadata.get("prompt_id")):
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    # This fallback is stable for the same input pathname and source row.
    tag = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    return f"{tag}:{row_index}"


def prompt_record(row: dict[str, Any], source: Path, row_index: int) -> dict[str, Any]:
    """Keep judge-relevant content, excluding pre-existing quality labels."""
    out: dict[str, Any] = {"id": record_id(row, source, row_index)}
    for key in ("subset_name", "question", "target_tools"):
        if row.get(key) not in (None, ""):
            out[key] = row[key]
    if row.get("messages") is not None:
        out["messages"] = decode_json(row["messages"], expected=list)
    if row.get("tools") is not None:
        out["tools"] = decode_json(row["tools"])
    elif row.get("available_tools") is not None:
        out["tools"] = decode_json(row["available_tools"])
    if "messages" not in out and "question" not in out:
        raise JudgeDataError(f"{source}:{row_index}: no messages or question")
    return out


def load_judged(path: Path) -> set[str]:
    if not path.exists():
        return set()
    judged: set[str] = set()
    for row in iter_jsonl(path):
        decision = validate_decision(row)
        if decision["id"] in judged:
            raise JudgeDataError(f"duplicate id in judgement ledger: {decision['id']}")
        judged.add(decision["id"])
    return judged


def batch_id(source: Path, first_index: int, last_index: int, rubric_sha: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in source.stem).strip("-")
    return f"{safe}-{first_index:09d}-{last_index:09d}-{rubric_sha[:12]}"


def render_prompt(rubric: str, rubric_sha: str, identifier: str, records: list[dict[str, Any]]) -> str:
    return f"""# Corpus Judge v1 batch\n\nBatch ID: `{identifier}`\nRubric SHA-256: `{rubric_sha}`\n\nYou are a strict data-quality judge. Apply the following rubric to every record.\nDo not execute tools, browse, or repair records. Do not reward machine validation alone.\nReturn **only JSONL**, exactly one object per supplied record, in the same order:\n`{{\"id\": \"...\", \"verdict\": \"accept|reject|borderline\", \"reason\": \"one-line reason\", \"rubric_sha\": \"{rubric_sha}\", \"batch_id\": \"{identifier}\"}}`\nIf uncertain, use `reject` as required by the rubric.\n\n## Rubric\n\n{rubric.rstrip()}\n\n## Records\n\n{json.dumps(records, ensure_ascii=False, indent=2)}\n"""


def prepare(args: argparse.Namespace) -> int:
    source = Path(args.input)
    rubric_path = Path(args.rubric)
    if not rubric_path.is_file():
        raise JudgeDataError(f"rubric is not a file: {rubric_path}")
    rubric = rubric_path.read_text(encoding="utf-8")
    rubric_sha = sha256_file(rubric_path)
    output = Path(args.output_dir)
    prompts = output / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    judged = load_judged(output / "judgements.jsonl")
    selected: list[tuple[int, dict[str, Any]]] = []
    skipped_judged = 0
    for index, row in enumerate(iter_rows(source, args.read_batch_size)):
        if index < args.start:
            continue
        if args.end is not None and index >= args.end:
            break
        prepared = prompt_record(row, source, index)
        if prepared["id"] in judged:
            skipped_judged += 1
            continue
        selected.append((index, prepared))
    written = 0
    pending: list[str] = []
    manifest_path = output / "batch_manifest.jsonl"
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for entry in iter_jsonl(manifest_path):
            key = str(entry.get("batch_id", ""))
            if not key:
                raise JudgeDataError("batch manifest entry has no batch_id")
            manifest[key] = entry
    new_manifest: list[dict[str, Any]] = []
    for offset in range(0, len(selected), args.batch_size):
        part = selected[offset:offset + args.batch_size]
        identifier = batch_id(source, part[0][0], part[-1][0], rubric_sha)
        path = prompts / f"{identifier}.md"
        content = render_prompt(rubric, rubric_sha, identifier, [record for _, record in part])
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise JudgeDataError(f"refusing to overwrite different existing prompt: {path}")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            written += 1
        pending.append(str(path.relative_to(output)))
        entry = {"batch_id": identifier, "ids": [record["id"] for _, record in part],
                 "rubric_sha": rubric_sha, "prompt_file": str(path.relative_to(output))}
        if identifier in manifest and manifest[identifier] != entry:
            raise JudgeDataError(f"refusing to replace different batch manifest entry: {identifier}")
        if identifier not in manifest:
            new_manifest.append(entry)
    if new_manifest:
        with manifest_path.open("a", encoding="utf-8") as handle:
            for entry in new_manifest:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    # Old prompts are deliberately retained for audit rather than deleted.  This
    # manifest is the authoritative current work queue after a resume.
    (output / "pending_batches.json").write_text(
        json.dumps({"rubric_sha": rubric_sha, "input": str(source), "prompt_files": pending},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rubric_sha": rubric_sha, "selected": len(selected),
                      "skipped_already_judged": skipped_judged, "prompt_files_written": written,
                      "prompt_files_total": len(pending),
                      "pending_manifest": str(output / "pending_batches.json")},
                     ensure_ascii=False, sort_keys=True))
    return 0


def validate_decision(row: dict[str, Any]) -> dict[str, str]:
    required = ("id", "verdict", "reason", "rubric_sha", "batch_id")
    missing = [key for key in required if key not in row]
    if missing:
        raise JudgeDataError(f"judgement missing fields: {', '.join(missing)}")
    decision = {key: str(row[key]).strip() for key in required}
    if not decision["id"] or decision["verdict"] not in VERDICTS:
        raise JudgeDataError("judgement has empty id or invalid verdict")
    if not decision["reason"] or "\n" in decision["reason"] or len(decision["reason"]) > 160:
        raise JudgeDataError("reason must be one non-empty line of at most 160 characters")
    if len(decision["rubric_sha"]) != 64 or any(c not in "0123456789abcdef" for c in decision["rubric_sha"]):
        raise JudgeDataError("rubric_sha must be a lowercase SHA-256 hex digest")
    if not decision["batch_id"]:
        raise JudgeDataError("batch_id is empty")
    return decision


def append(args: argparse.Namespace) -> int:
    incoming = Path(args.input)
    rubric_sha = sha256_file(Path(args.rubric))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "judgements.jsonl"
    already = load_judged(ledger)
    manifest_path = output / "batch_manifest.jsonl"
    if not manifest_path.exists():
        raise JudgeDataError(f"missing batch manifest: run prepare first ({manifest_path})")
    manifest = {str(entry.get("batch_id", "")): entry for entry in iter_jsonl(manifest_path)}
    additions: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in iter_jsonl(incoming):
        decision = validate_decision(row)
        if decision["rubric_sha"] != rubric_sha:
            raise JudgeDataError(f"rubric SHA mismatch for id {decision['id']}")
        batch = manifest.get(decision["batch_id"])
        if not batch or batch.get("rubric_sha") != rubric_sha or decision["id"] not in batch.get("ids", []):
            raise JudgeDataError(f"judgement id/batch is not present in this prepared manifest: {decision['id']}")
        if decision["id"] in already or decision["id"] in seen:
            raise JudgeDataError(f"duplicate or already appended judgement id: {decision['id']}")
        seen.add(decision["id"])
        additions.append(decision)
    with ledger.open("a", encoding="utf-8") as handle:
        for decision in additions:
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"appended": len(additions), "judgements": str(ledger), "rubric_sha": rubric_sha},
                     ensure_ascii=False, sort_keys=True))
    return 0


def summary(args: argparse.Namespace) -> int:
    ledger = Path(args.output_dir) / "judgements.jsonl"
    decisions = [validate_decision(row) for row in iter_jsonl(ledger)] if ledger.exists() else []
    counts = collections.Counter(item["verdict"] for item in decisions)
    reasons = collections.Counter(item["reason"] for item in decisions)
    accepted = counts["accept"]
    total = len(decisions)
    payload = {"n": total, "verdict_counts": {key: counts[key] for key in sorted(VERDICTS)},
               "accept_rate": (accepted / total) if total else None,
               "reason_distribution": [{"reason": reason, "n": count}
                                       for reason, count in reasons.most_common()]}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="write prompt files only; never call a judge")
    prep.add_argument("--input", required=True, help="one selfgen JSONL or Toucan parquet shard")
    prep.add_argument("--rubric", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--start", type=int, default=0, help="zero-based source row, inclusive")
    prep.add_argument("--end", type=int, help="zero-based source row, exclusive")
    prep.add_argument("--batch-size", type=int, default=15, help="must be 10 through 20")
    prep.add_argument("--read-batch-size", type=int, default=256, help="parquet IO batch size")
    prep.set_defaults(func=prepare)
    add = sub.add_parser("append", help="validate cx JSONL and append it to judgements.jsonl")
    add.add_argument("--input", required=True, help="JSONL produced from one or more prompt batches")
    add.add_argument("--rubric", required=True)
    add.add_argument("--output-dir", required=True)
    add.set_defaults(func=append)
    summ = sub.add_parser("summary", help="print verdict and reason counts from the durable ledger")
    summ.add_argument("--output-dir", required=True)
    summ.set_defaults(func=summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare" and (args.start < 0 or args.end is not None and args.end < args.start
                                       or not 10 <= args.batch_size <= 20 or args.read_batch_size <= 0):
        parser.error("require start >= 0, end >= start, batch-size 10..20, and positive read-batch-size")
    try:
        return args.func(args)
    except (JudgeDataError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
