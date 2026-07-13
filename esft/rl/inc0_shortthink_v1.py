#!/usr/bin/env python3
"""Build INC-0 short-thinking SFT data from frozen intent self-generation seeds.

This is a stage-0-only rejection sampler.  It deliberately reuses the
think-length probe's vLLM prompt and validator: for each seed it retains the
shortest *closed* ``<think>`` block among correct rollouts, then emits the raw
assistant completion (including that block) in the existing trainer JSONL
format.  The append-only rollout checkpoint is the resume authority; trainer
and summary outputs are rebuilt from it after every invocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
ESFT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ESFT))

import corpus_to_trainer_v1 as converter  # noqa: E402
import selfgen_toolcall_intent_v1 as intent  # noqa: E402
import selfgen_toolcall_v1 as v1  # noqa: E402
from think_length_probe import (  # noqa: E402
    MAX_TOKENS,
    ROLLOUTS_PER_SEED,
    TEMPERATURE,
    TOP_P,
    _client_for_endpoint,
    _require_thinking_environment,
    distribution,
    parse_tiers,
    score_rollout,
    select_stratified,
)


SOURCE_TAG = "inc0_shortthink_20260714"
DOMAIN = "toolcall"
DEFAULT_RUN_IDS = ("intent_r1", "intent_r2", "intent_r3")
DEFAULT_TIERS = ("T1", "T2", "T3", "T4")


def parse_run_ids(raw: str) -> tuple[str, ...]:
    """Parse a non-empty, unique list of frozen intent run IDs."""
    run_ids = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("--run-ids must be a non-empty comma-separated list without duplicates")
    # Let the canonical loader enforce the filename safety rule and existence.
    return run_ids


def seed_key(run_id: str, seed: dict[str, Any]) -> str:
    """Disambiguate identically named frozen seeds from separate intent runs."""
    seed_id = seed.get("seed_id")
    if not isinstance(seed_id, str) or not seed_id:
        raise ValueError("frozen seed has no non-empty seed_id")
    return f"{run_id}:{seed_id}"


def select_shortest_closed(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the shortest correct closed-think rollout, breaking ties by index."""
    eligible = [row for row in rows if row.get("pass") and row.get("think_closed")]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (int(row["think_chars"]), int(row["rollout_idx"])))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Durably commit one completed seed so an interrupted run can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(intent.canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rollout_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load an append-only checkpoint and reject malformed or duplicate seeds."""
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = row["seed_key"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid rollout checkpoint {path}:{line_no}") from exc
            if not isinstance(key, str) or key in records:
                raise ValueError(f"duplicate or invalid seed_key in {path}:{line_no}")
            if not isinstance(row.get("rollouts"), list) or not isinstance(row.get("seed"), dict):
                raise ValueError(f"invalid rollout record structure in {path}:{line_no}")
            records[key] = row
    return records


def make_trainer_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one selected self-generation completion to the canonical trainer shape."""
    selected = record.get("selected")
    if not isinstance(selected, dict):
        return None
    raw = selected.get("raw")
    seed = record.get("seed")
    if not isinstance(raw, str) or not isinstance(seed, dict):
        raise ValueError(f"selected record {record.get('seed_key')!r} is malformed")
    source = {
        "tools": seed.get("tools"),
        "messages": [
            {"role": "user", "content": seed.get("user_request", "")},
            # Keep the selected model text verbatim: INC-0 teaches the reasoning
            # channel and its tool-call JSON together.
            {"role": "assistant", "content": raw},
        ],
    }
    return converter.to_trainer_record(source, source_tag=SOURCE_TAG, domain=DOMAIN)


def summarize(records: Iterable[dict[str, Any]], *, tiers: tuple[str, ...]) -> dict[str, Any]:
    """Summarize selection coverage and the retained thinking-length distribution."""
    materialized = list(records)
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        rows = [row for row in materialized if row.get("tier") == tier]
        rollouts = [rollout for row in rows for rollout in row["rollouts"]]
        passed = [rollout for rollout in rollouts if rollout.get("pass")]
        eligible = [rollout for rollout in passed if rollout.get("think_closed")]
        selected = [row["selected"] for row in rows if isinstance(row.get("selected"), dict)]
        chars = [int(row["think_chars"]) for row in selected]
        by_tier[tier] = {
            "seeds": len(rows),
            "adopted": len(selected),
            "adoption_rate": len(selected) / len(rows) if rows else None,
            "rollouts": len(rollouts),
            "passes": len(passed),
            "pass_rate": len(passed) / len(rollouts) if rollouts else None,
            "closed_passes": len(eligible),
            "closed_pass_rate": len(eligible) / len(rollouts) if rollouts else None,
            "adopted_think_chars": distribution(chars),
            "adopted_zero_think_rate": (sum(value == 0 for value in chars) / len(chars)) if chars else None,
        }
    selected = [row["selected"] for row in materialized if isinstance(row.get("selected"), dict)]
    return {
        "seeds": len(materialized),
        "adopted": len(selected),
        "by_tier": by_tier,
        "all_tiers": {
            "adopted_think_chars": distribution([int(row["think_chars"]) for row in selected]),
            "adopted_zero_think_rate": (sum(int(row["think_chars"]) == 0 for row in selected) / len(selected))
            if selected else None,
        },
    }


def sample_frozen_seeds(run_ids: tuple[str, ...], tiers: tuple[str, ...], n_per_tier: int,
                        sample_seed: int) -> list[tuple[str, dict[str, Any]]]:
    """Stratify once over the requested runs, retaining exactly ``n_per_tier`` per tier."""
    pooled: list[dict[str, Any]] = []
    for run_id in run_ids:
        _target, _manifest, frozen = intent._load_frozen_seeds(run_id)
        pooled.extend({"run_id": run_id, "seed": seed, "tier": seed.get("tier")}
                      for seed in frozen["seeds"])
    selected = select_stratified(pooled, n=n_per_tier * len(tiers), tiers=tiers,
                                 sample_seed=sample_seed)
    return [(item["run_id"], item["seed"]) for item in selected]


def run_campaign(*, run_ids: tuple[str, ...], tiers: tuple[str, ...], n_per_tier: int,
                 sample_seed: int, endpoints: str, rollout_output: Path,
                 transport: Callable[..., Any] = urlopen,
                 sleep: Callable[[float], None] = __import__("time").sleep) -> list[dict[str, Any]]:
    """Generate missing seed groups and append their scored, selected checkpoint rows."""
    if n_per_tier < 1:
        raise ValueError("--n-per-tier must be positive")
    _require_thinking_environment()
    sampled = sample_frozen_seeds(run_ids, tiers, n_per_tier, sample_seed)
    keys = [seed_key(run_id, seed) for run_id, seed in sampled]
    if len(keys) != len(set(keys)):
        raise ValueError("sampled run/seed IDs are not unique")
    completed = load_rollout_records(rollout_output)
    endpoint_urls = intent.parse_vllm_endpoints(endpoints)
    pending = [(run_id, seed) for run_id, seed in sampled if seed_key(run_id, seed) not in completed]
    if pending:
        clients = [_client_for_endpoint(endpoint, transport=transport, sleep=sleep) for endpoint in endpoint_urls]
        spec = v1.GenerationSpec(sample_seed, TEMPERATURE, MAX_TOKENS, ROLLOUTS_PER_SEED, 0)
        for index, (run_id, seed) in enumerate(pending):
            prompt = v1.scaffold_prompt(seed, 0, [])
            raws = clients[index % len(clients)].generate(prompt, spec)
            scored = [score_rollout(seed, rollout_idx, raw) for rollout_idx, raw in enumerate(raws)]
            selected = select_shortest_closed(scored)
            selected_with_raw = None if selected is None else {**selected, "raw": raws[int(selected["rollout_idx"])]}
            record = {"seed_key": seed_key(run_id, seed), "run_id": run_id, "seed_id": seed["seed_id"],
                      "tier": seed["tier"], "seed": seed, "rollouts": scored,
                      "selected": selected_with_raw}
            append_jsonl(rollout_output, record)
            completed[record["seed_key"]] = record
    # Only the requested sample is eligible for the derived corpus/summary.
    return [completed[key] for key in keys]


def _default_outputs(run_ids: tuple[str, ...], tiers: tuple[str, ...], n_per_tier: int,
                     sample_seed: int) -> tuple[Path, Path, Path]:
    stem = f"inc0_shortthink_{'-'.join(run_ids)}_n{n_per_tier}_{'-'.join(tiers)}_seed{sample_seed}"
    root = ESFT / "data" / "inc0_shortthink_v1"
    return root / f"{stem}.rollouts.jsonl", root / f"{stem}.train.jsonl", root / f"{stem}.summary.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-ids", default=",".join(DEFAULT_RUN_IDS))
    result.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    result.add_argument("--n-per-tier", type=int, default=250)
    result.add_argument("--sample-seed", type=int, default=20260714)
    result.add_argument("--endpoints", required=True, help="two comma-separated vLLM HTTP endpoints")
    result.add_argument("--rollout-output", help="append-only, resume-safe JSONL checkpoint")
    result.add_argument("--trainer-output", help="derived trainer JSONL")
    result.add_argument("--summary-output", help="derived summary JSON")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_ids, tiers = parse_run_ids(args.run_ids), parse_tiers(args.tiers)
    defaults = _default_outputs(run_ids, tiers, args.n_per_tier, args.sample_seed)
    rollout_output = Path(args.rollout_output) if args.rollout_output else defaults[0]
    trainer_output = Path(args.trainer_output) if args.trainer_output else defaults[1]
    summary_output = Path(args.summary_output) if args.summary_output else defaults[2]
    records = run_campaign(run_ids=run_ids, tiers=tiers, n_per_tier=args.n_per_tier,
                           sample_seed=args.sample_seed, endpoints=args.endpoints,
                           rollout_output=rollout_output)
    trainer_rows = [row for record in records if (row := make_trainer_record(record)) is not None]
    v1.atomic_jsonl(trainer_output, trainer_rows)
    summary = summarize(records, tiers=tiers)
    summary["protocol"] = {"run_ids": list(run_ids), "tiers": list(tiers), "n_per_tier": args.n_per_tier,
                           "sample_seed": args.sample_seed, "stage": 0,
                           "rollouts_per_seed": ROLLOUTS_PER_SEED, "temperature": TEMPERATURE,
                           "top_p": TOP_P, "max_tokens": MAX_TOKENS, "thinking": "on",
                           "source_tag": SOURCE_TAG, "domain": DOMAIN,
                           "lineage": "A-clean self-generation only; no external text"}
    v1.atomic_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
