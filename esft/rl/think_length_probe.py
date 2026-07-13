#!/usr/bin/env python3
"""Measure thinking-length variation among correct intent-toolcall rollouts.

This is deliberately a stage-0-only, inference-only probe.  It uses frozen
intent_r1 seeds and the intent v1 OpenAI-compatible vLLM transport, but never
loads a model or starts a server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
ESFT = HERE.parent
sys.path.insert(0, str(ESFT))

import selfgen_toolcall_intent_v1 as intent  # noqa: E402
import selfgen_toolcall_v1 as v1  # noqa: E402


ROLLOUTS_PER_SEED = 8
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 4096
DEFAULT_TIERS = ("T2", "T3")


def parse_tiers(raw: str) -> tuple[str, ...]:
    """Accept a non-empty, non-duplicated subset of the frozen intent tiers."""
    tiers = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not tiers or len(set(tiers)) != len(tiers) or any(tier not in intent.TIERS for tier in tiers):
        raise ValueError("--tiers must be comma-separated unique intent tiers, e.g. T2,T3")
    return tiers


def select_stratified(seeds: Iterable[dict[str, Any]], *, n: int, tiers: tuple[str, ...],
                      sample_seed: int) -> list[dict[str, Any]]:
    """Sample as evenly as possible across tiers, deterministically.

    The first tiers in the user-supplied order receive any one-record remainder.
    """
    if n < 1:
        raise ValueError("--n must be positive")
    grouped = {tier: [seed for seed in seeds if seed.get("tier") == tier] for tier in tiers}
    base, remainder = divmod(n, len(tiers))
    wanted = {tier: base + (index < remainder) for index, tier in enumerate(tiers)}
    for tier, count in wanted.items():
        if len(grouped[tier]) < count:
            raise ValueError(f"frozen seeds contain only {len(grouped[tier])} {tier} records; need {count}")
    rng = random.Random(sample_seed)
    selected: list[dict[str, Any]] = []
    for tier in tiers:
        selected.extend(rng.sample(grouped[tier], wanted[tier]))
    return selected


def split_thinking(text: str) -> tuple[int, bool, str]:
    """Return character count, closed flag, and payload after a think block.

    No tokenizer is used: this measures literal Python character count.  A
    completion without an opening tag is parsed as-is; an unclosed block has no
    valid post-think payload and consequently will normally fail validation.
    """
    opening = text.find("<think>")
    if opening < 0:
        return 0, False, text
    content_start = opening + len("<think>")
    closing = text.find("</think>", content_start)
    if closing < 0:
        return len(text) - content_start, False, text
    return closing - content_start, True, text[closing + len("</think>"):]


def score_rollout(seed: dict[str, Any], rollout_idx: int, raw: str) -> dict[str, Any]:
    """Score one stage-0 completion through the unchanged v1 intent selector."""
    think_chars, think_closed, payload = split_thinking(raw)
    tools = {tool["name"]: tool for tool in seed["tools"]}
    calls, _reasons, _chosen = intent.select_candidate([payload], seed["expected_stages"][0], tools)
    return {"seed_id": seed["seed_id"], "tier": seed["tier"], "rollout_idx": rollout_idx,
            "think_chars": think_chars, "think_closed": think_closed, "total_chars": len(raw),
            "pass": calls is not None}


def percentile(values: list[float | int], fraction: float) -> float | None:
    """Linear-interpolated percentile (the common Hyndman--Fan type-7 rule)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    # JSON summaries are decision records; avoid exposing binary floating-point
    # noise such as 76.00000000000001 for an exactly interpretable percentile.
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 12)


def distribution(values: list[float | int]) -> dict[str, Any]:
    return {"count": len(values), "p10": percentile(values, 0.10),
            "p50": percentile(values, 0.50), "p90": percentile(values, 0.90)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce the direct ThinkingCap premise checks from rollout-level rows."""
    passed = [row for row in rows if row["pass"]]
    by_seed: dict[str, list[int]] = {}
    for row in passed:
        by_seed.setdefault(row["seed_id"], []).append(int(row["think_chars"]))

    ratios: list[float] = []
    pruneable = 0
    eligible = 0
    for lengths in by_seed.values():
        if len(lengths) < 2:
            continue
        eligible += 1
        minimum, maximum = min(lengths), max(lengths)
        ratios.append(1.0 if maximum == 0 else minimum / maximum)
        if minimum < percentile(lengths, 0.50) * 0.5:  # type is non-None for non-empty lengths
            pruneable += 1

    return {
        "rollouts": len(rows),
        "passes": len(passed),
        "pass_rate": (len(passed) / len(rows)) if rows else None,
        "correct_think_chars": distribution([int(row["think_chars"]) for row in passed]),
        "seed_correct_min_max_ratio": distribution(ratios),
        "direct_pruning_opportunity": {
            "eligible_seeds": eligible,
            "seeds_with_shortest_below_half_median": pruneable,
            "rate": (pruneable / eligible) if eligible else None,
        },
    }


def _require_thinking_environment() -> None:
    """Keep v1's stage hint while refusing its opt-in no-think prompt suffix."""
    if os.environ.get("SELFGEN_NOTHINK") == "1":
        raise ValueError("think-length probe requires thinking on; unset SELFGEN_NOTHINK")
    stage_hint = os.environ.get("SELFGEN_STAGE_HINT")
    if stage_hint not in (None, "", "1"):
        raise ValueError("think-length probe requires SELFGEN_STAGE_HINT=1")
    os.environ["SELFGEN_STAGE_HINT"] = "1"


def _client_for_endpoint(endpoint: str, *, transport: Callable[..., Any], sleep: Callable[[float], None]) -> intent._VLLMClient:
    return intent._VLLMClient(endpoint, transport=transport, sleep=sleep)


def run_probe(*, run_id: str, n: int, tiers: tuple[str, ...], sample_seed: int, endpoints: str,
              transport: Callable[..., Any] = urlopen, sleep: Callable[[float], None] = __import__("time").sleep
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate and score one fixed stage-0 group per sampled seed."""
    _require_thinking_environment()
    _target, _manifest, frozen = intent._load_frozen_seeds(run_id)
    seeds = select_stratified(frozen["seeds"], n=n, tiers=tiers, sample_seed=sample_seed)
    endpoint_urls = intent.parse_vllm_endpoints(endpoints)
    clients = [_client_for_endpoint(endpoint, transport=transport, sleep=sleep) for endpoint in endpoint_urls]
    spec = v1.GenerationSpec(sample_seed, TEMPERATURE, MAX_TOKENS, ROLLOUTS_PER_SEED, 0)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        # Stage 0 has no preceding tool results.  v1.scaffold_prompt reads the
        # required stage-hint environment itself, preserving its exact prompt.
        prompt = v1.scaffold_prompt(seed, 0, [])
        raws = clients[index % len(clients)].generate(prompt, spec)
        rows.extend(score_rollout(seed, rollout_idx, raw) for rollout_idx, raw in enumerate(raws))
    summary = summarize(rows)
    summary["protocol"] = {"run_id": run_id, "sample_seed": sample_seed, "sample_size": n,
                           "tiers": list(tiers),
                           "sample_tier_counts": {tier: sum(seed["tier"] == tier for seed in seeds)
                                                  for tier in tiers},
                           "stage": 0, "rollouts_per_seed": ROLLOUTS_PER_SEED,
                           "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
                           "thinking": "on", "stage_hint": 1, "endpoints": endpoint_urls}
    return rows, summary


def _default_output(run_id: str, n: int, sample_seed: int, tiers: tuple[str, ...]) -> Path:
    target, _manifest, _frozen = intent._load_frozen_seeds(run_id)
    return target / f"think_length_probe_n{n}_{'-'.join(tiers)}_seed{sample_seed}.jsonl"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", default="intent_r1")
    result.add_argument("--n", type=int, default=200, help="total stratified seed count")
    result.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    result.add_argument("--sample-seed", type=int, default=20260714)
    result.add_argument("--endpoints", required=True, help="two comma-separated vLLM HTTP endpoints")
    result.add_argument("--output", help="rollout JSONL output path")
    result.add_argument("--summary-output", help="summary JSON output path")
    result.add_argument("--overwrite", action="store_true", help="replace existing explicitly named outputs")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    tiers = parse_tiers(args.tiers)
    output = Path(args.output) if args.output else _default_output(args.run_id, args.n, args.sample_seed, tiers)
    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    if not args.overwrite and (output.exists() or summary_output.exists()):
        raise FileExistsError("probe output already exists; choose new paths or pass --overwrite")
    rows, summary = run_probe(run_id=args.run_id, n=args.n, tiers=tiers, sample_seed=args.sample_seed,
                              endpoints=args.endpoints)
    v1.atomic_jsonl(output, rows)
    v1.atomic_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
