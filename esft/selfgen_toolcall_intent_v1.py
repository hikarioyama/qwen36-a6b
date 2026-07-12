#!/usr/bin/env python3
"""CPU-first intent-level fork of :mod:`selfgen_toolcall_v1`.

The v1 generator remains the compatibility oracle.  This module only changes
seed presentation (and adds optional distractor schemas); its expected traces,
JSON/schema validation, mock executor, and best-of-four selection contract are
kept machine-verifiable.  ``prepare``, the paraphrase/audit batch commands, and
their tests are CPU-only.  ``execute`` retains v1's separately preflighted
two-GPU production path, but is deliberately not invoked by this implementation
task.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import sys
from collections import Counter
from typing import Any, Iterable

import selfgen_toolcall_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
ESFT = ROOT / "esft"
OUT_ROOT = ESFT / "data" / "selfgen_toolcall_intent_v1"
TIERS = ("T1", "T2", "T3", "T4")
DEFAULT_TIER_MIX = "T1:0.1,T2:0.4,T3:0.3,T4:0.2"
MAX_RENDER_ATTEMPTS = 8

# Public aliases make the validator contract explicit and allow existing CPU
# fixtures to be reused without copying the v1 implementation.
canonical = v1.canonical
validate_call = v1.validate_call
parse_model_turn = v1.parse_model_turn
select_candidate = v1.select_candidate
render_training = v1.render_training
contaminated = v1.contaminated
GenerationSpec = v1.GenerationSpec


def atomic_json(path: Path, value: Any) -> None:
    v1.atomic_json(path, value)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    v1.atomic_jsonl(path, rows)


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be safe filename characters")
    return OUT_ROOT / run_id


def parse_tier_mix(text: str) -> dict[str, float]:
    """Parse a complete probability mix and reject silent tier omissions."""
    values: dict[str, float] = {}
    for item in text.split(","):
        if ":" not in item:
            raise ValueError("tier mix must use T1:weight,...")
        tier, raw = (part.strip() for part in item.split(":", 1))
        if tier not in TIERS or tier in values:
            raise ValueError(f"invalid or duplicate tier {tier!r}")
        try:
            weight = float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid weight for {tier}: {raw!r}") from exc
        if weight < 0:
            raise ValueError("tier weights must be non-negative")
        values[tier] = weight
    if set(values) != set(TIERS):
        raise ValueError(f"tier mix must name exactly {', '.join(TIERS)}")
    if abs(sum(values.values()) - 1.0) > 1e-9:
        raise ValueError("tier weights must sum to 1.0")
    return values


def tier_counts(count: int, mix: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment gives exact, inspectable small-n mixes."""
    raw = {tier: count * mix[tier] for tier in TIERS}
    result = {tier: int(raw[tier]) for tier in TIERS}
    remainder = count - sum(result.values())
    order = sorted(TIERS, key=lambda tier: (-(raw[tier] - result[tier]), TIERS.index(tier)))
    for tier in order[:remainder]:
        result[tier] += 1
    return result


def _literal(value: Any) -> str:
    """A value spelling that round-trips through ``value_occurrences`` exactly."""
    return canonical(value)


def _surface_values(seed: dict[str, Any]) -> list[Any]:
    """Values that must appear in an initial user request.

    Result-derived arguments (receipts and recovery codes) are intentionally
    excluded: the request names their source, and their actual values do not
    exist until a prior tool turn.  This distinction is recorded in the seed so
    a paraphraser cannot mistake it for an omission.
    """
    derived = {(item["stage"], item["field"]) for item in seed.get("derived_values", [])}
    values: list[Any] = []
    for stage_no, stage in enumerate(seed["expected_stages"]):
        for call in stage:
            for field, value in call["arguments"].items():
                if (stage_no, field) not in derived:
                    values.append(value)
    # Preserve order but do not burden a paraphraser with duplicated literals.
    out, seen = [], set()
    for value in values:
        key = canonical(value)
        if key not in seen:
            out.append(value)
            seen.add(key)
    return out


def value_occurrences(request: str, values: Iterable[Any]) -> list[str]:
    """Return canonical literals missing from request text (empty means pass)."""
    if not isinstance(request, str):
        return ["<request-not-string>"]
    return [_literal(value) for value in values if _literal(value) not in request]


def validate_value_occurrences(seed: dict[str, Any], request: str | None = None) -> tuple[bool, list[str]]:
    request = seed.get("natural_request") if request is None else request
    return (not (missing := value_occurrences(request, seed.get("request_values", _surface_values(seed)))), missing)


def _intent_request(seed: dict[str, Any]) -> str:
    """Template rendering which never leaks a tool or argument identifier."""
    groups: list[list[str]] = []
    derived = {(item["stage"], item["field"]) for item in seed.get("derived_values", [])}
    for stage_no, stage in enumerate(seed["expected_stages"]):
        literals = []
        for call in stage:
            literals.extend(_literal(value) for field, value in call["arguments"].items()
                            if (stage_no, field) not in derived)
        groups.append(literals)
    def phrase(index: int) -> str:
        return ", ".join(groups[index]) if groups[index] else "the result from the previous step"
    pattern = seed["pattern"]
    domain = seed["domain"]
    if pattern == "single":
        return f"Please complete this {domain} request using these details: {phrase(0)}."
    if pattern == "parallel":
        return (f"For this {domain} request, handle these two independent needs in parallel: "
                f"one with {phrase(0)}. Return both results.")
    if pattern == "multi_turn":
        return (f"For this {domain} request, begin with {phrase(0)}. After that result arrives, continue "
                f"with {phrase(1)} and use the receipt returned by the first step.")
    if pattern == "error_recovery":
        return (f"For this {domain} request, first handle {phrase(0)}. If it cannot be completed, recover "
                f"with {phrase(1)} using the error code returned by that attempt.")
    if pattern == "long_chain":
        return (f"For this {domain} request, start the two independent preparations in parallel using "
                f"{phrase(0)}. Once both receipts are available, continue with {phrase(1)}. Then attempt "
                f"{phrase(2)}; if that step reports an error, recover with {phrase(3)} using the reported "
                "error code and the earlier receipt.")
    raise ValueError(f"unknown pattern {pattern!r}")


def _distractor(correct: dict[str, Any], ordinal: int) -> dict[str, Any]:
    """Near-name/schema tool that cannot accept the corresponding expected call."""
    out = copy.deepcopy(correct)
    out["name"] = correct["name"] + f"_assistant_{ordinal + 1}"
    props = out["parameters"]["properties"]
    required = out["parameters"]["required"]
    original = required[0]
    replacement = original + "_routing"
    props[replacement] = props.pop(original)
    required[0] = replacement
    out["description"] = ("Related offline mock operation, but it requires a distinct routing field; "
                          "do not substitute it for the requested operation.")
    return out


def _add_distractors(seed: dict[str, Any], rng: random.Random) -> None:
    count = 3 + rng.randrange(3)
    correct = list(seed["tools"])
    distractors = [_distractor(correct[index % len(correct)], index) for index in range(count)]
    seed["tools"].extend(distractors)
    seed["distractor_tools"] = distractors


def _long_chain_seed(seed_id: int, rng: random.Random) -> dict[str, Any]:
    schema_id = seed_id % v1.SCHEMA_COUNT
    domain = v1.DOMAINS[schema_id % len(v1.DOMAINS)]
    tools = [v1.build_tool(domain, schema_id, ordinal, rng) for ordinal in range(5)]
    first_a, first_b = v1.expected_call(tools[0], seed_id), v1.expected_call(tools[1], seed_id + 11)
    receipt_a = mock_execute(first_a, 0, "long_chain")["result"]["receipt"]
    receipt_b = mock_execute(first_b, 0, "long_chain")["result"]["receipt"]
    v1.require_link_field(tools[2], "first_receipt", {"type": "string"})
    v1.require_link_field(tools[2], "second_receipt", {"type": "string"})
    second = v1.expected_call(tools[2], seed_id + 19)
    second["arguments"].update({"first_receipt": receipt_a, "second_receipt": receipt_b})
    receipt_second = mock_execute(second, 1, "long_chain")["result"]["receipt"]
    v1.require_link_field(tools[3], "prior_receipt", {"type": "string"})
    third = v1.expected_call(tools[3], seed_id + 29)
    third["arguments"]["prior_receipt"] = receipt_second
    v1.require_link_field(tools[4], "recovery_code", {"type": "string", "enum": ["UNAVAILABLE"]})
    v1.require_link_field(tools[4], "prior_receipt", {"type": "string"})
    fourth = v1.expected_call(tools[4], seed_id + 37)
    fourth["arguments"].update({"recovery_code": "UNAVAILABLE", "prior_receipt": receipt_second})
    transcription = (
        f"For this {domain} request, independently and in parallel, "
        f"{v1.call_request_fragment(first_a)}; also {v1.call_request_fragment(first_b)}. "
        f"After both results, {v1.call_request_fragment(second, omit={'first_receipt', 'second_receipt'})} "
        "and set the receipt fields to the two returned receipts. Then "
        f"{v1.call_request_fragment(third, omit={'prior_receipt'})} and set prior_receipt to the prior result. "
        f"If that returns an error, recover by {v1.call_request_fragment(fourth, omit={'recovery_code', 'prior_receipt'})} "
        "using its error code and the earlier receipt."
    )
    return {
        "seed_id": f"seed-{seed_id:04d}", "schema_id": f"schema-{schema_id:03d}", "domain": domain,
        "pattern": "long_chain", "tools": tools, "user_request": transcription,
        "expected_stages": [[first_a, first_b], [second], [third], [fourth]],
        "derived_values": [
            {"stage": 1, "field": "first_receipt", "source": "stage0.call0.result.receipt"},
            {"stage": 1, "field": "second_receipt", "source": "stage0.call1.result.receipt"},
            {"stage": 2, "field": "prior_receipt", "source": "stage1.call0.result.receipt"},
            {"stage": 3, "field": "recovery_code", "source": "stage2.call0.error.code"},
            {"stage": 3, "field": "prior_receipt", "source": "stage1.call0.result.receipt"},
        ],
    }


def _decorate(seed: dict[str, Any], tier: str, rng: random.Random) -> dict[str, Any]:
    seed = copy.deepcopy(seed)
    seed["tier"] = tier
    seed["distractor_tools"] = []
    seed.setdefault("derived_values", [])
    transcription = seed["user_request"]
    seed["transcription_request"] = transcription
    seed["request_values"] = _surface_values(seed)
    if tier == "T1":
        seed["natural_request"] = transcription
        return seed
    seed["natural_request"] = _intent_request(seed)
    seed["user_request"] = seed["natural_request"]
    if tier in {"T3", "T4"}:
        _add_distractors(seed, rng)
    return seed


def make_seed(seed_id: int, rng: random.Random, tier: str = "T1", *, max_attempts: int = MAX_RENDER_ATTEMPTS) -> dict[str, Any]:
    """Create one tiered seed; failed intent renderings are boundedly regenerated."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for attempt in range(max_attempts):
        base = _long_chain_seed(seed_id, rng) if tier == "T4" else v1.make_seed(seed_id, rng)
        seed = _decorate(base, tier, rng)
        ok, missing = validate_value_occurrences(seed)
        if tier == "T1" or ok:
            seed["render_attempt"] = attempt + 1
            return seed
    raise RuntimeError(f"intent rendering failed value-occurrence check after {max_attempts} attempts: {missing}")


def build_seeds(count: int, rng_seed: int, profile: dict[str, Any], eval_grams: set[tuple[str, ...]],
                eval_names: set[str], tier_mix: dict[str, float]) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(rng_seed)
    requested = tier_counts(count, tier_mix)
    schedule = [tier for tier in TIERS for _ in range(requested[tier])]
    rng.shuffle(schedule)
    seeds, rejected = [], Counter()
    candidate, schedule_index = 0, 0
    while len(seeds) < count:
        tier = schedule[schedule_index]
        seed = make_seed(candidate, rng, tier)
        reason = contaminated(seed, eval_grams, eval_names)
        if reason:
            rejected[reason] += 1
        else:
            seed["structure_reference"] = {
                "bfcl_function_count": profile.get("function_count", 0),
                "bfcl_arity_histogram": profile.get("arity_histogram", {}),
            }
            seeds.append(seed)
            schedule_index += 1
        candidate += 1
        if candidate > count * 20:
            raise RuntimeError("contamination filter excluded too many generated seeds")
    return seeds, rejected


def mock_execute(call: dict[str, Any], stage: int, pattern: str) -> dict[str, Any]:
    if pattern == "long_chain" and stage == 2:
        return {"ok": False, "error": {"code": "UNAVAILABLE", "retryable": True}, "call": call["name"]}
    return v1.mock_execute(call, stage, pattern)


def _long_chain_links_match(seed: dict[str, Any], selected: list[list[dict[str, Any]]],
                            results: list[list[dict[str, Any]]]) -> bool:
    if seed["pattern"] != "long_chain":
        return True
    return (
        selected[1][0]["arguments"].get("first_receipt") == results[0][0]["result"]["receipt"] and
        selected[1][0]["arguments"].get("second_receipt") == results[0][1]["result"]["receipt"] and
        selected[2][0]["arguments"].get("prior_receipt") == results[1][0]["result"]["receipt"] and
        selected[3][0]["arguments"].get("recovery_code") == results[2][0]["error"]["code"] and
        selected[3][0]["arguments"].get("prior_receipt") == results[1][0]["result"]["receipt"]
    )


def cpu_fixture_records(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        selected = copy.deepcopy(seed["expected_stages"])
        results = [[mock_execute(call, stage, seed["pattern"]) for call in calls]
                   for stage, calls in enumerate(selected)]
        records.append({"seed": seed, "selected": selected, "results": results,
                        "selection": [{"stage": i, "candidate_index": 0, "candidate_rejections": []}
                                      for i in range(len(selected))], "failures": []})
    return records


def evaluate_records(records: list[dict[str, Any]], eval_grams: set[tuple[str, ...]],
                     eval_names: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    accepted, rejected, reasons = [], [], Counter()
    for record in records:
        seed, failure = record["seed"], contaminated(record["seed"], eval_grams, eval_names)
        if not failure and record["failures"]:
            failure = record["failures"][0]
        tools = {tool["name"]: tool for tool in seed["tools"]}
        if not failure and len(record["selected"]) != len(seed["expected_stages"]):
            failure = "incomplete_dialogue"
        if not failure:
            for stage, (calls, expected, results) in enumerate(zip(
                    record["selected"], seed["expected_stages"], record["results"])):
                if canonical(calls) != canonical(expected):
                    failure = "plan_alignment"
                    break
                if len(calls) != len(results):
                    failure = "executor_cardinality"
                    break
                for call, result in zip(calls, results):
                    error, _ = validate_call(call, tools)
                    if error:
                        failure = error
                        break
                    if result != mock_execute(call, stage, seed["pattern"]):
                        failure = "executor_inconsistent"
                        break
                if failure:
                    break
            if not failure and not _long_chain_links_match(seed, record["selected"], record["results"]):
                failure = "long_chain_result_mismatch"
        meta = {"id": seed["seed_id"], "domain": seed["domain"], "pattern": seed["pattern"],
                "tier": seed["tier"], "schema_sha256": hashlib.sha256(canonical(seed["tools"]).encode()).hexdigest(),
                "best_of": 4, "selection": record["selection"],
                "validator": {"json": True, "schema": not failure, "executor": not failure,
                              "plan_alignment": not failure, "contamination": not failure}}
        if failure:
            reasons[failure] += 1
            rejected.append({"id": seed["seed_id"], "reason": failure, "seed": seed,
                             "selection": record["selection"]})
        else:
            accepted.append({**render_training(seed, record["selected"], record["results"]), "metadata": meta})
    return accepted, rejected, reasons


def _load_frozen_seeds(run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    target = run_dir(run_id)
    manifest_path, seeds_path = target / "manifest.json", target / "seeds.json"
    if not manifest_path.is_file() or not seeds_path.is_file():
        raise FileNotFoundError("run must be prepared first")
    return target, json.loads(manifest_path.read_text(encoding="utf-8")), json.loads(seeds_path.read_text(encoding="utf-8"))


def emit_paraphrase_batch(args: argparse.Namespace) -> None:
    target, _, frozen = _load_frozen_seeds(args.run_id)
    output = Path(args.output) if args.output else target / "paraphrase_batch.jsonl"
    rows = []
    for seed in frozen["seeds"]:
        if seed["tier"] == "T1":
            continue
        rows.append({"schema_version": 1, "seed_id": seed["seed_id"], "tier": seed["tier"],
                     "transcription_request": seed["transcription_request"],
                     "value_literals": [_literal(value) for value in seed["request_values"]],
                     "derived_values": seed.get("derived_values", []),
                     "writeback": {"seed_id": seed["seed_id"], "natural_request": "<paraphrased request>"}})
    atomic_jsonl(output, rows)
    print(output)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def ingest_paraphrase(args: argparse.Namespace) -> None:
    target, manifest, frozen = _load_frozen_seeds(args.run_id)
    if manifest.get("state") != "prepared":
        raise RuntimeError(f"run state must be prepared, got {manifest.get('state')!r}")
    writebacks: dict[str, str] = {}
    known = {seed["seed_id"] for seed in frozen["seeds"]}
    for row in _read_jsonl(Path(args.input)):
        seed_id, natural = row.get("seed_id"), row.get("natural_request")
        if seed_id not in known or seed_id in writebacks or not isinstance(natural, str):
            raise ValueError("writeback rows require unique known seed_id and string natural_request")
        writebacks[seed_id] = natural
    result = Counter()
    for seed in frozen["seeds"]:
        if seed["tier"] == "T1":
            continue
        natural = writebacks.get(seed["seed_id"])
        valid, missing = validate_value_occurrences(seed, natural) if natural is not None else (False, ["<missing-writeback>"])
        if valid:
            seed["natural_request"] = natural
            seed["user_request"] = natural
            seed["paraphrase"] = {"status": "accepted"}
            result["accepted"] += 1
        else:
            original = seed["tier"]
            seed["natural_request"] = seed["transcription_request"]
            seed["user_request"] = seed["transcription_request"]
            seed["tier"] = "T1"
            seed["tier_original"] = original
            seed["tier_downgrade"] = {"reason": "value_occurrence_failed", "missing_literals": missing}
            seed["paraphrase"] = {"status": "fallback_transcription"}
            result["fallback"] += 1
    frozen["paraphrase_ingested_at"] = dt.datetime.now(dt.UTC).isoformat()
    frozen["paraphrase_result"] = dict(result)
    atomic_json(target / "seeds.json", frozen)
    atomic_json(target / "paraphrase_ingest.json", {"schema_version": 1, "run_id": args.run_id,
                                                       "input": str(args.input), "result": dict(result)})
    print(canonical(dict(result)))


def _audit_prompt(seed: dict[str, Any]) -> str:
    payload = {"tools": seed["tools"], "natural_request": seed["natural_request"],
               "expected_trace": seed["expected_stages"]}
    return ("Audit unique solvability of this offline tool-call task. Decide whether the request and declared "
            "tools determine exactly the expected trace; identify alternate valid calls, missing information, "
            "or ambiguity. Do not rewrite the request. Return JSON with verdict (PASS|FAIL), reasons, and "
            "alternative_trace if applicable.\nTASK=" + canonical(payload))


def emit_audit_batch(args: argparse.Namespace) -> None:
    target, _, frozen = _load_frozen_seeds(args.run_id)
    output = Path(args.output) if args.output else target / "audit_batch.jsonl"
    rows = [{"schema_version": 1, "seed_id": seed["seed_id"], "tier": seed["tier"],
             "tools": seed["tools"], "natural_request": seed["natural_request"],
             "expected_trace": seed["expected_stages"], "audit_prompt": _audit_prompt(seed)}
            for seed in frozen["seeds"]]
    atomic_jsonl(output, rows)
    print(output)


def generation_worker(gpu: int, seeds: list[dict[str, Any]], spec: GenerationSpec,
                      checkpoint: Path, output: mp.Queue) -> None:
    """v1's sampling contract with this fork's long-chain executor semantics."""
    try:
        torch, tokenizer, model = v1.load_stock_model(gpu)
        for number, seed in enumerate(seeds, 1):
            tool_map = {tool["name"]: tool for tool in seed["tools"]}
            selected, all_results, selections, failures = [], [], [], []
            prior_results: list[dict[str, Any]] = []
            for stage, expected in enumerate(seed["expected_stages"]):
                prompt = v1.scaffold_prompt(seed, stage, prior_results)
                raws = v1.generate_candidates(torch, tokenizer, model, prompt, spec)
                calls, reasons, chosen = select_candidate(raws, expected, tool_map)
                if calls is None:
                    failures.extend(reasons)
                    break
                results = [mock_execute(call, stage, seed["pattern"]) for call in calls]
                selected.append(calls)
                all_results.append(results)
                selections.append({"stage": stage, "candidate_index": chosen,
                                   "candidate_rejections": reasons})
                prior_results.extend(results)
            v1.append_checkpoint_record(checkpoint, {"seed": seed, "selected": selected,
                                                       "results": all_results, "selection": selections,
                                                       "failures": failures})
            print(f"[selfgen-intent gpu{gpu}] {number}/{len(seeds)}", flush=True)
        output.put({"kind": "complete", "gpu": gpu, "records_written": len(seeds)})
    except BaseException as exc:
        output.put({"kind": "error", "gpu": gpu, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": __import__("traceback").format_exc()})


def gpu_preflight(run_id: str) -> dict[str, Any]:
    """Fork-local copy so the production path cannot accidentally use v1's run dir."""
    target, manifest, _ = _load_frozen_seeds(run_id)
    if manifest.get("state") != "prepared":
        raise RuntimeError(f"run state must be prepared, got {manifest.get('state')!r}")
    identity = v1.verified_stock_identity()
    if manifest.get("model") != {"identity": identity, "topk": 8, "patch": None}:
        raise RuntimeError("true-stock identity changed since preparation")
    proc = v1.subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, stdout=v1.subprocess.PIPE, stderr=v1.subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU preflight failed: nvidia-smi exit {proc.returncode}: {proc.stdout.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "memory_used_mib": int(fields[2]),
                     "utilization_percent": int(fields[3])})
    by_id = {row["index"]: row for row in rows}
    if not {0, 1}.issubset(by_id) or 2 not in by_id:
        raise RuntimeError("expected local physical GPUs 0, 1, and display GPU 2")
    busy = [by_id[index] for index in (0, 1)
            if by_id[index]["memory_used_mib"] > 1024 or by_id[index]["utilization_percent"] > 5]
    if busy:
        raise RuntimeError(f"GPU 0/1 are busy: {busy}")
    result = {"checked_at": dt.datetime.now(dt.UTC).isoformat(), "run_id": run_id,
              "allowed_physical_gpus": [0, 1], "forbidden_display_gpu": 2, "gpus": rows,
              "stock": identity, "result": "PASS"}
    atomic_json(target / f"preflight_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}.json", result)
    return result


def execute(args: argparse.Namespace) -> None:
    target, manifest, frozen = _load_frozen_seeds(args.run_id)
    if manifest.get("state") != "prepared":
        raise RuntimeError(f"run state must be prepared, got {manifest.get('state')!r}")
    if manifest.get("model") != {"identity": v1.verified_stock_identity(), "topk": 8, "patch": None}:
        raise RuntimeError("true-stock k8 identity in manifest changed")
    if args.fixture:
        records = cpu_fixture_records(frozen["seeds"])
        atomic_json(target / "fixture_validation.json", {"not_training_data": True, "records": records})
    else:
        completed = v1.load_checkpoint_records(target, frozen["seeds"])
        todo = v1.pending_seeds(frozen["seeds"], completed)
        print(f"[selfgen-intent] resuming {args.run_id}: {len(completed)} completed, {len(todo)} pending", flush=True)
        if todo:
            gpu_preflight(args.run_id)
            by_gpu = {0: todo[0::2], 1: todo[1::2]}
            output: mp.Queue = mp.Queue()
            workers = [mp.Process(target=generation_worker, args=(gpu, by_gpu[gpu],
                       GenerationSpec(args.seed, args.temperature, args.max_new, args.best_of, gpu),
                       v1.checkpoint_path(target, gpu), output)) for gpu in (0, 1)]
            for worker in workers:
                worker.start()
            try:
                v1.wait_for_worker_completion(workers, output)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join(timeout=30)
                output.close()
                output.join_thread()
        records = list(v1.load_checkpoint_records(target, frozen["seeds"]).values())
        records.sort(key=lambda record: record["seed"]["seed_id"])
        if len(records) != len(frozen["seeds"]):
            raise RuntimeError("missing generated records")
    contamination, grams, names = v1.contamination_corpus()
    accepted, rejected, reasons = evaluate_records(records, grams, names)
    if args.fixture:
        atomic_json(target / "fixture_summary.json", {"accepted": len(accepted), "rejected": len(rejected),
                                                       "reason_counts": dict(reasons)})
        print("fixture validation passed; no training jsonl written")
        return
    atomic_jsonl(target / "train.jsonl", accepted)
    atomic_jsonl(target / "rejected.jsonl", rejected)
    def count_by(items: Iterable[Any], key: str, nested: bool = False) -> dict[str, int]:
        values = {tier: 0 for tier in TIERS} if key == "tier" else {}
        for item in items:
            source = item["metadata"] if nested else item["seed"]
            value = source[key]
            values[value] = values.get(value, 0) + 1
        return values
    summary = {"schema_version": 1, "run_id": args.run_id, "completed_at": dt.datetime.now(dt.UTC).isoformat(),
               "generated": len(records), "accepted": len(accepted), "rejected": len(rejected),
               "acceptance_rate": len(accepted) / len(records) if records else 0.0,
               "rejection_reasons": dict(reasons), "strata": {"generated_tier": count_by(records, "tier"),
               "accepted_tier": count_by(accepted, "tier", nested=True)}, "truncation_count": 0,
               "contamination": contamination}
    atomic_json(target / "summary.json", summary)
    manifest.update({"state": "completed", "status": "complete", "completed_at": summary["completed_at"]})
    manifest["artifacts"].update({"generation_checkpoints": ["generation_records_gpu0.jsonl",
                                                               "generation_records_gpu1.jsonl"],
                                  "train": "train.jsonl", "rejected": "rejected.jsonl", "summary": "summary.json"})
    atomic_json(target / "manifest.json", manifest)
    print(canonical(summary))


def prepare(args: argparse.Namespace) -> None:
    target = run_dir(args.run_id)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    mix = parse_tier_mix(args.tier_mix)
    stock = v1.verified_stock_identity()
    profile = v1.bfcl_structural_profile()
    contamination, grams, names = v1.contamination_corpus()
    seeds, rejected = build_seeds(args.n, args.seed, profile, grams, names, mix)
    target.mkdir(parents=True)
    atomic_json(target / "seeds.json", {"schema_version": 1, "created_at": dt.datetime.now(dt.UTC).isoformat(),
                                         "rng_seed": args.seed, "count": args.n, "tier_mix": mix,
                                         "tier_counts": tier_counts(args.n, mix), "bfcl_structure": profile,
                                         "contamination": contamination, "seed_filter_rejections": dict(rejected),
                                         "seeds": seeds})
    atomic_json(target / "manifest.json", {"schema_version": 1, "run_id": args.run_id, "state": "prepared",
                                             "created_at": dt.datetime.now(dt.UTC).isoformat(),
                                             "model": {"identity": stock, "topk": 8, "patch": None},
                                             "generation": {"gpus": [0, 1], "best_of": args.best_of,
                                                            "temperature": args.temperature, "max_new": args.max_new},
                                             "strata": {"tiers": list(TIERS), "tier_mix": mix,
                                                         "patterns": list(v1.PATTERNS) + ["long_chain"],
                                                         "domains": list(v1.DOMAINS), "schema_count": v1.SCHEMA_COUNT},
                                             "contamination": contamination,
                                             "artifacts": {"seeds": "seeds.json", "paraphrase_batch": "paraphrase_batch.jsonl",
                                                           "audit_batch": "audit_batch.jsonl",
                                                           "generation_checkpoints": ["generation_records_gpu0.jsonl",
                                                                                        "generation_records_gpu1.jsonl"]}})
    print(target)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--run-id", required=True)
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=20260711)
    s.add_argument("--best-of", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-new", type=int, default=512)
    s.add_argument("--tier-mix", default=DEFAULT_TIER_MIX)
    s.set_defaults(func=prepare)
    s = sub.add_parser("execute")
    s.add_argument("--run-id", required=True)
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=20260711)
    s.add_argument("--best-of", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-new", type=int, default=512)
    s.add_argument("--fixture", action="store_true")
    s.set_defaults(func=execute)
    s = sub.add_parser("preflight")
    s.add_argument("--run-id", required=True)
    s.set_defaults(func=lambda args: print(json.dumps(gpu_preflight(args.run_id), ensure_ascii=False, indent=2)))
    for name, func in (("emit-paraphrase-batch", emit_paraphrase_batch),
                       ("emit-audit-batch", emit_audit_batch)):
        s = sub.add_parser(name)
        s.add_argument("--run-id", required=True)
        s.add_argument("--output")
        s.set_defaults(func=func)
    s = sub.add_parser("ingest-paraphrase")
    s.add_argument("--run-id", required=True)
    s.add_argument("--input", required=True)
    s.set_defaults(func=ingest_paraphrase)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if hasattr(args, "n") and (args.n <= 0 or args.best_of != 4 or args.max_new < 32):
        raise SystemExit("require n>0, best-of=4, and max-new>=32")
    args.func(args)
