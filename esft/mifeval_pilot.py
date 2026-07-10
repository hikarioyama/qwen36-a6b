#!/usr/bin/env python3
"""Offline M-IFEval Japanese strict-score seed-dispersion pilot.

This is deliberately a pilot harness, rather than a frozen benchmark protocol.
It keeps upstream M-IFEval's strict, rule-based evaluator as the sole scorer:
an item passes only when every verifiable instruction in that item passes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ESFT = ROOT / "esft"
MIFEVAL = ROOT / "external" / "M-IFEval"
INPUT = MIFEVAL / "data" / "ja_input_data.jsonl"
RUN_ROOT = ESFT / "reports" / "eval" / "codex_runs"
CONFIG = ESFT / "codex_harness_jmmlu_b2_1000.toml"
SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260711
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
SCORER_PYTHON = ROOT / "external" / "mifeval-venv" / "bin" / "python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def atomic_json(path: Path, value: Any) -> None:
    """Update only the live manifest; all measurement assets use O_EXCL."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def source_hashes() -> dict[str, str]:
    paths = {
        "mifeval_pilot.py": Path(__file__),
        "evaluation_main.py": MIFEVAL / "evaluation_main.py",
        "instructions_registry.py": MIFEVAL / "instructions_registry.py",
        "ja_instructions.py": MIFEVAL / "instructions" / "ja_instructions.py",
        "ja_instructions_util.py": MIFEVAL / "instruction_utils" / "ja_instructions_util.py",
        "ja_input_data.jsonl": INPUT,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def import_upstream_scorer() -> dict[str, Any]:
    """Import the unmodified upstream strict checker, or fail before GPU use."""
    if not MIFEVAL.is_dir() or not INPUT.is_file():
        raise RuntimeError("M-IFEval source or Japanese input data is missing")
    if str(MIFEVAL) not in sys.path:
        sys.path.insert(0, str(MIFEVAL))
    try:
        from evaluation_main import InputExample, test_instruction_following_strict
        import instructions_registry
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "M-IFEval upstream rule scorer dependency is unavailable: "
            f"{exc.name}. Install the pinned local/offline requirements before running; "
            "do not substitute a local scorer."
        ) from exc
    return {
        "InputExample": InputExample,
        "strict": test_instruction_following_strict,
        "registry_size": len(instructions_registry.INSTRUCTION_DICT),
    }


def run_scorer(*args: str) -> str:
    """Run the untouched upstream scorer in its dedicated Python 3.12 venv."""
    if not SCORER_PYTHON.is_file():
        raise RuntimeError(f"required M-IFEval scorer interpreter is missing: {SCORER_PYTHON}")
    completed = subprocess.run(
        [str(SCORER_PYTHON), str(Path(__file__)), *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "dedicated upstream M-IFEval scorer failed "
            f"(rc={completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def scorer_preflight() -> dict[str, Any]:
    """Verify the exact strict scorer before allocating any GPU memory."""
    try:
        result = json.loads(run_scorer("scorer-preflight", "--json"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("dedicated M-IFEval scorer emitted invalid preflight JSON") from exc
    if result.get("status") != "pass":
        raise RuntimeError(f"dedicated M-IFEval scorer preflight did not pass: {result}")
    return result


def prepare(run_dir: Path) -> None:
    """Freeze all 172 Japanese upstream inputs with stable content keys."""
    out = run_dir / "subset.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    rows = []
    with INPUT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            item_key = hashlib.sha256(canonical_json({
                "key": row["key"], "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"],
            }).encode("utf-8")).hexdigest()
            rows.append({**row, "item_key": item_key})
    if len(rows) != 172 or len({row["key"] for row in rows}) != len(rows):
        raise RuntimeError(f"unexpected Japanese M-IFEval cardinality: {len(rows)}")
    write_new_json(out, {
        "schema_version": 1,
        "benchmark": "M-IFEval Japanese",
        "selection": {"n": len(rows), "method": "all upstream ja_input_data.jsonl rows in source order"},
        "source": {"root": str(MIFEVAL), "input": str(INPUT), "sha256": sha256_file(INPUT)},
        "items": rows,
    })


def count_gen_tokens(row_ids: list[int], eos_ids: list[int]) -> int:
    hits = [row_ids.index(eos) for eos in eos_ids if eos in row_ids]
    return min(hits) + 1 if hits else len(row_ids)


def hits_cap(row_ids: list[int], eos_ids: list[int], max_new: int) -> bool:
    return len(row_ids) >= max_new and not any(eos in row_ids for eos in eos_ids)


def worker_seed(global_seed: int, physical_gpu: int) -> int:
    """Stable, recorded split-stream derivation for the fixed two-GPU partition."""
    return global_seed * 1_000_003 + physical_gpu


def generation_worker(gpu_id: int, spec: dict[str, Any], items: list[dict[str, Any]],
                      seed: int, batch_size: int, max_new: int, temperature: float,
                      top_p: float, output: mp.Queue) -> None:
    import torch

    sys.path.insert(0, str(ESFT))
    from eval_harness import EOS_IDS, load_subject_model

    torch.cuda.set_device(gpu_id)
    local_seed = worker_seed(seed, gpu_id)
    random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    tokenizer, model, _ = load_subject_model(spec, gpu_id)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}], add_generation_prompt=True,
        tokenize=False, enable_thinking=False,
    ) for item in items]
    records = []
    started = time.time()
    for start in range(0, len(prompts), batch_size):
        selected = items[start:start + batch_size]
        enc = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(f"cuda:{gpu_id}")
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                top_p=top_p, eos_token_id=EOS_IDS, pad_token_id=tokenizer.pad_token_id,
            )
        for row, item in zip(generated[:, input_len:], selected):
            ids = row.tolist()
            records.append({
                "id": item["key"], "item_key": item["item_key"],
                "response": tokenizer.decode(row, skip_special_tokens=True),
                "truncated": hits_cap(ids, EOS_IDS, max_new),
                "gen_len": count_gen_tokens(ids, EOS_IDS),
            })
        print(f"[mifeval gpu{gpu_id} seed={seed}] {min(start + len(selected), len(prompts))}/{len(prompts)}", flush=True)
    output.put({"gpu": gpu_id, "items": records, "elapsed_s": time.time() - started,
                "worker_seed": local_seed})


def model_spec(arm: str) -> dict[str, Any]:
    sys.path.insert(0, str(ESFT))
    from eval_harness import resolve_model_spec
    import codex_harness

    cfg = codex_harness.load_config(CONFIG)
    if arm == "base_k8":
        return resolve_model_spec("base", model_path=cfg["stock"]["path"], topk=8)
    if arm == "b2_k32":
        return resolve_model_spec("patched", model_path=cfg["stock"]["path"],
                                  patch=cfg["patches"]["b2"]["path"], topk=32)
    raise ValueError(f"unknown arm {arm}")


def generate(run_dir: Path, arm: str, seed: int, batch_size: int, max_new: int,
             temperature: float, top_p: float) -> None:
    out = run_dir / f"{arm}_seed{seed}_raw.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    items = read_json(run_dir / "subset.json")["items"]
    spec = model_spec(arm)
    by_gpu = {0: items[0::2], 1: items[1::2]}
    output: mp.Queue = mp.Queue()
    procs = [mp.Process(target=generation_worker,
                        args=(gpu, spec, by_gpu[gpu], seed, batch_size, max_new,
                              temperature, top_p, output)) for gpu in (0, 1)]
    for proc in procs:
        proc.start()
    received = []
    try:
        while len(received) < 2:
            try:
                received.append(output.get(timeout=15))
            except queue.Empty:
                failed = [(p.pid, p.exitcode) for p in procs if p.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"M-IFEval generation worker failed: {failed}")
                if all(not p.is_alive() for p in procs):
                    raise RuntimeError("M-IFEval workers exited without all results")
    except BaseException:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        raise
    finally:
        for proc in procs:
            proc.join(timeout=30)
    failed = [(p.pid, p.exitcode) for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"M-IFEval generation worker failed: {failed}")
    raw = [row for group in received for row in group["items"]]
    raw.sort(key=lambda row: row["item_key"])
    if len(raw) != len(items) or len({row["item_key"] for row in raw}) != len(items):
        raise RuntimeError("M-IFEval generation output has missing or duplicate item keys")
    write_new_json(out, {
        "schema_version": 1, "arm": arm, "seed": seed,
        "model": spec["kind"], "topk": spec["topk"], "patch": spec["patch"],
        "inference_python_executable": sys.executable, "inference_python_version": sys.version,
        "physical_gpus": [0, 1], "batch_size": batch_size, "max_new": max_new,
        "sampling": {"do_sample": True, "temperature": temperature, "top_p": top_p,
                     "worker_seed_derivation": "seed * 1000003 + physical_gpu"},
        "worker_seconds": {str(row["gpu"]): row["elapsed_s"] for row in received},
        "worker_seeds": {str(row["gpu"]): row["worker_seed"] for row in received}, "items": raw,
    })


def score(run_dir: Path, arm: str, seed: int) -> None:
    out = run_dir / f"{arm}_seed{seed}_items.json"
    summary = run_dir / f"{arm}_seed{seed}_summary.json"
    if out.exists() or summary.exists():
        raise FileExistsError(f"refusing to overwrite score artifacts for {arm}/seed{seed}")
    scorer = import_upstream_scorer()
    subset = read_json(run_dir / "subset.json")
    raw = read_json(run_dir / f"{arm}_seed{seed}_raw.json")
    raw_by_key = {row["item_key"]: row for row in raw["items"]}
    if len(raw_by_key) != len(subset["items"]):
        raise RuntimeError("raw output keys do not match frozen subset")
    records = []
    for item in subset["items"]:
        raw_row = raw_by_key[item["item_key"]]
        example = scorer["InputExample"](key=item["key"], prompt=item["prompt"],
            instruction_id_list=item["instruction_id_list"], kwargs=item["kwargs"])
        strict = scorer["strict"](example, {item["prompt"]: raw_row["response"]})
        followed = [bool(value) for value in strict.follow_instruction_list]
        records.append({
            "id": item["key"], "item_key": item["item_key"], "correct": bool(strict.follow_all_instructions),
            "follow_instruction_list": followed, "instruction_id_list": item["instruction_id_list"],
            "response": raw_row["response"], "truncated": bool(raw_row["truncated"]),
            "gen_len": int(raw_row["gen_len"]),
        })
    meta = {
        "benchmark": "mifeval_ja_strict", "model_path": raw["patch"] or "true_stock",
        "n_per_benchmark": len(records), "batch_size": raw["batch_size"], "max_new": raw["max_new"],
        "seed": seed, "shuffle": False, "gpus": [0, 1], "no_think": True,
        "choice_logprob": False, "effective_prompt_modes": {"mifeval": "generation_no_think"},
        "split": "all upstream Japanese rows in source order", "harness_sha256": sha256_file(Path(__file__)),
        "source_sha256": source_hashes(), "inference_python_executable": raw["inference_python_executable"],
        "inference_python_version": raw["inference_python_version"], "scorer_python_executable": sys.executable,
        "scorer_python_version": sys.version, "package_versions": {},
        "upstream_registry_size": scorer["registry_size"],
        "sampling": raw["sampling"], "arm": arm, "topk": raw["topk"], "patch": raw["patch"],
    }
    write_new_json(out, {"_meta": meta, "items": {"mifeval_ja": records}})
    write_new_json(summary, {"n": len(records), "correct": sum(row["correct"] for row in records),
        "pass_rate": sum(row["correct"] for row in records) / len(records),
        "truncated_n": sum(row["truncated"] for row in records), "items_sha256": sha256_file(out)})


def agreement(values: list[bool]) -> float:
    """Fraction of all unordered seed pairs with the same binary strict outcome."""
    n = len(values)
    if n < 2:
        raise ValueError("agreement requires two or more seeds")
    passed = sum(values)
    return (math.comb(passed, 2) + math.comb(n - passed, 2)) / math.comb(n, 2)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    index = (len(sorted_values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo] if lo == hi else sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def summarize_arm(per_seed: list[list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, list[float]]]:
    by_key: dict[str, list[bool]] = {}
    truncations = 0
    seed_rates = []
    for records in per_seed:
        seed_rates.append(sum(row["correct"] for row in records) / len(records))
        truncations += sum(row["truncated"] for row in records)
        for row in records:
            by_key.setdefault(row["item_key"], []).append(bool(row["correct"]))
    if not by_key or any(len(values) != len(SEEDS) for values in by_key.values()):
        raise RuntimeError("incomplete seed matrix")
    per_item_pass = {key: sum(values) / len(values) for key, values in by_key.items()}
    per_item_agreement = {key: agreement(values) for key, values in by_key.items()}
    return ({
        "n": len(by_key), "seeds": list(SEEDS), "pass_rate": sum(seed_rates) / len(seed_rates),
        "pass_rate_by_seed": seed_rates, "pass_rate_seed_sd": math.sqrt(sum(
            (rate - sum(seed_rates) / len(seed_rates)) ** 2 for rate in seed_rates) / (len(seed_rates) - 1)),
        "agreement": sum(per_item_agreement.values()) / len(per_item_agreement),
        "truncated_n_total": truncations,
    }, {"pass": per_item_pass, "agreement": per_item_agreement})


def paired_bootstrap(base: dict[str, float], b2: dict[str, float]) -> dict[str, Any]:
    keys = sorted(base)
    if keys != sorted(b2):
        raise RuntimeError("paired seed matrices have different item keys")
    observed = sum(b2[key] - base[key] for key in keys) / len(keys)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        draws.append(sum(b2[keys[rng.randrange(len(keys))]] - base[keys[rng.randrange(len(keys))]]
                         for _ in keys) / len(keys))
    draws.sort()
    return {"delta_b2_minus_base": observed, "ci95": [percentile(draws, .025), percentile(draws, .975)],
            "method": "paired percentile bootstrap over items; all five seed outcomes remain clustered within item",
            "replicates": BOOTSTRAP_REPS, "rng_seed": BOOTSTRAP_SEED}


def analyze(run_dir: Path) -> dict[str, Any]:
    matrices = {}
    for arm in ("base_k8", "b2_k32"):
        per_seed = []
        for seed in SEEDS:
            payload = read_json(run_dir / f"{arm}_seed{seed}_items.json")
            per_seed.append(payload["items"]["mifeval_ja"])
        matrices[arm] = summarize_arm(per_seed)
    report = {
        "n": matrices["base_k8"][0]["n"], "seeds": list(SEEDS),
        "base_k8": matrices["base_k8"][0], "b2_k32": matrices["b2_k32"][0],
        "paired": {
            "pass_rate": paired_bootstrap(matrices["base_k8"][1]["pass"], matrices["b2_k32"][1]["pass"]),
            "agreement": paired_bootstrap(matrices["base_k8"][1]["agreement"], matrices["b2_k32"][1]["agreement"]),
        },
    }
    write_new_json(run_dir / "analysis.json", report)
    return report


def campaign(run_id: str, batch_size: int, max_new: int, temperature: float, top_p: float) -> Path:
    if not 0 < temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
    import codex_harness
    scorer = scorer_preflight()  # Dedicated scorer gate: no GPU command precedes this.
    cfg = codex_harness.load_config(CONFIG)
    with codex_harness.campaign_lock():
        audit = codex_harness.preflight(cfg, include_gpu=True)
        run_dir.mkdir(parents=True)
        manifest = {"schema_version": 1, "status": "running", "run_id": run_id,
            "benchmark": "mifeval_ja_strict_seed_dispersion", "n": 172,
            "protocol": {"arms": ["base_k8", "b2_k32"], "seeds": list(SEEDS), "serial": True,
                "batch_size": batch_size, "max_new": max_new, "temperature": temperature, "top_p": top_p,
                "scorer": "unmodified upstream M-IFEval strict rule-based"},
            "preflight": {"campaign": audit, "upstream_registry_size": scorer["registry_size"]},
            "started_at": dt.datetime.now(dt.UTC).isoformat(), "arms": {}}
        atomic_json(run_dir / "manifest.json", manifest)
        try:
            prepare(run_dir)
            for arm in ("base_k8", "b2_k32"):
                for seed in SEEDS:
                    label = f"{arm}_seed{seed}"
                    manifest["arms"][label] = {"status": "running", "started_at": dt.datetime.now(dt.UTC).isoformat()}
                    atomic_json(run_dir / "manifest.json", manifest)
                    generate(run_dir, arm, seed, batch_size, max_new, temperature, top_p)
                    run_scorer("score", "--run-dir", str(run_dir), "--arm", arm, "--seed", str(seed))
                    manifest["arms"][label] = {"status": "complete", "finished_at": dt.datetime.now(dt.UTC).isoformat(),
                        "summary": read_json(run_dir / f"{label}_summary.json")}
                    atomic_json(run_dir / "manifest.json", manifest)
            manifest["analysis"] = analyze(run_dir)
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            manifest["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
            atomic_json(run_dir / "manifest.json", manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="check dedicated upstream strict scorer before any GPU allocation")
    pre.add_argument("--json", action="store_true")
    scorer_pre = sub.add_parser("scorer-preflight", help=argparse.SUPPRESS)
    scorer_pre.add_argument("--json", action="store_true")
    scorer_score = sub.add_parser("score", help=argparse.SUPPRESS)
    scorer_score.add_argument("--run-dir", type=Path, required=True)
    scorer_score.add_argument("--arm", choices=("base_k8", "b2_k32"), required=True)
    scorer_score.add_argument("--seed", type=int, choices=SEEDS, required=True)
    camp = sub.add_parser("campaign", help="run serial base/B2 × five seeds")
    camp.add_argument("--run-id", required=True)
    camp.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    camp.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    camp.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    camp.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    args = parser.parse_args()
    if args.command == "preflight":
        result = scorer_preflight()
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "M-IFEval strict scorer preflight PASS")
        return
    if args.command == "scorer-preflight":
        if Path(sys.executable).resolve() != SCORER_PYTHON.resolve():
            raise RuntimeError("strict scorer must run in external/mifeval-venv Python")
        scorer = import_upstream_scorer()
        result = {"status": "pass", "n": 172, "registry_size": scorer["registry_size"], "source_sha256": source_hashes(),
                  "scorer_python_executable": sys.executable, "scorer_python_version": sys.version}
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "M-IFEval strict scorer preflight PASS")
        return
    if args.command == "score":
        if Path(sys.executable).resolve() != SCORER_PYTHON.resolve():
            raise RuntimeError("strict scorer must run in external/mifeval-venv Python")
        score(args.run_dir, args.arm, args.seed)
        return
    path = campaign(args.run_id, args.batch_size, args.max_new, args.temperature, args.top_p)
    print(f"M-IFEval pilot complete: {path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
