#!/usr/bin/env python3
"""Offline, paired BFCL-v4 AST pilot for Qwen3.6-35B-A6B.

The script deliberately splits its work between two interpreters:

* ``/usr/bin/python3`` owns model loading/generation, matching the established
  local Qwen evaluation path;
* ``external/bfcl-venv/bin/python`` owns BFCL data loading, the upstream AST
  parser, and the upstream AST checker.  The local pinned Gorilla checkout is
  prepended to ``PYTHONPATH`` for those calls.

Only deterministic, non-live BFCL v4 AST categories are eligible.  The native
Qwen3.6 template emits ``<function=...>/<parameter=...>`` calls, whereas BFCL's
older QwenFC handler expects JSON.  ``score`` therefore only normalizes that
documented native syntax into a Python call expression and then invokes BFCL's
upstream ``ast_parse`` and ``ast_checker``; it does not reimplement scoring.
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
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ESFT = ROOT / "esft"
GORILLA = ROOT / "external" / "gorilla" / "berkeley-function-call-leaderboard"
BFCL_PYTHON = ROOT / "external" / "bfcl-venv" / "bin" / "python"
MODEL_PATH = Path(
    "/mnt/data/hf_cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
PATCH_PATH = Path("~/models/esft/b2_1000/expert_patch.safetensors")
PATCH_SHA256 = "c1b3f041051e9c184e5a3ea14126f921e3a2619b29454e3e73b96f79f45199d3"
PATCH_TENSORS = 1666
RUN_ROOT = ESFT / "reports" / "eval" / "codex_runs"

# Explicitly excludes all live, web_search, memory, multi-turn and executable
# categories.  These six are the deterministic AST-only BFCL v4 subset.
CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
)
LANGUAGE_BY_CATEGORY = {
    "simple_java": "JAVA",
    "simple_javascript": "JAVASCRIPT",
}
DEFAULT_MAX_NEW = 512
DEFAULT_BATCH_SIZE = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_new_json(path: Path, value: Any) -> None:
    """Create an artifact once; never silently replace a prior run artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def source_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(GORILLA) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def common_meta(subset: dict[str, Any], *, n: int, batch_size: int, max_new: int) -> dict:
    return {
        "benchmark": "bfcl_v4_nonlive_ast",
        "model_path": str(MODEL_PATH),
        "n_per_benchmark": n,
        "batch_size": batch_size,
        "max_new": max_new,
        "seed": 0,
        "shuffle": True,
        "gpus": [0, 1],
        "no_think": True,
        "choice_logprob": False,
        "effective_prompt_modes": {"bfcl": "generation_no_think_native_tools"},
        "split": "global deterministic shuffle seed 0; first-N",
        "harness_sha256": sha256_file(Path(__file__)),
        "source_sha256": {
            "bfcl_pilot": sha256_file(Path(__file__)),
            "upstream_ast_parser": sha256_file(GORILLA / "bfcl_eval" / "model_handler" / "utils.py"),
            "upstream_ast_checker": sha256_file(GORILLA / "bfcl_eval" / "eval_checker" / "ast_eval" / "ast_checker.py"),
        },
        "python_executable": "/usr/bin/python3",
        "python_version": sys.version,
        "package_versions": {},
        "bfcl": subset["bfcl"],
        "subset_sha256": sha256_file(Path(subset["_path"])),
    }


def prepare(run_dir: Path, n: int) -> None:
    """Run under bfcl-venv: use the pinned upstream loader and freeze subset."""
    if not BFCL_PYTHON.samefile(Path(sys.executable)):
        raise RuntimeError(f"prepare must use {BFCL_PYTHON}; got {sys.executable}")
    if n <= 0 or n > 400:
        raise ValueError("BFCL pilot n must be in 1..400")
    out = run_dir / "subset.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    candidates = []
    file_hashes = {}
    for category in CATEGORIES:
        prompts = load_dataset_entry(category)
        answers = load_ground_truth_entry(category)
        answer_by_id = {row["id"]: row for row in answers}
        if len(prompts) != len(answer_by_id):
            raise RuntimeError(f"{category}: prompt/ground-truth cardinality mismatch")
        for prompt in prompts:
            gt = answer_by_id.get(prompt["id"])
            if gt is None:
                raise RuntimeError(f"{category}: missing ground truth for {prompt['id']}")
            item_key = hashlib.sha256(canonical_json({
                "category": category, "id": prompt["id"],
                "question": prompt["question"], "function": prompt["function"],
                "ground_truth": gt["ground_truth"],
            }).encode()).hexdigest()
            candidates.append({
                "category": category,
                "id": prompt["id"],
                "item_key": item_key,
                "question": prompt["question"],
                "function": prompt["function"],
                "ground_truth": gt["ground_truth"],
            })
        for rel in (
            Path("bfcl_eval/data") / f"BFCL_v4_{category}.json",
            Path("bfcl_eval/data/possible_answer") / f"BFCL_v4_{category}.json",
        ):
            file_hashes[str(rel)] = sha256_file(GORILLA / rel)

    # Freeze a single globally shuffled first-N subset.  There is no category
    # reweighting after seeing model output.
    candidates.sort(key=lambda row: (row["category"], row["id"]))
    random.Random(0).shuffle(candidates)
    chosen = candidates[:n]
    allocation = {category: sum(x["category"] == category for x in chosen)
                  for category in CATEGORIES}
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "selection": {
            "n": n, "seed": 0, "shuffle": True,
            "method": "all eligible entries sorted(category,id), global random.Random(0).shuffle(), first-N",
            "categories": list(CATEGORIES),
            "category_allocation": allocation,
            "external_api_categories_excluded": ["web_search"],
        },
        "bfcl": {
            "gorilla_checkout": git_head(GORILLA.parent),
            "bfcl_eval_distribution": "2026.3.23",
            "source_root": str(GORILLA),
            "data_sha256": file_hashes,
        },
        "items": chosen,
    }
    write_new_json(out, payload)


def post_think(text: str) -> str:
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def count_gen_tokens(row_ids: list[int], eos_ids: list[int]) -> int:
    hits = [row_ids.index(eos) for eos in eos_ids if eos in row_ids]
    return min(hits) + 1 if hits else len(row_ids)


def hits_cap(row_ids: list[int], eos_ids: list[int], max_new: int) -> bool:
    return len(row_ids) >= max_new and not any(eos in row_ids for eos in eos_ids)


def generation_worker(gpu_id: int, spec: dict, items: list[dict], batch_size: int,
                      max_new: int, output: mp.Queue) -> None:
    """Model worker; this exact loading path is shared with the JS/JMMLU evals."""
    import torch
    sys.path.insert(0, str(ESFT))
    from eval_harness import EOS_IDS, load_subject_model

    torch.cuda.set_device(gpu_id)
    tok, model, _ = load_subject_model(spec, gpu_id)
    rendered = [tok.apply_chat_template(
        item["question"][0], tools=item["function"], add_generation_prompt=True,
        tokenize=False, enable_thinking=False,
    ) for item in items]
    records = []
    t0 = time.time()
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start:start + batch_size]
        selected = items[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(f"cuda:{gpu_id}")
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False,
                eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id,
            )
        for row, item in zip(generated[:, input_len:], selected):
            ids = row.tolist()
            records.append({
                "id": item["id"], "item_key": item["item_key"],
                "category": item["category"],
                "raw_completion": tok.decode(row, skip_special_tokens=True),
                "truncated": hits_cap(ids, EOS_IDS, max_new),
                "gen_len": count_gen_tokens(ids, EOS_IDS),
            })
        print(f"[bfcl gpu{gpu_id}] {min(start + len(chunk), len(rendered))}/{len(rendered)}", flush=True)
    output.put({"gpu": gpu_id, "items": records, "elapsed_s": time.time() - t0})


def generate(run_dir: Path, arm: str, batch_size: int, max_new: int) -> None:
    if arm not in {"base_k8", "b2_k32"}:
        raise ValueError(f"unknown arm {arm}")
    subset_path = run_dir / "subset.json"
    out = run_dir / f"{arm}_raw.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    subset = read_json(subset_path)
    items = subset["items"]
    if not items:
        raise RuntimeError("empty BFCL subset")
    sys.path.insert(0, str(ESFT))
    from eval_harness import resolve_model_spec

    spec = resolve_model_spec(
        "base" if arm == "base_k8" else "patched", model_path=str(MODEL_PATH),
        patch=str(PATCH_PATH) if arm == "b2_k32" else None,
        topk=8 if arm == "base_k8" else 32,
    )
    by_gpu = {0: items[0::2], 1: items[1::2]}
    output: mp.Queue = mp.Queue()
    procs = [mp.Process(target=generation_worker,
                        args=(gpu, spec, by_gpu[gpu], batch_size, max_new, output))
             for gpu in (0, 1)]
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
                    raise RuntimeError(f"BFCL generation worker failed: {failed}")
                if all(not p.is_alive() for p in procs):
                    raise RuntimeError("BFCL workers exited without all results")
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
        raise RuntimeError(f"BFCL generation worker failed: {failed}")
    raw = [row for group in received for row in group["items"]]
    raw.sort(key=lambda row: row["item_key"])
    if len(raw) != len(items) or len({row["item_key"] for row in raw}) != len(items):
        raise RuntimeError("BFCL generation output has missing or duplicate item keys")
    write_new_json(out, {
        "schema_version": 1,
        "arm": arm,
        "model": spec["kind"], "topk": spec["topk"], "patch": spec["patch"],
        "physical_gpus": [0, 1], "logical_gpus": [0, 1],
        "batch_size": batch_size, "max_new": max_new,
        "worker_seconds": {str(row["gpu"]): row["elapsed_s"] for row in received},
        "items": raw,
    })


_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def native_qwen_to_python(text: str) -> str:
    """Thin syntax adapter; BFCL's upstream parser performs AST decoding."""
    import ast
    calls = []
    for block in _TOOL_RE.findall(post_think(text)):
        function = _FUNCTION_RE.search(block)
        if not function:
            continue
        name = function.group(1).strip()
        args = []
        for parameter, raw in _PARAM_RE.findall(function.group(2)):
            raw = raw.strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    value = raw
            args.append(f"{parameter.strip()}={value!r}")
        calls.append(f"{name}({', '.join(args)})")
    if not calls:
        raise ValueError("no native Qwen <tool_call>/<function=...> block")
    return calls[0] if len(calls) == 1 else "[" + ", ".join(calls) + "]"


def score(run_dir: Path, arm: str, batch_size: int, max_new: int) -> None:
    """Run under bfcl-venv and score every output through upstream BFCL code."""
    if not BFCL_PYTHON.samefile(Path(sys.executable)):
        raise RuntimeError(f"score must use {BFCL_PYTHON}; got {sys.executable}")
    subset_path = run_dir / "subset.json"
    raw_path = run_dir / f"{arm}_raw.json"
    out = run_dir / f"{arm}_items.json"
    summary = run_dir / f"{arm}_summary.json"
    if out.exists() or summary.exists():
        raise FileExistsError(f"refusing to overwrite score artifacts for {arm}")
    subset = read_json(subset_path)
    subset["_path"] = str(subset_path)
    raw = read_json(raw_path)
    raw_by_key = {row["item_key"]: row for row in raw["items"]}
    if len(raw_by_key) != len(subset["items"]):
        raise RuntimeError("raw output keys do not match frozen subset")

    # bfcl-eval imports Qwen's optional API handler registry while importing the
    # checker.  This wheel omitted the unrelated audio extra (``soundfile``).
    # The pilot never instantiates that handler; an in-process empty module keeps
    # the pinned official parser/checker importable without installing anything
    # or touching the venv.  Record this compatibility shim in output metadata.
    soundfile_stubbed = False
    try:
        import soundfile  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "soundfile":
            raise
        import types
        sys.modules["soundfile"] = types.ModuleType("soundfile")
        soundfile_stubbed = True

    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.utils import ast_parse

    records = []
    errors = {}
    for item in subset["items"]:
        raw_row = raw_by_key.get(item["item_key"])
        if raw_row is None:
            raise RuntimeError(f"missing raw completion for {item['id']}")
        category = item["category"]
        language = getattr(Language, LANGUAGE_BY_CATEGORY.get(category, "PYTHON"))
        try:
            # ast_parse is the pinned upstream BFCL parser.  The adapter only
            # changes Qwen3.6's native XML syntax into its accepted input form.
            decoded = ast_parse(native_qwen_to_python(raw_row["raw_completion"]),
                                ReturnFormat.PYTHON)
            checked = ast_checker(item["function"], decoded, item["ground_truth"],
                                  language, category, "Qwen/Qwen3-32B-FC")
            correct = bool(checked["valid"])
            error_type = None if correct else checked.get("error_type")
        except Exception as exc:
            correct = False
            error_type = f"adapter_or_upstream_parser:{type(exc).__name__}"
        errors[error_type or "valid"] = errors.get(error_type or "valid", 0) + 1
        records.append({
            "id": item["id"], "item_key": item["item_key"], "category": category,
            "pred": raw_row["raw_completion"], "gold": item["ground_truth"],
            "correct": correct, "truncated": bool(raw_row["truncated"]),
            "gen_len": int(raw_row["gen_len"]), "error_type": error_type,
        })
    meta = common_meta(subset, n=len(records), batch_size=batch_size, max_new=max_new)
    meta.update({"model": raw["model"], "topk": raw["topk"], "patch": raw["patch"],
                 "bfcl_adapter": "native_qwen_xml -> upstream ast_parse(ReturnFormat.PYTHON) -> upstream ast_checker",
                 "bfcl_optional_soundfile_stubbed": soundfile_stubbed})
    payload = {"_meta": meta, "items": {"bfcl": records}}
    write_new_json(out, payload)
    correct = sum(row["correct"] for row in records)
    truncated = sum(row["truncated"] for row in records)
    write_new_json(summary, {
        "n": len(records), "correct": correct, "acc": correct / len(records),
        "truncated_n": truncated, "error_type_counts": errors,
        "items_sha256": sha256_file(out),
    })


def static_preflight() -> dict:
    from safetensors import safe_open
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"true-stock model unavailable: {MODEL_PATH}")
    if sha256_file(PATCH_PATH) != PATCH_SHA256:
        raise RuntimeError("B2 patch SHA-256 mismatch")
    with safe_open(str(PATCH_PATH), framework="pt", device="cpu") as f:
        if len(list(f.keys())) != PATCH_TENSORS:
            raise RuntimeError("B2 patch tensor-count mismatch")
    if not BFCL_PYTHON.is_file() or not GORILLA.is_dir():
        raise RuntimeError("pinned BFCL checkout or bfcl venv missing")
    imported = subprocess.run(
        [str(BFCL_PYTHON), "-c", (
            "import sys, types; "
            "sys.modules.setdefault('soundfile', types.ModuleType('soundfile')); "
            "from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker; "
            "import bfcl_eval; print(bfcl_eval.__file__)"
        )],
        env=source_env(), check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "stock_path": str(MODEL_PATH),
        "patch": {"path": str(PATCH_PATH), "sha256": PATCH_SHA256, "tensor_count": PATCH_TENSORS},
        "gorilla_checkout": git_head(GORILLA.parent),
        "bfcl_import": imported,
        "physical_gpus_requested": [0, 1],
        "gpu2_excluded_by": "CUDA_VISIBLE_DEVICES=0,1; actual CUDA model-load is the availability check",
        "network": "offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)",
    }


def invoke_bfcl(mode: str, run_dir: Path, **kwargs: Any) -> None:
    argv = [str(BFCL_PYTHON), str(Path(__file__)), mode, "--run-dir", str(run_dir)]
    for key, value in kwargs.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(argv, env=source_env(), check=True)


def campaign(run_id: str, n: int, batch_size: int, max_new: int,
             noninferiority_margin: float) -> Path:
    if not 0 <= noninferiority_margin <= 1:
        raise ValueError("noninferiority margin must be within [0, 1]")
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
    # The directory is intentionally created only after all static immutable
    # inputs pass; every subsequent artifact is exclusive-create.
    audit = static_preflight()
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "benchmark": "bfcl_v4_nonlive_ast", "n": n,
        "noninferiority_margin": noninferiority_margin,
        "protocol": {"seed": 0, "shuffle": True, "batch_size": batch_size,
                     "max_new": max_new, "categories": list(CATEGORIES),
                     "arms": ["base_k8", "b2_k32"], "serial": True},
        "preflight": audit, "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "arms": {},
    }
    write_new_json(run_dir / "manifest.json", manifest)
    try:
        invoke_bfcl("prepare", run_dir, n=n)
        for arm in ("base_k8", "b2_k32"):
            manifest["arms"][arm] = {"status": "running", "started_at": dt.datetime.now(dt.UTC).isoformat()}
            (run_dir / "manifest.json").unlink()
            write_new_json(run_dir / "manifest.json", manifest)
            generate(run_dir, arm, batch_size, max_new)
            invoke_bfcl("score", run_dir, arm=arm, batch_size=batch_size, max_new=max_new)
            manifest["arms"][arm] = {
                "status": "complete", "finished_at": dt.datetime.now(dt.UTC).isoformat(),
                "summary": read_json(run_dir / f"{arm}_summary.json"),
            }
            (run_dir / "manifest.json").unlink()
            write_new_json(run_dir / "manifest.json", manifest)
        sys.path.insert(0, str(ESFT))
        from eval_harness import noninferiority_verdict, paired_verdict
        a = read_json(run_dir / "base_k8_items.json")["items"]["bfcl"]
        b = read_json(run_dir / "b2_k32_items.json")["items"]["bfcl"]
        verdicts = {key: paired_verdict(a, b, key=key) for key in ("correct", "truncated")}
        verdicts["correct"]["noninferiority"] = noninferiority_verdict(
            verdicts["correct"], noninferiority_margin)
        manifest["verdicts"] = verdicts
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
        (run_dir / "manifest.json").unlink()
        write_new_json(run_dir / "manifest.json", manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p = sub.add_parser("score")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arm", choices=("base_k8", "b2_k32"), required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-new", type=int, required=True)
    p = sub.add_parser("campaign")
    p.add_argument("--run-id", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    p.add_argument("--noninferiority-margin", type=float, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_dir, args.n)
    elif args.command == "score":
        score(args.run_dir, args.arm, args.batch_size, args.max_new)
    else:
        path = campaign(args.run_id, args.n, args.batch_size, args.max_new,
                        args.noninferiority_margin)
        print(f"BFCL campaign complete: {path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
