#!/usr/bin/env python3
"""Offline, paired BFCL-v4 AST pilot for full-FFN checkpoints and alpha dials.

This is the full-FFN successor to :mod:`bfcl_pilot`.  It retains that pilot's
frozen BFCL-v4 AST subset, native Qwen tool-call adapter, upstream scoring, and
two-interpreter layout, but evaluates arbitrary HF checkpoint directories.
Arms are specified as ``name=model_path:topk:alpha`` and are run serially.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any

import bfcl_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
ESFT = ROOT / "esft"
BFCL_PYTHON = pilot.BFCL_PYTHON
GORILLA = pilot.GORILLA
RUN_ROOT = ESFT / "reports" / "eval" / "codex_runs"
MODEL_PATH = pilot.MODEL_PATH
CATEGORIES = pilot.CATEGORIES
LANGUAGE_BY_CATEGORY = pilot.LANGUAGE_BY_CATEGORY
DEFAULT_MAX_NEW = pilot.DEFAULT_MAX_NEW
DEFAULT_BATCH_SIZE = pilot.DEFAULT_BATCH_SIZE
# Kept explicitly so the native base can be named in an arm without copying a
# machine-specific path into a launcher.  A campaign still requires explicit
# --arm values, because its paired comparison must be declared up front.
DEFAULT_BASE_ARM = {"name": "base_k8", "model_path": str(MODEL_PATH),
                    "topk": 8, "alpha": 0.0}


def parse_arm(value: str) -> dict[str, Any]:
    """Parse ``name=model_path:topk:alpha`` without requiring a model on disk.

    Splitting from the right keeps ordinary absolute paths intact and supports a
    colon in a parent directory name.  Existence is intentionally checked by
    ``static_preflight`` so this parser remains CPU-unit-testable.
    """
    if "=" not in value:
        raise ValueError("--arm must have the form name=model_path:topk:alpha")
    name, encoded = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--arm name must not be empty")
    try:
        model_text, topk_text, alpha_text = encoded.rsplit(":", 2)
    except ValueError as exc:
        raise ValueError("--arm must have the form name=model_path:topk:alpha") from exc
    model_text = model_text.strip()
    if not model_text:
        raise ValueError(f"arm {name!r} has an empty model path")
    try:
        topk = int(topk_text)
    except ValueError as exc:
        raise ValueError(f"arm {name!r} topk must be an integer") from exc
    try:
        alpha = float(alpha_text)
    except ValueError as exc:
        raise ValueError(f"arm {name!r} alpha must be a number") from exc
    if not 1 <= topk <= 32:
        raise ValueError(f"arm {name!r} topk must be in 1..32")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"arm {name!r} alpha must be finite and in [0, 1]")
    return {"name": name, "model_path": str(Path(model_text).expanduser()),
            "topk": topk, "alpha": alpha}


def parse_arms(values: list[str]) -> list[dict[str, Any]]:
    arms = [parse_arm(value) for value in values]
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("--arm names must be unique")
    return arms


def make_router_tail_scale_hook(alpha: float):
    """Return the eval_harness router-tail-scale forward hook.

    This is copied from ``eval_harness.py`` lines 770--795 because importing its
    model loader would retain the legacy patch-capable loading route.  Scores
    are already normalized selected-expert weights; scaling ranks 9+ followed
    by normalization is equivalent to scaling before normalization.
    """
    import torch

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")

    def hook(_mod, _inp, out):
        if alpha == 1.0:
            return out
        logits, scores, idx = out
        if scores.shape[-1] <= 8:
            return None
        s = scores.float()
        order = s.argsort(dim=-1, descending=True)
        mask = torch.ones_like(s)
        mask.scatter_(-1, order[:, 8:], alpha)
        s = s * mask
        s = s / s.sum(-1, keepdim=True).clamp_min(1e-9)
        return (logits, s.to(scores.dtype), idx)

    return hook


def load_arm_model(arm: dict[str, Any], gpu_id: int):
    """Load one HF directory directly and configure its routed-MoE top-k.

    No ESFT patch is applied here: a full-FFN checkpoint is already a complete
    Hugging Face model directory.  GPU workers load one arm at a time.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ESFT))
    from esft_qwen.common import find_moe_blocks

    tok = AutoTokenizer.from_pretrained(arm["model_path"])
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        arm["model_path"], dtype=torch.bfloat16, device_map={"": gpu_id})
    model.eval()
    refs = find_moe_blocks(model)
    for ref in refs:
        ref.gate.top_k = int(arm["topk"])
    if int(arm["topk"]) == 32 and float(arm["alpha"]) < 1.0:
        hook = make_router_tail_scale_hook(float(arm["alpha"]))
        for ref in refs:
            ref.gate.register_forward_hook(hook)
        print(f"[gpu{gpu_id}] router tail-scale alpha={arm['alpha']} armed on "
              f"{len(refs)} gates", flush=True)
    print(f"[gpu{gpu_id}] {arm['name']} loaded moe_layers={len(refs)} "
          f"topk={arm['topk']} alpha={arm['alpha']}", flush=True)
    return tok, model


def generation_worker(gpu_id: int, arm: dict[str, Any], items: list[dict],
                      batch_size: int, max_new: int, output: mp.Queue) -> None:
    import torch
    sys.path.insert(0, str(ESFT))
    from eval_harness import EOS_IDS

    torch.cuda.set_device(gpu_id)
    tok, model = load_arm_model(arm, gpu_id)
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
                "truncated": pilot.hits_cap(ids, EOS_IDS, max_new),
                "gen_len": pilot.count_gen_tokens(ids, EOS_IDS),
            })
        print(f"[bfcl {arm['name']} gpu{gpu_id}] "
              f"{min(start + len(chunk), len(rendered))}/{len(rendered)}", flush=True)
    output.put({"gpu": gpu_id, "items": records, "elapsed_s": time.time() - t0})


def generate(run_dir: Path, arm: dict[str, Any], batch_size: int, max_new: int) -> None:
    """Generate a single arm across GPUs 0/1; campaigns invoke this serially."""
    subset = pilot.read_json(run_dir / "subset.json")
    items = subset["items"]
    if not items:
        raise RuntimeError("empty BFCL subset")
    out = run_dir / f"{arm['name']}_raw.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    by_gpu = {0: items[0::2], 1: items[1::2]}
    output: mp.Queue = mp.Queue()
    procs = [mp.Process(target=generation_worker,
                        args=(gpu, arm, by_gpu[gpu], batch_size, max_new, output))
             for gpu in (0, 1)]
    for proc in procs:
        proc.start()
    received = []
    try:
        while len(received) < len(procs):
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
    pilot.write_new_json(out, {
        "schema_version": 1, "arm": arm["name"], "model_path": arm["model_path"],
        "topk": arm["topk"], "alpha": arm["alpha"], "physical_gpus": [0, 1],
        "logical_gpus": [0, 1], "batch_size": batch_size, "max_new": max_new,
        "worker_seconds": {str(row["gpu"]): row["elapsed_s"] for row in received},
        "items": raw,
    })


def common_meta(subset: dict[str, Any], arm: dict[str, Any], *, n: int,
                batch_size: int, max_new: int) -> dict[str, Any]:
    meta = pilot.common_meta(subset, n=n, batch_size=batch_size, max_new=max_new)
    meta["model_path"] = arm["model_path"]
    meta["source_sha256"].pop("bfcl_pilot", None)
    meta["source_sha256"]["bfcl_fullffn_v1"] = pilot.sha256_file(Path(__file__))
    meta.update({"arm": arm["name"], "topk": arm["topk"], "alpha": arm["alpha"],
                 "model_loading": "AutoModelForCausalLM HF directory; no ESFT patch"})
    return meta


def score(run_dir: Path, arm: dict[str, Any], batch_size: int, max_new: int) -> None:
    """Score an arm through the same upstream AST parser/checker as bfcl_pilot."""
    if not BFCL_PYTHON.samefile(Path(sys.executable)):
        raise RuntimeError(f"score must use {BFCL_PYTHON}; got {sys.executable}")
    subset_path = run_dir / "subset.json"
    raw_path = run_dir / f"{arm['name']}_raw.json"
    out = run_dir / f"{arm['name']}_items.json"
    summary = run_dir / f"{arm['name']}_summary.json"
    if out.exists() or summary.exists():
        raise FileExistsError(f"refusing to overwrite score artifacts for {arm['name']}")
    subset = pilot.read_json(subset_path)
    subset["_path"] = str(subset_path)
    raw = pilot.read_json(raw_path)
    for key in ("arm", "model_path", "topk", "alpha"):
        expected = arm["name"] if key == "arm" else arm[key]
        if raw.get(key) != expected:
            raise RuntimeError(
                f"raw artifact {raw_path} does not match supplied arm: "
                f"{key}={raw.get(key)!r}, expected {expected!r}"
            )
    raw_by_key = {row["item_key"]: row for row in raw["items"]}
    if len(raw_by_key) != len(subset["items"]):
        raise RuntimeError("raw output keys do not match frozen subset")

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

    records, errors = [], {}
    for item in subset["items"]:
        raw_row = raw_by_key.get(item["item_key"])
        if raw_row is None:
            raise RuntimeError(f"missing raw completion for {item['id']}")
        category = item["category"]
        language = getattr(Language, LANGUAGE_BY_CATEGORY.get(category, "PYTHON"))
        try:
            decoded = ast_parse(pilot.native_qwen_to_python(raw_row["raw_completion"]),
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
    meta = common_meta(subset, arm, n=len(records), batch_size=batch_size, max_new=max_new)
    meta.update({"bfcl_adapter": "native_qwen_xml -> upstream ast_parse(ReturnFormat.PYTHON) -> upstream ast_checker",
                 "bfcl_optional_soundfile_stubbed": soundfile_stubbed})
    pilot.write_new_json(out, {"_meta": meta, "items": {"bfcl": records}})
    correct = sum(row["correct"] for row in records)
    truncated = sum(row["truncated"] for row in records)
    pilot.write_new_json(summary, {
        "n": len(records), "correct": correct, "acc": correct / len(records),
        "truncated_n": truncated, "error_type_counts": errors,
        "items_sha256": pilot.sha256_file(out),
    })


def static_preflight(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if not BFCL_PYTHON.is_file() or not GORILLA.is_dir():
        raise RuntimeError("pinned BFCL checkout or bfcl venv missing")
    missing = [arm["model_path"] for arm in arms if not Path(arm["model_path"]).is_dir()]
    if missing:
        raise RuntimeError(f"HF model directory unavailable: {missing}")
    # Keep preflight entirely CPU-only while checking the exact upstream checker.
    import subprocess
    bfcl_import = subprocess.run(
        [str(BFCL_PYTHON), "-c", (
            "import sys, types; sys.modules.setdefault('soundfile', types.ModuleType('soundfile')); "
            "from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker; "
            "import bfcl_eval; print(bfcl_eval.__file__)"
        )], env=pilot.source_env(), check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"checked_at": dt.datetime.now(dt.UTC).isoformat(), "arms": arms,
            "gorilla_checkout": pilot.git_head(GORILLA.parent), "bfcl_import": bfcl_import,
            "physical_gpus_requested": [0, 1],
            "gpu2_excluded_by": "generation workers are fixed to GPU 0 and GPU 1",
            "network": "offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"}


def invoke_bfcl(mode: str, run_dir: Path, *, arm: dict[str, Any] | None = None,
                n: int | None = None, batch_size: int | None = None,
                max_new: int | None = None) -> None:
    argv = [str(BFCL_PYTHON), str(Path(__file__)), mode, "--run-dir", str(run_dir)]
    if arm is not None:
        argv += ["--arm", arm_to_cli(arm)]
    if n is not None:
        argv += ["--n", str(n)]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    if max_new is not None:
        argv += ["--max-new", str(max_new)]
    import subprocess
    subprocess.run(argv, env=pilot.source_env(), check=True)


def arm_to_cli(arm: dict[str, Any]) -> str:
    return f"{arm['name']}={arm['model_path']}:{arm['topk']}:{arm['alpha']!r}"


def campaign(run_id: str, arms: list[dict[str, Any]], n: int, batch_size: int,
             max_new: int, noninferiority_margin: float) -> Path:
    if len(arms) < 2:
        raise ValueError("campaign requires at least two --arm values for paired comparisons")
    if not 0 <= noninferiority_margin <= 1:
        raise ValueError("noninferiority margin must be within [0, 1]")
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
    audit = static_preflight(arms)
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "running", "run_id": run_id,
        "benchmark": "bfcl_v4_nonlive_ast", "n": n,
        "noninferiority_margin": noninferiority_margin,
        "protocol": {"seed": 0, "shuffle": True, "batch_size": batch_size,
                     "max_new": max_new, "categories": list(CATEGORIES),
                     "arms": arms, "serial": True}, "preflight": audit,
        "started_at": dt.datetime.now(dt.UTC).isoformat(), "arms": {},
    }
    pilot.write_new_json(run_dir / "manifest.json", manifest)
    try:
        invoke_bfcl("prepare", run_dir, n=n)
        for arm in arms:
            name = arm["name"]
            manifest["arms"][name] = {"status": "running", "started_at": dt.datetime.now(dt.UTC).isoformat()}
            _rewrite_manifest(run_dir, manifest)
            generate(run_dir, arm, batch_size, max_new)
            invoke_bfcl("score", run_dir, arm=arm, batch_size=batch_size, max_new=max_new)
            manifest["arms"][name] = {"status": "complete", "finished_at": dt.datetime.now(dt.UTC).isoformat(),
                                      "summary": pilot.read_json(run_dir / f"{name}_summary.json")}
            _rewrite_manifest(run_dir, manifest)
        sys.path.insert(0, str(ESFT))
        from eval_harness import noninferiority_verdict, paired_verdict
        comparisons = {}
        for i, left in enumerate(arms):
            a = pilot.read_json(run_dir / f"{left['name']}_items.json")["items"]["bfcl"]
            for right in arms[i + 1:]:
                b = pilot.read_json(run_dir / f"{right['name']}_items.json")["items"]["bfcl"]
                verdicts = {key: paired_verdict(a, b, key=key) for key in ("correct", "truncated")}
                verdicts["correct"]["noninferiority"] = noninferiority_verdict(
                    verdicts["correct"], noninferiority_margin)
                comparisons[f"{left['name']}__vs__{right['name']}"] = verdicts
        manifest["paired_verdicts"] = comparisons
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
        _rewrite_manifest(run_dir, manifest)
    return run_dir


def _rewrite_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    path.unlink()
    pilot.write_new_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p = sub.add_parser("score")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-new", type=int, required=True)
    p = sub.add_parser("campaign")
    p.add_argument("--run-id", required=True)
    p.add_argument("--arm", action="append", required=True,
                   help="repeat: name=model_path:topk:alpha")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    p.add_argument("--noninferiority-margin", type=float, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        pilot.prepare(args.run_dir, args.n)
    elif args.command == "score":
        score(args.run_dir, parse_arm(args.arm), args.batch_size, args.max_new)
    else:
        path = campaign(args.run_id, parse_arms(args.arm), args.n, args.batch_size,
                        args.max_new, args.noninferiority_margin)
        print(f"BFCL campaign complete: {path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
