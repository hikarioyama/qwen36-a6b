#!/usr/bin/env python3
"""Codex-owned launcher for reproducible local A6B evaluation campaigns.

The underlying ``eval_harness.py`` remains the benchmark implementation. This
wrapper owns experiment identity, asset verification, GPU exclusion, provenance,
serial base/B2 execution, and paired verdict collection.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tomllib
from typing import Any


ESFT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ESFT_ROOT.parent
DEFAULT_CONFIG = ESFT_ROOT / "codex_harness.toml"
EVAL_HARNESS = ESFT_ROOT / "eval_harness.py"
LOCK_PATH = Path("/tmp/qwen36-a6b-gpu01.lock")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PreflightError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if cfg.get("schema_version") != 1:
        raise PreflightError(f"unsupported config schema: {cfg.get('schema_version')!r}")
    for section in ("runtime", "stock", "patches", "protocols"):
        if section not in cfg:
            raise PreflightError(f"missing config section [{section}]")
    gpu_ids = [int(part) for part in str(cfg["runtime"]["gpus"]).split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise PreflightError("runtime.gpus must name exactly two distinct physical GPUs")
    cfg["_config_provenance"] = {"path": str(path), "sha256": sha256_file(path)}
    return cfg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(argv: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise PreflightError(
            f"command failed ({proc.returncode}): {shlex.join(argv)}\n{proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = git("status", "--short")
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty_paths": status.splitlines() if status else [],
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "datasets", "safetensors", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def evaluation_source_hashes() -> dict[str, str]:
    paths = {
        "eval_harness.py": EVAL_HARNESS,
        "esft_qwen/common.py": ESFT_ROOT / "esft_qwen" / "common.py",
        "esft_qwen/esft_patch.py": ESFT_ROOT / "esft_qwen" / "esft_patch.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def code_sandbox_self_test() -> dict[str, Any]:
    sys.path.insert(0, str(ESFT_ROOT))
    from eval_harness import _run_sandboxed

    passed, pass_detail, pass_reason = _run_sandboxed("assert 1 + 1 == 2")
    bypassed, _, bypass_reason = _run_sandboxed("raise SystemExit(0)\nassert False")
    source_bypassed, _, source_reason = _run_sandboxed(
        "import os, re\n"
        "src = open(__file__).read()\n"
        "token = re.search(r'__ESFT_TESTS_PASSED_[0-9a-f]+__', src).group(0)\n"
        "print(token, flush=True)\n"
        "os._exit(0)\n"
        "assert False"
    )
    escape_path = Path(f"/tmp/esft_sandbox_escape_probe_{os.getpid()}")
    isolated, isolate_detail, isolate_reason = _run_sandboxed(
        f"open({str(escape_path)!r}, 'w').write('sandbox-only')")
    timed_out, timeout_detail, timeout_reason = _run_sandboxed(
        "while True: pass", timeout=0.2)
    host_write = escape_path.exists()
    if (passed is not True or pass_reason != "pass"
            or bypassed is not False or bypass_reason != "failed"
            or source_bypassed is not False or source_reason != "failed"
            or isolated is not True or host_write
            or timed_out is not False or timeout_reason != "timeout"):
        raise PreflightError(
            "code sandbox self-test failed: "
            f"pass={passed} bypassed={bypassed} source_bypassed={source_bypassed} "
            f"isolated={isolated} "
            f"reasons={pass_reason}/{bypass_reason}/{source_reason}/{isolate_reason} "
            f"timeout={timed_out}/{timeout_reason} host_write={host_write} "
            f"detail={pass_detail or isolate_detail or timeout_detail}"
        )
    return {
        "backend": "bubblewrap",
        "pass_case": passed,
        "system_exit_bypass_blocked": not bypassed,
        "source_read_bypass_blocked": not source_bypassed,
        "host_filesystem_isolated": not host_write,
        "timeout_cleanup": timeout_reason == "timeout",
    }


def stock_identity(stock: dict[str, Any]) -> dict[str, Any]:
    from safetensors import safe_open

    root = Path(stock["path"]).resolve()
    forbidden = Path(stock["forbidden_path"]).resolve()
    if root == forbidden:
        raise PreflightError(f"configured stock path is forbidden merged model: {root}")
    if not root.is_dir():
        raise PreflightError(f"stock snapshot not found: {root}")
    if root.name != stock["revision"]:
        raise PreflightError(
            f"stock revision mismatch: path={root.name}, expected={stock['revision']}"
        )

    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid stock index {index_path}: {exc}") from exc
    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    if len(shards) != int(stock["shard_count"]):
        raise PreflightError(
            f"stock shard count mismatch: {len(shards)} != {stock['shard_count']}"
        )
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise PreflightError(f"stock shards missing: {missing[:3]}")

    suffix = stock["fingerprint_key_suffix"]
    keys = [key for key in weight_map if key.endswith(suffix)]
    if len(keys) != 1:
        raise PreflightError(f"fingerprint key suffix matched {len(keys)} tensors: {suffix}")
    key = keys[0]
    with safe_open(root / weight_map[key], framework="pt", device="cpu") as f:
        tensor = f.get_slice(key)[int(stock["fingerprint_expert"])]
        payload = tensor.reshape(-1).float().numpy().tobytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    if fingerprint != stock["fingerprint_sha256"]:
        raise PreflightError(
            "stock tensor fingerprint mismatch: "
            f"{fingerprint} != {stock['fingerprint_sha256']}"
        )
    return {
        "path": str(root),
        "revision": root.name,
        "index_sha256": sha256_file(index_path),
        "weight_count": len(weight_map),
        "shard_count": len(shards),
        "fingerprint_key": key,
        "fingerprint_expert": int(stock["fingerprint_expert"]),
        "fingerprint_sha256": fingerprint,
    }


def patch_identity(patch: dict[str, Any]) -> dict[str, Any]:
    from safetensors import safe_open

    path = Path(patch["path"]).resolve()
    if not path.is_file():
        raise PreflightError(f"B2 patch not found: {path}")
    digest = sha256_file(path)
    if digest != patch["sha256"]:
        raise PreflightError(f"B2 patch SHA-256 mismatch: {digest} != {patch['sha256']}")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    router_count = sum(key.startswith("router.") for key in keys)
    if len(keys) != int(patch["tensor_count"]):
        raise PreflightError(
            f"B2 patch tensor count mismatch: {len(keys)} != {patch['tensor_count']}"
        )
    if router_count != int(patch["router_tensor_count"]):
        raise PreflightError(
            "B2 patch router tensor count mismatch: "
            f"{router_count} != {patch['router_tensor_count']}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "tensor_count": len(keys),
        "router_tensor_count": router_count,
    }


def active_eval_processes() -> list[dict[str, Any]]:
    active = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        parts = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        if not parts:
            continue
        command = " ".join(parts)
        executable = Path(parts[0]).name
        is_python = executable.startswith("python")
        is_eval = is_python and any(part.endswith("eval_harness.py") for part in parts[1:])
        is_campaign = is_python and any(part.endswith("codex_harness.py") for part in parts[1:]) \
            and "campaign" in parts[1:]
        if is_eval or is_campaign:
            active.append({"pid": int(entry.name), "command": command})
    return sorted(active, key=lambda item: item["pid"])


def gpu_inventory(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            raise PreflightError(f"unexpected nvidia-smi row: {row!r}")
        rows.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "memory_total_mib": int(row[2]),
                "memory_used_mib": int(row[3]),
                "utilization_percent": int(row[4]),
            }
        )
    requested = {int(part) for part in str(runtime["gpus"]).split(",")}
    found = {row["index"] for row in rows}
    if not requested <= found:
        raise PreflightError(f"configured GPUs missing: {sorted(requested - found)}")
    limit = int(runtime["gpu_max_used_mib"])
    utilization_limit = int(runtime["gpu_max_utilization_percent"])
    busy = [
        row for row in rows
        if row["index"] in requested
        and (row["memory_used_mib"] > limit
             or row["utilization_percent"] > utilization_limit)
    ]
    if busy:
        detail = ", ".join(
            f"gpu{row['index']}={row['memory_used_mib']}MiB" for row in busy
        )
        raise PreflightError(f"configured evaluation GPUs are busy: {detail}")
    return rows


def preflight(cfg: dict[str, Any], *, include_gpu: bool = True) -> dict[str, Any]:
    runtime = cfg["runtime"]
    python = Path(runtime["python"]).resolve()
    if not python.is_file():
        raise PreflightError(f"configured Python not found: {python}")
    if Path(sys.executable).resolve() != python:
        raise PreflightError(
            f"run this launcher with {python}; current interpreter is {sys.executable}"
        )
    if not EVAL_HARNESS.is_file():
        raise PreflightError(f"evaluation harness missing: {EVAL_HARNESS}")

    active = active_eval_processes()
    if active:
        summary = ", ".join(str(item["pid"]) for item in active)
        raise PreflightError(f"another evaluation campaign is active (PIDs: {summary})")

    result = {
        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "config": cfg["_config_provenance"],
        "python": {
            "executable": str(python),
            "version": sys.version,
            "packages": package_versions(),
        },
        "git": git_provenance(),
        "eval_harness": {
            "path": str(EVAL_HARNESS),
            "sha256": sha256_file(EVAL_HARNESS),
            "source_sha256": evaluation_source_hashes(),
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "stock": stock_identity(cfg["stock"]),
        "patches": {"b2": patch_identity(cfg["patches"]["b2"])},
        "active_evaluations": [],
    }
    result["gpus"] = gpu_inventory(runtime) if include_gpu else None
    result["code_sandbox"] = code_sandbox_self_test() if include_gpu else None
    return result


def protocol_args(protocol: dict[str, Any]) -> list[str]:
    args = [
        "--n", str(protocol["n"]),
        "--seed", str(protocol["seed"]),
        "--batch-size", str(protocol["batch_size"]),
    ]
    max_new = int(protocol.get("max_new", 0))
    if max_new > 0:
        args += ["--max-new", str(max_new)]
    if protocol.get("shuffle"):
        args.append("--shuffle")
    if protocol.get("no_think"):
        args.append("--no-think")
    if protocol.get("choice_logprob"):
        args.append("--choice-logprob")
    return args


def eval_command(
    cfg: dict[str, Any], benchmark: str, arm: str, run_dir: Path
) -> tuple[list[str], str]:
    runtime = cfg["runtime"]
    tag = f"{arm}_{benchmark}"
    command = [
        runtime["python"],
        str(EVAL_HARNESS),
        "--model", "base" if arm == "base_k8" else "patched",
        "--model-path", cfg["stock"]["path"],
        "--topk", "8" if arm == "base_k8" else "32",
        "--benchmark", benchmark,
        # CUDA_VISIBLE_DEVICES remaps the selected physical pair to logical 0,1.
        "--gpus", "0,1",
        "--report-dir", str(run_dir),
        "--tag", tag,
        *protocol_args(cfg["protocols"][benchmark]),
    ]
    if arm == "b2_k32":
        command += ["--patch", cfg["patches"]["b2"]["path"]]
    return command, tag


def command_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    runtime = cfg["runtime"]
    env["CUDA_DEVICE_ORDER"] = str(runtime["cuda_device_order"])
    env["CUDA_VISIBLE_DEVICES"] = str(runtime["gpus"])
    env["PYTORCH_CUDA_ALLOC_CONF"] = str(runtime["pytorch_cuda_alloc_conf"])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def assert_asset_identity(
    cfg: dict[str, Any], audit: dict[str, Any], arm: str,
) -> dict[str, Any]:
    """Recheck immutable inputs immediately before an evaluation arm starts."""
    identities = {"stock": stock_identity(cfg["stock"])}
    if identities["stock"] != audit["stock"]:
        raise PreflightError(f"stock identity changed after preflight before {arm}")
    if arm == "b2_k32":
        identities["b2"] = patch_identity(cfg["patches"]["b2"])
        if identities["b2"] != audit["patches"]["b2"]:
            raise PreflightError(f"B2 patch identity changed after preflight before {arm}")
    return identities


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_logged(argv: list[str], log_path: Path, env: dict[str, str]) -> None:
    def terminate_group(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    print(f"$ {shlex.join(argv)}", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            argv,
            cwd=ESFT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            returncode = proc.wait()
        except BaseException:
            terminate_group(proc)
            raise
    if returncode != 0:
        terminate_group(proc)
        raise RuntimeError(f"evaluation exited {returncode}; see {log_path}")


def validate_arm_output(
    cfg: dict[str, Any], audit: dict[str, Any], benchmark: str, arm: str,
    result_path: Path, items_path: Path, logical_gpus: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        report = json.loads(result_path.read_text())
        item_report = json.loads(items_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid output for {arm}/{benchmark}: {exc}") from exc

    protocol = cfg["protocols"][benchmark]
    expected_model = "base" if arm == "base_k8" else "patched"
    expected_topk = 8 if arm == "base_k8" else 32
    expected_patch = None if arm == "base_k8" else cfg["patches"]["b2"]["path"]
    expected_max_new = int(protocol["max_new"]) or None
    expected_prompt_mode = (
        "choice_logprob_no_think" if protocol["choice_logprob"]
        else ("generation_no_think" if protocol["no_think"] else "generation_think")
    )
    meta = report.get("_meta", {})
    results = report.get("results", {})
    result = results.get(benchmark)
    errors = []

    expected_meta = {
        "model": expected_model,
        "model_path": cfg["stock"]["path"],
        "patch": expected_patch,
        "topk": expected_topk,
        "benchmarks": [benchmark],
        "n_per_benchmark": int(protocol["n"]),
        "batch_size": int(protocol["batch_size"]),
        "max_new": expected_max_new,
        "seed": int(protocol["seed"]),
        "shuffle": bool(protocol["shuffle"]),
        "gpus": list(logical_gpus or [0, 1]),
        "no_think": bool(protocol["no_think"]),
        "choice_logprob": bool(protocol["choice_logprob"]),
        "effective_prompt_modes": {benchmark: expected_prompt_mode},
        "split": "first-N" + (" shuffled" if protocol["shuffle"] else ""),
        "harness_sha256": audit["eval_harness"]["sha256"],
        "source_sha256": audit["eval_harness"]["source_sha256"],
        "python_executable": cfg["runtime"]["python"],
        "python_version": audit["python"]["version"],
        "package_versions": audit["python"]["packages"],
    }
    for field, expected in expected_meta.items():
        if meta.get(field) != expected:
            errors.append(f"meta.{field}={meta.get(field)!r}, expected={expected!r}")
    if set(results) != {benchmark} or not isinstance(result, dict):
        errors.append(f"results keys={sorted(results)}, expected={[benchmark]}")
    else:
        expected_result = {
            "model": expected_model,
            "topk": expected_topk,
            "benchmark": benchmark,
            "n": int(protocol["n"]),
            "max_new": expected_max_new,
        }
        for field, expected in expected_result.items():
            if result.get(field) != expected:
                errors.append(
                    f"results.{benchmark}.{field}={result.get(field)!r}, "
                    f"expected={expected!r}"
                )

    item_meta = item_report.get("_meta", {})
    items = item_report.get("items", {}).get(benchmark, [])
    if item_meta != meta:
        errors.append("result and item metadata differ")
    expected_n = int(protocol["n"])
    if len(items) != expected_n:
        errors.append(f"item count={len(items)}, expected={expected_n}")
    item_keys = [item.get("item_key") for item in items]
    if any(not key for key in item_keys) or len(set(item_keys)) != len(item_keys):
        errors.append("stable item keys are missing or duplicated")
    if result:
        correct = sum(bool(item.get("correct")) for item in items)
        truncated = sum(bool(item.get("truncated")) for item in items)
        if result.get("correct") != correct:
            errors.append(f"correct aggregate={result.get('correct')}, items={correct}")
        if result.get("truncated_n") != truncated:
            errors.append(
                f"truncation aggregate={result.get('truncated_n')}, items={truncated}"
            )
    if errors:
        raise RuntimeError(
            f"output validation failed for {arm}/{benchmark}: " + "; ".join(errors)
        )
    return report, {
        "item_count": len(items),
        "unique_item_keys": len(set(item_keys)),
        "items_sha256": sha256_file(items_path),
    }


def structured_verdicts(
    base_items: Path, b2_items: Path, benchmark: str, noninferiority_margin: float,
):
    sys.path.insert(0, str(ESFT_ROOT))
    from eval_harness import (
        _load_items_file, _validate_paired_protocol, noninferiority_verdict,
        paired_verdict,
    )

    (base, base_meta), (b2, b2_meta) = (
        _load_items_file(base_items), _load_items_file(b2_items)
    )
    _validate_paired_protocol(base_meta, b2_meta, require_complete=True)
    verdicts = {
        key: paired_verdict(base[benchmark], b2[benchmark], key=key)
        for key in ("correct", "truncated")
    }
    verdicts["correct"]["noninferiority"] = noninferiority_verdict(
        verdicts["correct"], noninferiority_margin)
    return verdicts


@contextlib.contextmanager
def campaign_lock():
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PreflightError(f"GPU campaign lock is already held: {LOCK_PATH}") from exc
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        yield


def default_run_id(benchmark: str) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{benchmark}"


def campaign(
    cfg: dict[str, Any], benchmark: str, *, run_id: str | None, dry_run: bool,
    noninferiority_margin: float | None,
) -> Path | None:
    if benchmark not in cfg["protocols"]:
        raise PreflightError(f"no configured protocol for {benchmark!r}")
    run_id = run_id or default_run_id(benchmark)
    if not RUN_ID_RE.fullmatch(run_id):
        raise PreflightError(f"invalid run id: {run_id!r}")
    if (noninferiority_margin is not None
            and (not math.isfinite(noninferiority_margin)
                 or not 0 <= noninferiority_margin <= 1)):
        raise PreflightError(
            "--noninferiority-margin must be a finite fraction from 0 to 1")
    if not dry_run and noninferiority_margin is None:
        raise PreflightError(
            "real campaigns require a predeclared --noninferiority-margin "
            "expressed as an absolute accuracy fraction"
        )
    run_root = ESFT_ROOT / cfg["runtime"]["run_root"]
    run_dir = run_root / run_id
    commands = [eval_command(cfg, benchmark, arm, run_dir) for arm in ("base_k8", "b2_k32")]

    if dry_run:
        audit = preflight(cfg, include_gpu=False)
        print(json.dumps({
            "preflight": audit,
            "run_dir": str(run_dir),
            "noninferiority_margin": noninferiority_margin,
            "commands": [shlex.join(command) for command, _tag in commands],
        }, indent=2, ensure_ascii=False))
        return None
    if run_dir.exists():
        raise PreflightError(f"run directory already exists: {run_dir}")

    with campaign_lock():
        audit = preflight(cfg, include_gpu=True)
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / "manifest.json"
        env = command_env(cfg)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "benchmark": benchmark,
            "protocol": cfg["protocols"][benchmark],
            "noninferiority_margin": noninferiority_margin,
            "preflight": audit,
            "started_at": dt.datetime.now(dt.UTC).isoformat(),
            "commands": [command for command, _tag in commands],
            "runtime_environment": {
                "physical_gpus": [int(part) for part in cfg["runtime"]["gpus"].split(",")],
                "logical_gpus": [0, 1],
                "CUDA_DEVICE_ORDER": env["CUDA_DEVICE_ORDER"],
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "PYTORCH_CUDA_ALLOC_CONF": env["PYTORCH_CUDA_ALLOC_CONF"],
                "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
            },
            "arms": {},
        }
        atomic_json(manifest, manifest_path)
        try:
            for command, tag in commands:
                started = dt.datetime.now(dt.UTC).isoformat()
                arm = tag.rsplit(f"_{benchmark}", 1)[0]
                manifest["arms"][tag] = {
                    "status": "running",
                    "started_at": started,
                    "asset_identity": assert_asset_identity(cfg, audit, arm),
                }
                atomic_json(manifest, manifest_path)
                run_logged(command, run_dir / f"{tag}.log", env)
                manifest["arms"][tag]["asset_identity_after"] = assert_asset_identity(
                    cfg, audit, arm)
                result_path = run_dir / f"{tag}.json"
                items_path = run_dir / f"{tag}_items.json"
                if not result_path.is_file() or not items_path.is_file():
                    raise RuntimeError(f"evaluation did not produce both result files for {tag}")
                report, validation = validate_arm_output(
                    cfg, audit, benchmark, arm,
                    result_path, items_path,
                )
                manifest["arms"][tag].update({
                    "status": "complete",
                    "finished_at": dt.datetime.now(dt.UTC).isoformat(),
                    "result": report,
                    "validation": validation,
                })
                atomic_json(manifest, manifest_path)

            base_items = run_dir / f"base_k8_{benchmark}_items.json"
            b2_items = run_dir / f"b2_k32_{benchmark}_items.json"
            verdict_text = {}
            for key in ("correct", "truncated"):
                argv = [
                    cfg["runtime"]["python"], str(EVAL_HARNESS),
                    "--paired-verdict", str(base_items), str(b2_items),
                    "--verdict-key", key,
                ]
                verdict_text[key] = run_capture(argv, cwd=ESFT_ROOT)
                print(verdict_text[key], flush=True)
            manifest["verdicts"] = structured_verdicts(
                base_items, b2_items, benchmark, noninferiority_margin)
            manifest["verdict_text"] = verdict_text
            manifest["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            manifest["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
            atomic_json(manifest, manifest_path)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="verify local assets and optional GPU readiness")
    pre.add_argument("--skip-gpu", action="store_true")
    pre.add_argument("--json", action="store_true")

    camp = sub.add_parser("campaign", help="run fresh serial base@k8 and B2@k32 arms")
    camp.add_argument("benchmark", choices=["mmlu", "jmmlu", "gsm8k", "humaneval"])
    camp.add_argument("--run-id")
    camp.add_argument("--dry-run", action="store_true")
    camp.add_argument(
        "--noninferiority-margin", type=float,
        help="predeclared absolute accuracy margin, e.g. 0.01 for one point",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = load_config(args.config.resolve())
        if args.command == "preflight":
            result = preflight(cfg, include_gpu=not args.skip_gpu)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("PRECHECK PASS")
                print(f"stock: {result['stock']['revision']} {result['stock']['fingerprint_sha256'][:16]}")
                print(f"B2: {result['patches']['b2']['sha256'][:16]} ({result['patches']['b2']['tensor_count']} tensors)")
                print(f"harness: {result['eval_harness']['sha256'][:16]}")
                if result["gpus"] is not None:
                    selected = {int(x) for x in cfg["runtime"]["gpus"].split(",")}
                    ready = [row for row in result["gpus"] if row["index"] in selected]
                    print("gpus: " + ", ".join(
                        f"{row['index']}={row['memory_used_mib']}MiB" for row in ready
                    ))
            return 0
        run_dir = campaign(
            cfg,
            args.benchmark,
            run_id=args.run_id,
            dry_run=args.dry_run,
            noninferiority_margin=args.noninferiority_margin,
        )
        if run_dir is not None:
            print(f"campaign complete: {run_dir}")
        return 0
    except (PreflightError, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
