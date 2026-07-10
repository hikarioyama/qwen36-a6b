#!/usr/bin/env python3
"""Run the frozen gpu-host B2-1000 preservation evaluation.

This is deliberately campaign-specific.  It supervises three benchmark
processes per wave, validates every input and output, and never starts the B2
wave until all base outputs have passed validation and GPUs are idle again.
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
import time
from typing import Any


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE if (HERE / "codex_harness.py").is_file() else HERE.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import codex_harness as base_harness  # noqa: E402
import eval_harness  # noqa: E402


BENCHMARKS = ("mmlu", "gsm8k", "humaneval")
ARMS = ("base_k8", "b2_1000_k32")
EXPECTED_RUN_ID = "b2_1000_eval_20260710_v1"
EXPECTED_RUN_ROOT = Path(
    "/mnt/docker-raid/models/esft/codex_runs/b2_1000_eval_20260710_v1"
)
EXPECTED_STOCK_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
EXPECTED_STOCK_FINGERPRINT = (
    "3a1ca2a61e9a86af44c5114d72a9033504d3a20e27c3c6838f4162b87e3aa315"
)
EXPECTED_PATCH_SHA256 = (
    "c1b3f041051e9c184e5a3ea14126f921e3a2619b29454e3e73b96f79f45199d3"
)
EXPECTED_PATCH_SIZE = 5_240_961_944
EXPECTED_PATCH_TENSORS = 1_666
EXPECTED_EVALUATOR_HASHES = {
    "eval_harness.py": "0beed4f5bcef48ee811f57c9eadfcf663bcaa425f980bb6d12dd2e8d961c6bce",
    "esft_qwen/common.py": "206af106a400f5608746f080cdcc995a09de42301b1995618fe5c8aa6412f3b5",
    "esft_qwen/esft_patch.py": "fbf13ac7efd9a2fbd6cfa3cc5dfd2410b951aefb9466a43fa274d97132ecf927",
}
EXPECTED_GPU_PAIRS = {
    "mmlu": [0, 1],
    "gsm8k": [2, 3],
    "humaneval": [4, 5, 6, 7],
}
EXPECTED_PROTOCOLS = {
    "mmlu": {
        "n": 600,
        "seed": 0,
        "shuffle": True,
        "batch_size": 16,
        "max_new": 0,
        "no_think": True,
        "choice_logprob": True,
        "noninferiority_margin": 0.01,
    },
    "gsm8k": {
        "n": 600,
        "seed": 0,
        "shuffle": True,
        "batch_size": 16,
        "max_new": 2048,
        "no_think": True,
        "choice_logprob": False,
        "noninferiority_margin": 0.02,
    },
    "humaneval": {
        "n": 164,
        "seed": 0,
        "shuffle": True,
        "batch_size": 8,
        "max_new": 4096,
        "no_think": False,
        "choice_logprob": False,
        "noninferiority_margin": 0.05,
    },
}
EXPECTED_DATASET_HASHES = {
    "mmlu": "45381e9cd163388f409e385edb446daf30209708afd7996df0bc6398c4a5275f",
    "gsm8k": "9956b5c7f18d42c30e4355a3e1ab4216b335e77c726f7f25ebf46e304f78327e",
    "humaneval": "8cf3d4a36641d9890353e12b6f0e2e98cdc807a39e970418435ecb95d7f719c2",
}
EXPECTED_GPU_UUIDS = {
    "0": "GPU-8c2aa6bd-dbd9-69df-9285-6af673f3f80d",
    "1": "GPU-b1c8023e-3621-cd5d-1407-588525cd38a4",
    "2": "GPU-6978b4b4-293c-3cc0-3ab3-855b3d1b372c",
    "3": "GPU-fe8716e5-8985-6320-7025-5db6fcb7c2c1",
    "4": "GPU-9433cf9b-f5f8-ada9-090f-28deebec15f2",
    "5": "GPU-e704a9d4-b503-27d3-4827-6402f1b1307a",
    "6": "GPU-dc9b2149-2ce4-9873-2f19-f083f90de616",
    "7": "GPU-2dbf4efc-0042-10a0-0d58-7337325097a9",
}
LOG_FAILURE_RE = re.compile(
    r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory|"
    r"evaluation worker exited before reporting|all evaluation workers exited|"
    r"\bnan\b|non[- ]finite",
    re.IGNORECASE,
)
XID_RE = re.compile(r"(?:NVRM:\s*)?Xid(?:\s*\(|\s*:)", re.IGNORECASE)


class CampaignError(RuntimeError):
    """A hard failure that invalidates the campaign."""


class CampaignSignal(CampaignError):
    """The controlling process received a termination signal."""


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    atomic_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _exact_dict(actual: Any, expected: dict[str, Any], label: str) -> None:
    if actual != expected:
        raise CampaignError(f"{label} differs from the frozen value: {actual!r}")


def load_config(path: Path) -> dict[str, Any]:
    cfg = read_json(path.resolve())
    if cfg.get("schema_version") != 1:
        raise CampaignError(f"unsupported config schema: {cfg.get('schema_version')!r}")
    for section in (
        "run", "runtime", "stock", "patches", "protocols", "gpu_pairs",
        "source_sha256", "expected_packages", "dependencies", "dataset_hashes",
    ):
        if not isinstance(cfg.get(section), dict):
            raise CampaignError(f"missing config object: {section}")

    required_runtime = {
        "expected_hostname": "gpu-host",
        "expected_gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "expected_driver_version": "595.71.05",
        "expected_compute_capability": "12.0",
    }
    for key, expected in required_runtime.items():
        if cfg["runtime"].get(key) != expected:
            raise CampaignError(f"runtime.{key} is not frozen to {expected!r}")

    run = cfg["run"]
    if run.get("id") != EXPECTED_RUN_ID:
        raise CampaignError(f"unexpected run id: {run.get('id')!r}")
    if Path(run["root"]).resolve() != EXPECTED_RUN_ROOT:
        raise CampaignError(f"unexpected run root: {run['root']}")
    if CODE_ROOT.resolve() != EXPECTED_RUN_ROOT / "code":
        raise CampaignError(f"wrapper is not in the isolated code directory: {CODE_ROOT}")
    _exact_dict(cfg["gpu_pairs"], EXPECTED_GPU_PAIRS, "gpu_pairs")
    _exact_dict(cfg["runtime"].get("expected_gpu_uuids"), EXPECTED_GPU_UUIDS,
                "expected_gpu_uuids")
    _exact_dict(cfg["protocols"], EXPECTED_PROTOCOLS, "protocols")
    _exact_dict(cfg["dataset_hashes"], EXPECTED_DATASET_HASHES, "dataset_hashes")

    stock = cfg["stock"]
    if (stock.get("revision") != EXPECTED_STOCK_REVISION
            or stock.get("fingerprint_sha256") != EXPECTED_STOCK_FINGERPRINT):
        raise CampaignError("stock identity constants do not match the frozen job")
    patch = cfg["patches"].get("b2")
    if not isinstance(patch, dict):
        raise CampaignError("missing patches.b2")
    if (patch.get("sha256") != EXPECTED_PATCH_SHA256
            or int(patch.get("size_bytes", -1)) != EXPECTED_PATCH_SIZE
            or int(patch.get("tensor_count", -1)) != EXPECTED_PATCH_TENSORS
            or int(patch.get("router_tensor_count", -1)) != 0):
        raise CampaignError("B2 patch identity constants do not match the frozen job")
    for name, digest in EXPECTED_EVALUATOR_HASHES.items():
        if cfg["source_sha256"].get(name) != digest:
            raise CampaignError(f"frozen evaluator hash mismatch in config: {name}")

    python = Path(cfg["runtime"]["python"]).resolve()
    if Path(sys.executable).resolve() != python:
        raise CampaignError(f"run with {python}; current Python is {sys.executable}")
    cfg["_config"] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
    }
    return cfg


def configure_parent_environment(cfg: dict[str, Any]) -> None:
    """Apply the same offline and run-scoped dependency paths to the supervisor."""
    runtime = cfg["runtime"]
    python_path = str(Path(runtime["python_path"]).resolve())
    if python_path not in sys.path:
        sys.path.insert(0, python_path)
    os.environ.update({
        "HF_HOME": runtime["hf_home"],
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": python_path,
    })
    os.environ["PATH"] = (
        runtime["bubblewrap_bin_dir"] + os.pathsep + os.environ.get("PATH", "")
    )
    thread_count = str(runtime["thread_count"])
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = thread_count


def install_signal_handlers() -> None:
    def interrupted(signum: int, _frame: Any) -> None:
        raise CampaignSignal(f"received signal {signal.Signals(signum).name}")

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupted)


def source_paths() -> dict[str, Path]:
    return {
        "eval_harness.py": CODE_ROOT / "eval_harness.py",
        "codex_harness.py": CODE_ROOT / "codex_harness.py",
        "run_b2_1000_eval.py": Path(__file__).resolve(),
        "esft_qwen/common.py": CODE_ROOT / "esft_qwen" / "common.py",
        "esft_qwen/esft_patch.py": CODE_ROOT / "esft_qwen" / "esft_patch.py",
    }


def verify_sources(cfg: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, path in source_paths().items():
        if not path.is_file():
            raise CampaignError(f"deployed source is missing: {path}")
        observed[name] = sha256_file(path)
        expected = cfg["source_sha256"].get(name)
        if observed[name] != expected:
            raise CampaignError(
                f"deployed source hash mismatch for {name}: {observed[name]} != {expected}"
            )
    return observed


def package_versions() -> dict[str, str | None]:
    names = (
        "torch", "transformers", "datasets", "safetensors", "numpy",
        "accelerate", "scipy",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def verify_packages(cfg: dict[str, Any]) -> dict[str, str | None]:
    observed = package_versions()
    expected = cfg["expected_packages"]
    if observed != expected:
        raise CampaignError(f"runtime packages changed: {observed!r} != {expected!r}")
    try:
        from scipy.stats import beta

        probe = float(beta.ppf(0.025, 2, 5))
    except Exception as exc:  # noqa: BLE001
        raise CampaignError(f"SciPy exact-bound import/calculation failed: {exc}") from exc
    if not math.isfinite(probe) or not 0.0 < probe < 1.0:
        raise CampaignError(f"SciPy exact-bound probe returned {probe!r}")
    return {**observed, "scipy_beta_probe": probe}


def verify_dependencies(cfg: dict[str, Any]) -> dict[str, Any]:
    dep = cfg["dependencies"]
    bwrap = Path(dep["bwrap_path"]).resolve()
    if not bwrap.is_file() or not os.access(bwrap, os.X_OK):
        raise CampaignError(f"run-scoped bubblewrap is unavailable: {bwrap}")
    digest = sha256_file(bwrap)
    if digest != dep["bwrap_sha256"]:
        raise CampaignError(f"bubblewrap hash mismatch: {digest} != {dep['bwrap_sha256']}")
    package = Path(dep["bubblewrap_deb_path"]).resolve()
    if not package.is_file() or sha256_file(package) != dep["bubblewrap_deb_sha256"]:
        raise CampaignError("the recorded Ubuntu bubblewrap package changed")
    proc = subprocess.run(
        [str(bwrap), "--version"], text=True, capture_output=True, check=False,
    )
    if proc.returncode != 0 or dep["bwrap_version"] not in proc.stdout:
        raise CampaignError(
            f"unexpected bubblewrap version: rc={proc.returncode} out={proc.stdout!r}"
        )
    return {
        "bwrap_path": str(bwrap),
        "bwrap_sha256": digest,
        "bwrap_version_output": proc.stdout.strip(),
        "bubblewrap_deb_path": str(package),
        "bubblewrap_deb_sha256": sha256_file(package),
    }


def stock_and_patch_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    stock = base_harness.stock_identity(cfg["stock"])
    patch = base_harness.patch_identity(cfg["patches"]["b2"])
    if patch["size_bytes"] != EXPECTED_PATCH_SIZE:
        raise CampaignError(
            f"B2 patch size mismatch: {patch['size_bytes']} != {EXPECTED_PATCH_SIZE}"
        )
    return {"stock": stock, "b2": patch}


def dataset_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for name in BENCHMARKS:
        protocol = cfg["protocols"][name]
        benchmark = eval_harness.get_benchmark(name)
        items = benchmark.load(
            int(protocol["n"]), int(protocol["seed"]), bool(protocol["shuffle"])
        )
        keys = [benchmark.item_key(item) for item in items]
        if len(items) != int(protocol["n"]):
            raise CampaignError(
                f"offline dataset {name} returned {len(items)} items, "
                f"expected {protocol['n']}"
            )
        if len(set(keys)) != len(keys):
            raise CampaignError(f"offline dataset {name} has duplicate stable item keys")
        digest = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()
        identities[name] = {
            "selected_items": len(items),
            "stable_item_keys": len(set(keys)),
            "ordered_item_key_sha256": digest,
            "first_item_key": keys[0],
            "last_item_key": keys[-1],
        }
        if digest != cfg["dataset_hashes"][name]:
            raise CampaignError(
                f"offline dataset identity differs from the frozen cache for {name}"
            )
    return identities


def run_capture(argv: list[str]) -> str:
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise CampaignError(
            f"command failed ({proc.returncode}): {shlex.join(argv)}\n{proc.stdout[-2000:]}"
        )
    return proc.stdout.strip()


def gpu_rows() -> list[dict[str, Any]]:
    output = run_capture([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,ecc.errors.uncorrected.volatile.total",
        "--format=csv,noheader,nounits",
    ])
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(raw) != 10:
            raise CampaignError(f"unexpected nvidia-smi row: {raw!r}")
        try:
            ecc = int(raw[9])
        except ValueError as exc:
            raise CampaignError(f"uncorrectable ECC is not numeric: {raw!r}") from exc
        rows.append({
            "index": int(raw[0]),
            "uuid": raw[1],
            "name": raw[2],
            "driver_version": raw[3],
            "compute_capability": raw[4],
            "memory_total_mib": int(raw[5]),
            "memory_used_mib": int(raw[6]),
            "utilization_percent": int(raw[7]),
            "temperature_c": int(raw[8]),
            "uncorrectable_ecc_volatile": ecc,
        })
    return rows


def compute_processes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uuid_to_index = {row["uuid"]: row["index"] for row in rows}
    output = run_capture([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    processes: list[dict[str, Any]] = []
    if not output.strip():
        return processes
    for raw in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(raw) != 3:
            raise CampaignError(f"unexpected compute-process row: {raw!r}")
        processes.append({
            "pid": int(raw[0]),
            "gpu_uuid": raw[1],
            "gpu_index": uuid_to_index.get(raw[1]),
            "used_memory_mib": int(raw[2]),
        })
    return processes


def descendants(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parents[int(entry.name)] = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                ValueError, IndexError):
            continue
    allowed = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in allowed and pid not in allowed:
                allowed.add(pid)
                changed = True
    return allowed


def active_foreign_evaluators() -> list[dict[str, Any]]:
    allowed = descendants(os.getpid())
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in allowed:
            continue
        try:
            parts = [part.decode(errors="replace") for part in
                     (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = " ".join(parts)
        executable = Path(parts[0]).name if parts else ""
        script_names = {Path(part).name for part in parts[1:] if part.endswith(".py")}
        is_python = executable.startswith("python")
        if is_python and script_names & {"eval_harness.py", "run_b2_1000_eval.py"}:
            active.append({"pid": int(entry.name), "command": command})
    return active


def memory_and_disk(cfg: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            token = rest.strip().split()[0]
            if token.isdigit():
                values[key] = int(token)
    available_gib = values.get("MemAvailable", 0) / 1024 / 1024
    stat = os.statvfs(cfg["run"]["root"])
    disk_free_gib = stat.f_bavail * stat.f_frsize / 1024 ** 3
    return {
        "memory_available_gib": available_gib,
        "disk_free_gib": disk_free_gib,
    }


def xid_messages(since_epoch: float) -> dict[str, Any]:
    proc = subprocess.run(
        ["journalctl", "-k", "--since", f"@{int(since_epoch)}", "--no-pager", "-o", "cat"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if proc.returncode != 0:
        raise CampaignError(f"cannot inspect kernel log for NVIDIA Xid: {proc.stdout[-800:]}")
    matches = [line for line in proc.stdout.splitlines() if XID_RE.search(line)]
    if matches:
        raise CampaignError("NVIDIA Xid detected: " + " | ".join(matches[-5:]))
    return {"journal_available": True, "xid_count": 0}


def system_health(
    cfg: dict[str, Any], *, require_idle: bool, campaign_epoch: float,
    ecc_baseline: dict[int, int] | None,
) -> dict[str, Any]:
    runtime = cfg["runtime"]
    expected_hostname = runtime.get("expected_hostname")
    if expected_hostname and os.uname().nodename != expected_hostname:
        raise CampaignError(
            f"unexpected host: {os.uname().nodename} != {expected_hostname}")
    allocated = {gpu for pair in cfg["gpu_pairs"].values() for gpu in pair}
    rows = gpu_rows()
    by_index = {row["index"]: row for row in rows}
    if not allocated <= set(by_index):
        raise CampaignError(f"allocated GPUs are missing: {sorted(allocated - set(by_index))}")
    selected_rows = [by_index[gpu] for gpu in sorted(allocated)]
    expected_name = runtime.get("expected_gpu_name")
    expected_driver = runtime.get("expected_driver_version")
    if expected_name and any(row["name"] != expected_name for row in selected_rows):
        raise CampaignError(f"unexpected GPU model: {selected_rows!r}")
    expected_uuids = runtime["expected_gpu_uuids"]
    if any(row["uuid"] != expected_uuids[str(row["index"])] for row in selected_rows):
        raise CampaignError(f"unexpected physical GPU UUID: {selected_rows!r}")
    if expected_driver and any(row["driver_version"] != expected_driver for row in selected_rows):
        raise CampaignError(f"unexpected NVIDIA driver: {selected_rows!r}")
    expected_capability = runtime.get("expected_compute_capability")
    if expected_capability and any(
            row["compute_capability"] != expected_capability for row in selected_rows):
        raise CampaignError(f"unexpected GPU compute capability: {selected_rows!r}")
    limits = cfg["runtime"]
    if require_idle:
        busy = [row for row in rows if row["index"] in allocated and (
            row["memory_used_mib"] > int(limits["gpu_idle_max_used_mib"])
            or row["utilization_percent"] > int(limits["gpu_idle_max_utilization_percent"])
        )]
        if busy:
            raise CampaignError(f"allocated GPUs are not idle: {busy!r}")
    hot = [row for row in rows if row["index"] in allocated
           and row["temperature_c"] >= int(limits["gpu_stop_temperature_c"])]
    if hot:
        raise CampaignError(f"GPU temperature stop threshold reached: {hot!r}")

    if require_idle:
        nonzero = [row for row in selected_rows
                   if row["uncorrectable_ecc_volatile"] != 0]
        if nonzero:
            raise CampaignError(f"uncorrectable GPU ECC is already nonzero: {nonzero!r}")
    if ecc_baseline is not None:
        changed = [row for row in rows if row["index"] in allocated and
                   row["uncorrectable_ecc_volatile"] != ecc_baseline[row["index"]]]
        if changed:
            raise CampaignError(f"uncorrectable GPU ECC count increased: {changed!r}")

    processes = compute_processes(rows)
    allocated_processes = [p for p in processes if p["gpu_index"] in allocated]
    if require_idle and allocated_processes:
        raise CampaignError(f"allocated GPUs have compute processes: {allocated_processes!r}")
    if not require_idle:
        allowed = descendants(os.getpid())
        foreign = [p for p in allocated_processes if p["pid"] not in allowed]
        if foreign:
            raise CampaignError(f"foreign process appeared on allocated GPU: {foreign!r}")

    resources = memory_and_disk(cfg)
    minimum_memory = float(
        runtime["min_memory_available_gib_idle" if require_idle
                else "min_memory_available_gib_running"]
    )
    if resources["memory_available_gib"] < minimum_memory:
        raise CampaignError(
            f"available RAM {resources['memory_available_gib']:.1f} GiB "
            f"is below {minimum_memory:.1f} GiB"
        )
    if resources["disk_free_gib"] < float(runtime["min_disk_free_gib"]):
        raise CampaignError(
            f"free disk {resources['disk_free_gib']:.1f} GiB is below the stop threshold"
        )
    foreign_evals = active_foreign_evaluators()
    if foreign_evals:
        raise CampaignError(f"another evaluation campaign is active: {foreign_evals!r}")
    return {
        "checked_at": utc_now(),
        "gpus": rows,
        "compute_processes": allocated_processes,
        "resources": resources,
        "xid": xid_messages(campaign_epoch),
    }


def sandbox_self_test() -> dict[str, Any]:
    try:
        return base_harness.code_sandbox_self_test()
    except Exception as exc:  # noqa: BLE001
        raise CampaignError(f"bubblewrap sandbox self-test failed: {exc}") from exc


def invariant_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "sources": verify_sources(cfg),
        "packages": verify_packages(cfg),
        "dependencies": verify_dependencies(cfg),
        "assets": stock_and_patch_identity(cfg),
        "datasets": dataset_identity(cfg),
        "sandbox": sandbox_self_test(),
    }


def eval_tag(arm: str, benchmark: str) -> str:
    return f"{arm}_{benchmark}"


def protocol_args(protocol: dict[str, Any]) -> list[str]:
    args = [
        "--n", str(protocol["n"]),
        "--seed", str(protocol["seed"]),
        "--batch-size", str(protocol["batch_size"]),
    ]
    if int(protocol["max_new"]) > 0:
        args += ["--max-new", str(protocol["max_new"])]
    if protocol["shuffle"]:
        args.append("--shuffle")
    if protocol["no_think"]:
        args.append("--no-think")
    if protocol["choice_logprob"]:
        args.append("--choice-logprob")
    return args


def eval_command(cfg: dict[str, Any], arm: str, benchmark: str) -> list[str]:
    if arm not in ARMS or benchmark not in BENCHMARKS:
        raise CampaignError(f"unsupported arm/benchmark: {arm}/{benchmark}")
    command = [
        cfg["runtime"]["python"],
        str(CODE_ROOT / "eval_harness.py"),
        "--model", "base" if arm == "base_k8" else "patched",
        "--model-path", cfg["stock"]["path"],
        "--topk", "8" if arm == "base_k8" else "32",
        "--benchmark", benchmark,
        "--gpus", ",".join(str(i) for i in range(len(cfg["gpu_pairs"][benchmark]))),
        "--report-dir", str(Path(cfg["run"]["root"]) / "results"),
        "--tag", eval_tag(arm, benchmark),
        *protocol_args(cfg["protocols"][benchmark]),
    ]
    if arm != "base_k8":
        command += ["--patch", cfg["patches"]["b2"]["path"]]
    return command


def child_environment(cfg: dict[str, Any], benchmark: str) -> dict[str, str]:
    runtime = cfg["runtime"]
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(
            runtime["expected_gpu_uuids"][str(index)]
            for index in cfg["gpu_pairs"][benchmark]
        ),
        "HF_HOME": runtime["hf_home"],
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": runtime["pytorch_cuda_alloc_conf"],
        "PYTHONPATH": runtime["python_path"],
        "PYTHONUNBUFFERED": "1",
    })
    thread_count = str(runtime["thread_count"])
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = thread_count
    env["PATH"] = runtime["bubblewrap_bin_dir"] + os.pathsep + env.get("PATH", "")
    return env


def output_paths(cfg: dict[str, Any], arm: str, benchmark: str) -> tuple[Path, Path]:
    root = Path(cfg["run"]["root"]) / "results"
    tag = eval_tag(arm, benchmark)
    return root / f"{tag}.json", root / f"{tag}_items.json"


def validate_item_details(
    cfg: dict[str, Any], benchmark: str, result_path: Path, items_path: Path,
    expected_dataset: dict[str, Any], process_started_epoch: float,
) -> dict[str, Any]:
    """Apply strict type, order, freshness, and dataset-identity checks."""
    if result_path.stat().st_mtime < process_started_epoch - 1:
        raise CampaignError(f"stale result file predates its process: {result_path}")
    if items_path.stat().st_mtime < process_started_epoch - 1:
        raise CampaignError(f"stale item file predates its process: {items_path}")
    result_doc = read_json(result_path)
    item_doc = read_json(items_path)
    items = item_doc.get("items", {}).get(benchmark)
    if not isinstance(items, list):
        raise CampaignError(f"missing item list for {benchmark}: {items_path}")
    expected_n = int(cfg["protocols"][benchmark]["n"])
    if len(items) != expected_n:
        raise CampaignError(f"strict item count mismatch for {benchmark}")

    ids = [item.get("id") for item in items]
    if ids != list(range(expected_n)):
        raise CampaignError(f"item ids are not the ordered range 0..{expected_n - 1}")
    keys: list[str] = []
    timeout_count = 0
    for index, item in enumerate(items):
        key = item.get("item_key")
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise CampaignError(f"invalid stable item key at {benchmark}[{index}]")
        keys.append(key)
        for field in ("correct", "truncated", "sandbox_timeout"):
            if type(item.get(field)) is not bool:  # bool specifically, not truthy values
                raise CampaignError(f"{benchmark}[{index}].{field} is not boolean")
        gen_len = item.get("gen_len")
        if isinstance(gen_len, bool) or not isinstance(gen_len, int) or gen_len < 0:
            raise CampaignError(f"{benchmark}[{index}].gen_len is not a nonnegative integer")
        timeout_count += int(item["sandbox_timeout"])
    key_digest = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()
    if key_digest != expected_dataset["ordered_item_key_sha256"]:
        raise CampaignError(
            f"result item order/content differs from CPU preflight for {benchmark}"
        )

    aggregate = result_doc.get("results", {}).get(benchmark)
    if not isinstance(aggregate, dict):
        raise CampaignError(f"missing aggregate for {benchmark}")
    correct = sum(item["correct"] for item in items)
    truncated = sum(item["truncated"] for item in items)
    expected_acc = round(correct / expected_n, 4)
    if (type(aggregate.get("correct")) is not int
            or aggregate["correct"] != correct
            or type(aggregate.get("truncated_n")) is not int
            or aggregate["truncated_n"] != truncated
            or type(aggregate.get("sandbox_timeout_n")) is not int
            or aggregate["sandbox_timeout_n"] != timeout_count
            or not isinstance(aggregate.get("acc"), (int, float))
            or isinstance(aggregate.get("acc"), bool)
            or aggregate["acc"] != expected_acc):
        raise CampaignError(f"strict aggregate mismatch for {benchmark}")
    return {
        "ordered_ids": True,
        "typed_items": True,
        "ordered_item_key_sha256": key_digest,
        "sandbox_timeout_n": timeout_count,
        "fresh_outputs": True,
    }


def terminate_processes(processes: dict[str, subprocess.Popen[Any]]) -> None:
    for proc in processes.values():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for proc in processes.values():
        if proc.poll() is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
    for proc in processes.values():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    for proc in processes.values():
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def scan_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    match = LOG_FAILURE_RE.search(text)
    if match:
        start = max(0, match.start() - 300)
        end = min(len(text), match.end() + 1200)
        raise CampaignError(f"failure text in {path.name}:\n{text[start:end]}")


def run_wave(
    cfg: dict[str, Any], arm: str, manifest: dict[str, Any], manifest_path: Path,
    campaign_epoch: float, ecc_baseline: dict[int, int],
) -> None:
    processes: dict[str, subprocess.Popen[Any]] = {}
    log_handles: dict[str, Any] = {}
    started = time.monotonic()
    try:
        for benchmark in BENCHMARKS:
            tag = eval_tag(arm, benchmark)
            command = eval_command(cfg, arm, benchmark)
            log_path = Path(cfg["run"]["root"]) / "logs" / f"{tag}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("x")
            log_handles[tag] = handle
            blocked_signals = {
                signum for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
                if signum is not None
            }
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked_signals)
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=CODE_ROOT,
                    env=child_environment(cfg, benchmark),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes[tag] = proc
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            manifest["arms"][tag] = {
                "status": "running",
                "benchmark": benchmark,
                "arm": arm,
                "physical_gpus": cfg["gpu_pairs"][benchmark],
                "logical_gpus": list(range(len(cfg["gpu_pairs"][benchmark]))),
                "pid": proc.pid,
                "command": command,
                "log": str(log_path),
                "started_at": utc_now(),
                "started_epoch": time.time(),
            }
        atomic_json(manifest, manifest_path)
        print(
            f"[{utc_now()}] {arm} wave started: "
            + ", ".join(f"{tag}=pid{proc.pid}" for tag, proc in processes.items()),
            flush=True,
        )

        monitor_interval = float(cfg["runtime"]["monitor_interval_seconds"])
        next_monitor = time.monotonic()
        while True:
            failed = {tag: proc.returncode for tag, proc in processes.items()
                      if proc.poll() not in (None, 0)}
            if failed:
                raise CampaignError(f"{arm} child process failed: {failed!r}")
            if all(proc.poll() == 0 for proc in processes.values()):
                break
            if time.monotonic() - started > float(cfg["runtime"]["wave_max_seconds"]):
                raise CampaignError(
                    f"{arm} wave exceeded {cfg['runtime']['wave_max_seconds']} seconds")
            if time.monotonic() >= next_monitor:
                health = system_health(
                    cfg, require_idle=False, campaign_epoch=campaign_epoch,
                    ecc_baseline=ecc_baseline,
                )
                manifest["last_health"] = health
                manifest["heartbeat_at"] = utc_now()
                atomic_json(manifest, manifest_path)
                elapsed = int(time.monotonic() - started)
                running = [tag for tag, proc in processes.items() if proc.poll() is None]
                print(f"[{utc_now()}] {arm} running {elapsed}s: {','.join(running)}", flush=True)
                next_monitor = time.monotonic() + monitor_interval
            time.sleep(2)

        for handle in log_handles.values():
            handle.flush()
        for benchmark in BENCHMARKS:
            tag = eval_tag(arm, benchmark)
            scan_log(Path(manifest["arms"][tag]["log"]))
            result_path, items_path = output_paths(cfg, arm, benchmark)
            if not result_path.is_file() or not items_path.is_file():
                raise CampaignError(f"{tag} did not produce both required JSON files")
            report, validation = base_harness.validate_arm_output(
                cfg,
                manifest["initial_preflight"]["validation_audit"],
                benchmark,
                "base_k8" if arm == "base_k8" else "b2_k32",
                result_path,
                items_path,
                logical_gpus=list(range(len(cfg["gpu_pairs"][benchmark]))),
            )
            validation["strict"] = validate_item_details(
                cfg,
                benchmark,
                result_path,
                items_path,
                manifest["initial_preflight"]["datasets"][benchmark],
                float(manifest["arms"][tag]["started_epoch"]),
            )
            manifest["arms"][tag].update({
                "status": "complete",
                "finished_at": utc_now(),
                "returncode": 0,
                "result_path": str(result_path),
                "items_path": str(items_path),
                "result_sha256": sha256_file(result_path),
                "items_sha256": sha256_file(items_path),
                "validation": validation,
                "result": report,
            })
        atomic_json(manifest, manifest_path)
    except BaseException:
        terminate_processes(processes)
        raise
    finally:
        for handle in log_handles.values():
            handle.close()


def wait_for_idle(
    cfg: dict[str, Any], campaign_epoch: float, ecc_baseline: dict[int, int],
) -> dict[str, Any]:
    deadline = time.monotonic() + float(cfg["runtime"]["gpu_idle_wait_seconds"])
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return system_health(
                cfg, require_idle=True, campaign_epoch=campaign_epoch,
                ecc_baseline=ecc_baseline,
            )
        except CampaignError as exc:
            last_error = exc
            if "not idle" not in str(exc) and "compute processes" not in str(exc):
                raise
            time.sleep(5)
    raise CampaignError(f"GPUs did not return to idle state: {last_error}")


def compare_invariants(initial: dict[str, Any], current: dict[str, Any], label: str) -> None:
    if current != initial:
        raise CampaignError(f"frozen inputs changed before {label}")


def paired_results(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    verdicts: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        _, base_items = output_paths(cfg, "base_k8", benchmark)
        _, b2_items = output_paths(cfg, "b2_1000_k32", benchmark)
        verdict = base_harness.structured_verdicts(
            base_items,
            b2_items,
            benchmark,
            float(cfg["protocols"][benchmark]["noninferiority_margin"]),
        )
        verdicts[benchmark] = verdict

    _, human_base_items_path = output_paths(cfg, "base_k8", "humaneval")
    _, human_b2_items_path = output_paths(cfg, "b2_1000_k32", "humaneval")
    human_base, _ = eval_harness._load_items_file(human_base_items_path)
    human_b2, _ = eval_harness._load_items_file(human_b2_items_path)
    verdicts["humaneval"]["sandbox_timeout"] = eval_harness.paired_verdict(
        human_base["humaneval"], human_b2["humaneval"], key="sandbox_timeout"
    )

    correctness = {
        name: verdicts[name]["correct"]["noninferiority"]["status"]
        for name in BENCHMARKS
    }
    truncation_regressions = [
        name for name in BENCHMARKS
        if verdicts[name]["truncated"]["delta"] > 0
        and verdicts[name]["truncated"]["significant"]
    ]
    sandbox_timeout = verdicts["humaneval"]["sandbox_timeout"]
    sandbox_timeout_regression = (
        sandbox_timeout["delta"] > 0 and sandbox_timeout["significant"]
    )
    if any(status == "FAIL" for status in correctness.values()):
        disposition = "REJECT_B2_1000"
    elif (all(status == "PASS" for status in correctness.values())
          and not truncation_regressions and not sandbox_timeout_regression):
        disposition = "AUTO_ADOPT_B2_1000"
    else:
        disposition = "USER_DECISION_REQUIRED"
    decision = {
        "disposition": disposition,
        "correctness_noninferiority": correctness,
        "significant_truncation_increases": truncation_regressions,
        "significant_humaneval_sandbox_timeout_increase": sandbox_timeout_regression,
        "automatic_adoption": disposition == "AUTO_ADOPT_B2_1000",
    }
    return verdicts, decision


@contextlib.contextmanager
def campaign_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(f"campaign lock is held: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


def validation_audit(cfg: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    packages = {name: snapshot["packages"][name] for name in
                ("torch", "transformers", "datasets", "safetensors", "scipy")}
    return {
        "python": {"version": sys.version, "packages": packages},
        "eval_harness": {
            "sha256": snapshot["sources"]["eval_harness.py"],
            "source_sha256": {
                name: snapshot["sources"][name] for name in EXPECTED_EVALUATOR_HASHES
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-sentinel", default="START")
    args = parser.parse_args(argv)

    if (Path(args.start_sentinel).is_absolute()
            or Path(args.start_sentinel).name != args.start_sentinel):
        parser.error("--start-sentinel must be one plain filename")
    install_signal_handlers()

    cfg: dict[str, Any] | None = None
    manifest: dict[str, Any] = {}
    manifest_path: Path | None = None
    marker_path: Path | None = None
    success = False
    manifest_owned = False
    try:
        cfg = load_config(args.config)
        configure_parent_environment(cfg)
        run_root = Path(cfg["run"]["root"]).resolve()
        manifest_path = run_root / cfg["run"]["manifest"]
        marker_path = run_root / cfg["run"]["success_marker"]
        sentinel = run_root / args.start_sentinel
        pid_path = run_root / cfg["run"]["pid_file"]
        campaign_epoch = time.time()

        with campaign_lock(Path(cfg["runtime"]["lock_path"])):
            if (manifest_path.exists() or marker_path.exists() or sentinel.exists()
                    or pid_path.exists() or (run_root / "results").exists()
                    or (run_root / "logs").exists()):
                raise CampaignError(
                    "run state already exists; this run directory may not be reused")
            (run_root / "results").mkdir(exist_ok=False)
            (run_root / "logs").mkdir(exist_ok=False)
            initial_invariants = invariant_snapshot(cfg)
            initial_health = system_health(
                cfg, require_idle=True, campaign_epoch=campaign_epoch, ecc_baseline=None,
            )
            ecc_baseline = {
                row["index"]: row["uncorrectable_ecc_volatile"]
                for row in initial_health["gpus"]
            }
            initial_invariants["validation_audit"] = validation_audit(cfg, initial_invariants)
            commands = {
                arm: {benchmark: eval_command(cfg, arm, benchmark)
                      for benchmark in BENCHMARKS}
                for arm in ARMS
            }
            manifest = {
                "schema_version": 1,
                "status": "waiting_for_start",
                "run_id": cfg["run"]["id"],
                "pid": os.getpid(),
                "config": cfg["_config"],
                "started_at": utc_now(),
                "campaign_epoch": campaign_epoch,
                "frozen_protocols": cfg["protocols"],
                "physical_gpu_pairs": cfg["gpu_pairs"],
                "commands": commands,
                "initial_preflight": initial_invariants,
                "initial_health": initial_health,
                "arms": {},
            }
            manifest_owned = True
            atomic_json(manifest, manifest_path)
            atomic_text(f"{os.getpid()}\n", pid_path)
            print(
                f"B2_1000_EVAL_READY pid={os.getpid()} sentinel={sentinel}",
                flush=True,
            )
            while not sentinel.is_file():
                time.sleep(1)

            manifest["status"] = "running"
            manifest["released_at"] = utc_now()
            atomic_json(manifest, manifest_path)
            print(f"[{utc_now()}] START accepted", flush=True)

            for arm in ARMS:
                before = invariant_snapshot(cfg)
                compare_invariants(
                    {k: v for k, v in initial_invariants.items() if k != "validation_audit"},
                    before,
                    f"{arm} wave",
                )
                health = wait_for_idle(cfg, campaign_epoch, ecc_baseline)
                manifest.setdefault("wave_preflight", {})[arm] = {
                    "checked_at": utc_now(),
                    "invariants": before,
                    "health": health,
                }
                manifest["current_wave"] = arm
                atomic_json(manifest, manifest_path)
                run_wave(
                    cfg, arm, manifest, manifest_path, campaign_epoch, ecc_baseline,
                )
                after = invariant_snapshot(cfg)
                compare_invariants(
                    {k: v for k, v in initial_invariants.items() if k != "validation_audit"},
                    after,
                    f"completion of {arm} wave",
                )
                manifest.setdefault("wave_postflight", {})[arm] = {
                    "checked_at": utc_now(),
                    "invariants": after,
                    "health": wait_for_idle(cfg, campaign_epoch, ecc_baseline),
                }
                atomic_json(manifest, manifest_path)

            verdicts, decision = paired_results(cfg)
            manifest["verdicts"] = verdicts
            manifest["decision"] = decision
            manifest["final_health"] = wait_for_idle(cfg, campaign_epoch, ecc_baseline)
            manifest["status"] = "complete"
            manifest["finished_at"] = utc_now()
            manifest.pop("current_wave", None)
            manifest.pop("last_health", None)
            atomic_json(manifest, manifest_path)
            manifest_sha = sha256_file(manifest_path)
            atomic_text(
                f"B2_1000_EVAL_COMPLETE\nmanifest_sha256={manifest_sha}\n"
                f"disposition={decision['disposition']}\n",
                marker_path,
            )
            success = True
            print(
                f"B2_1000_EVAL_COMPLETE disposition={decision['disposition']} "
                f"manifest_sha256={manifest_sha}",
                flush=True,
            )
        return 0
    except BaseException as exc:  # noqa: BLE001
        if (manifest_owned and manifest_path is not None and not success
                and (marker_path is None or not marker_path.exists())):
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["finished_at"] = utc_now()
            with contextlib.suppress(Exception):
                atomic_json(manifest, manifest_path)
        print(f"B2_1000_EVAL_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
