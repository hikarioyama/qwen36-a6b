#!/usr/bin/env python3
"""Persist terminal events for local and SSH-hosted long-running jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from typing import Any


STOP = False
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def process_matches(pid: int, needle: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return needle in command


def tail(path: Path, limit: int = 6000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def ssh_run(job: dict[str, Any], script: str) -> subprocess.CompletedProcess[str]:
    ssh_args = ["ssh"]
    if job.get("ssh_config"):
        ssh_args.extend(["-F", str(Path(job["ssh_config"]).expanduser())])
    ssh_args.extend([job["host"], script])
    return subprocess.run(
        ssh_args,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(job.get("timeout_seconds", 15)),
        check=False,
    )


def probe_local_manifest(job: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(job["manifest_path"])
    manifest = read_json(manifest_path, {})
    if process_matches(int(job["pid"]), job["cmd_contains"]):
        state = "running"
    elif manifest.get("status") == "complete":
        state = "completed"
    elif manifest.get("status") == "failed":
        state = "failed"
    else:
        state = "failed"
    manifest_summary = {
        key: manifest[key]
        for key in (
            "status",
            "benchmark",
            "started_at",
            "finished_at",
            "noninferiority_margin",
            "verdicts",
            "verdict_text",
        )
        if key in manifest
    }
    return {
        "state": state,
        "pid": int(job["pid"]),
        "manifest_path": str(manifest_path),
        "manifest": manifest_summary,
    }


def remote_script(job: dict[str, Any]) -> str:
    pid = int(job["pid"])
    needle = shlex.quote(job["cmd_contains"])
    success_files = " && ".join(
        f"test -s {shlex.quote(item)}" for item in job.get("success_files", [])
    ) or "true"
    not_before = job.get("success_not_before_epoch")
    if not_before is not None:
        for item in job.get("success_files", []):
            success_files += (
                f" && test \"$(stat -c %Y {shlex.quote(item)})\" "
                f"-ge {int(not_before)}"
            )
    json_check_code = """import json, sys
value = json.load(open(sys.argv[1]))
spec = json.loads(sys.argv[2])
for key in spec.get("keys", []):
    value = value[key]
if "length" in spec:
    assert len(value) == spec["length"]
if "equals" in spec:
    assert value == spec["equals"]
"""
    remote_python = shlex.quote(job.get("remote_python", "python3"))
    for check in job.get("success_json_checks", []):
        spec = {key: value for key, value in check.items() if key != "path"}
        success_files += (
            f" && {remote_python} -c {shlex.quote(json_check_code)} "
            f"{shlex.quote(check['path'])} {shlex.quote(json.dumps(spec))}"
        )
    success_pattern = job.get("success_pattern")
    if success_pattern:
        success_files += (
            f" && grep -F {shlex.quote(success_pattern)} "
            f"{shlex.quote(job['log_path'])} >/dev/null"
        )
        if not_before is not None:
            success_files += (
                f" && test \"$(stat -c %Y {shlex.quote(job['log_path'])})\" "
                f"-ge {int(not_before)}"
            )
    log_path = shlex.quote(job.get("log_path", "/dev/null"))
    return (
        f"if ps -p {pid} -o args= 2>/dev/null | grep -F -- {needle} >/dev/null; "
        "then state=running; "
        f"elif ( {success_files} ) >/dev/null 2>&1; "
        "then state=completed; else state=failed; fi; "
        'printf "STATE=%s\\n" "$state"; '
        f"tail -c 6000 {log_path} 2>/dev/null || true"
    )


def probe_ssh(job: dict[str, Any]) -> dict[str, Any]:
    try:
        result = ssh_run(job, remote_script(job))
    except subprocess.TimeoutExpired as error:
        return {"state": "unknown", "error": f"SSH timeout: {error}"}
    lines = result.stdout.splitlines()
    marker = lines[0] if lines else ""
    state = marker.removeprefix("STATE=") if marker.startswith("STATE=") else "unknown"
    if result.returncode != 0:
        state = "unknown"
    return {
        "state": state,
        "host": job["host"],
        "pid": int(job["pid"]),
        "returncode": result.returncode,
        "summary_tail": "\n".join(lines[1:])[-6000:],
    }


def probe(job: dict[str, Any]) -> dict[str, Any]:
    kind = job["kind"]
    if kind == "local_manifest":
        return probe_local_manifest(job)
    if kind == "ssh_process":
        return probe_ssh(job)
    raise ValueError(f"unknown job kind: {kind}")


def event_summary(job: dict[str, Any], snapshot: dict[str, Any]) -> str:
    state = snapshot["state"]
    location = f" on {job['host']}" if job.get("host") else " locally"
    return f"{job['label']} {state}{location}. Inspect the recorded snapshot and artifacts."


def emit_event(root: Path, job: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    now = dt.datetime.now(dt.UTC)
    event_id = f"{job['id']}--{snapshot['state']}"
    event = {
        "id": event_id,
        "job_id": job["id"],
        "status": snapshot["state"],
        "observed_at": now.isoformat(),
        "summary": event_summary(job, snapshot),
        "snapshot": snapshot,
    }
    atomic_json(root / "events" / f"{event_id}.json", event)
    return event


def run_once(config: dict[str, Any], state_path: Path, root: Path) -> dict[str, Any]:
    old = read_json(state_path, {"jobs": {}})
    old_jobs = old.get("jobs", {})
    now = dt.datetime.now(dt.UTC).isoformat()
    new_jobs: dict[str, Any] = {}
    for job in config["jobs"]:
        try:
            snapshot = probe(job)
        except Exception as error:  # preserve monitoring after one malformed probe
            snapshot = {"state": "unknown", "error": f"{type(error).__name__}: {error}"}
        snapshot["observed_at"] = now
        old_job = old_jobs.get(job["id"], {})
        previous = old_job.get("state")
        terminal = snapshot["state"] in {"completed", "failed"}
        terminal_events = dict(old_job.get("terminal_events", {}))
        confirmation_count = (
            int(old_job.get("terminal_confirmation_count", 0)) + 1
            if terminal and previous == snapshot["state"]
            else (1 if terminal else 0)
        )
        required = max(
            1,
            int(job.get("terminal_confirmations", config.get("terminal_confirmations", 2))),
        )
        if (
            terminal
            and confirmation_count >= required
            and snapshot["state"] not in terminal_events
        ):
            event = emit_event(root, job, snapshot)
            terminal_events[snapshot["state"]] = event["id"]
        if terminal_events:
            snapshot["terminal_events"] = terminal_events
            latest = terminal_events.get(snapshot["state"])
            if latest:
                snapshot["terminal_event_id"] = latest
        snapshot["terminal_confirmation_count"] = confirmation_count
        snapshot["terminal_confirmations_required"] = required
        snapshot["previous_state"] = previous
        new_jobs[job["id"]] = snapshot
    state = {"schema_version": 1, "updated_at": now, "jobs": new_jobs}
    atomic_json(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    job_ids = [job["id"] for job in config["jobs"]]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("job ids must be unique")
    for job_id in job_ids:
        if SAFE_ID.fullmatch(job_id) is None:
            raise ValueError(f"unsafe job id: {job_id!r}")
    root = Path(config["events_root"]).expanduser().resolve()
    state_path = root / "state.json"
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "acks").mkdir(parents=True, exist_ok=True)

    def stop_handler(_signum: int, _frame: Any) -> None:
        global STOP
        STOP = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    interval = max(5, int(config.get("poll_interval_seconds", 30)))
    while not STOP:
        run_once(config, state_path, root)
        if args.once:
            break
        for _ in range(interval):
            if STOP:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
