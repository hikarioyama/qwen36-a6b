#!/usr/bin/env python3
"""Arm one never-reused local jobEvents identity for a prepared selfgen run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "tools" / "job-events-mcp" / "jobs.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pid", required=True, type=int)
    args = parser.parse_args()
    manifest = ROOT / "esft" / "data" / "selfgen_toolcall_v1" / args.run_id / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"prepared manifest not found: {manifest}")
    config = json.loads(JOBS.read_text(encoding="utf-8"))
    job_id = f"local-selfgen-toolcall-v1-{args.run_id}"
    if any(job.get("id") == job_id for job in config.get("jobs", [])):
        raise SystemExit(f"job id already exists and must never be reused: {job_id}")
    config["jobs"].append({
        "id": job_id,
        "label": f"Local scaffolded selfgen tool-call v1 ({args.run_id})",
        "kind": "local_manifest",
        "pid": args.pid,
        "cmd_contains": "selfgen_toolcall_v1",
        "manifest_path": str(manifest),
    })
    tmp = JOBS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(JOBS)
    subprocess.run(["systemctl", "--user", "restart", "qwen36-a6b-job-events-watcher.service"], check=True)
    subprocess.run(["systemctl", "--user", "is-active", "--quiet", "qwen36-a6b-job-events-watcher.service"], check=True)
    print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
