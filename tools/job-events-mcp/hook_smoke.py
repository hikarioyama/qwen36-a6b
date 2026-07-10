#!/usr/bin/env python3
"""Verify Codex hook injection and stop-continuation behavior."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile


SCRIPT = Path(__file__).with_name("codex_hook.py")


def call(root: Path, payload: dict[str, object]) -> dict[str, object]:
    environment = dict(os.environ)
    environment["JOB_EVENTS_ROOT"] = str(root)
    result = subprocess.run(
        ["python3", str(SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    return json.loads(result.stdout)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="job-hook-smoke-") as temporary:
        root = Path(temporary)
        event_id = "example-run--completed"
        write_json(
            root / "events" / f"{event_id}.json",
            {
                "id": event_id,
                "job_id": "example-run",
                "status": "completed",
                "observed_at": "2026-07-10T00:00:00+00:00",
                "summary": "Example run completed.",
            },
        )

        submitted = call(root, {"hook_event_name": "UserPromptSubmit"})
        context = submitted["hookSpecificOutput"]["additionalContext"]
        assert "[MCP jobEvents]" in context and event_id in context

        stopped = call(root, {"hook_event_name": "Stop", "stop_hook_active": False})
        assert stopped["decision"] == "block" and event_id in stopped["reason"]

        write_json(root / "acks" / f"{event_id}.json", {"event_id": event_id})
        write_json(
            root / "state.json",
            {
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
                "jobs": {"active-run": {"state": "running"}},
            },
        )
        waiting = call(root, {"hook_event_name": "Stop", "stop_hook_active": False})
        assert waiting["decision"] == "block"
        assert "wait_for_job_event" in waiting["reason"]

        repeated = call(root, {"hook_event_name": "Stop", "stop_hook_active": True})
        assert repeated == {}

        write_json(
            root / "state.json",
            {
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
                "jobs": {"active-run": {"state": "completed"}},
            },
        )
        assert call(root, {"hook_event_name": "Stop", "stop_hook_active": False}) == {}

    print(json.dumps({"ok": True, "checks": 5}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
