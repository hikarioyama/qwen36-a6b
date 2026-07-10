#!/usr/bin/env python3
"""Inject durable job events into Codex turns and prevent silent exits."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_ROOT = PROJECT_ROOT / "reports" / "job-events"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def events_root() -> Path:
    configured = os.environ.get("JOB_EVENTS_ROOT")
    return Path(configured).expanduser() if configured else DEFAULT_EVENTS_ROOT


def unread_events(root: Path) -> list[dict[str, Any]]:
    unread: list[dict[str, Any]] = []
    for path in sorted((root / "events").glob("*.json")):
        event = read_json(path, {})
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id:
            continue
        if (root / "acks" / f"{event_id}.json").exists():
            continue
        event["_event_file"] = str(path)
        unread.append(event)
    return unread


def job_state(root: Path) -> tuple[list[str], bool]:
    state_path = root / "state.json"
    state = read_json(state_path, {})
    running = [
        job_id
        for job_id, snapshot in state.get("jobs", {}).items()
        if snapshot.get("state") == "running"
    ]
    updated_at = state.get("updated_at")
    if not updated_at:
        return running, False
    try:
        updated = dt.datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=dt.UTC)
        fresh = (dt.datetime.now(dt.UTC) - updated).total_seconds() <= 120
    except (TypeError, ValueError):
        fresh = False
    return running, fresh


def event_context(events: list[dict[str, Any]]) -> str:
    lines = [
        "Unread durable job events were detected by the project hook.",
        "Before unrelated work, mirror each event visibly to the user with the prefix "
        "[MCP jobEvents]. Show event ID, job ID, status, observed time, summary, and "
        "the local event-file reference. Do not expose long log tails or sensitive paths.",
        "Inspect the artifacts and record the next decision before acknowledging events.",
    ]
    for event in events[:10]:
        lines.append(
            "- "
            + json.dumps(
                {
                    "id": event.get("id"),
                    "job_id": event.get("job_id"),
                    "status": event.get("status"),
                    "observed_at": event.get("observed_at"),
                    "summary": event.get("summary"),
                    "event_file": event.get("_event_file"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        emit({})
        return 0

    root = events_root()
    events = unread_events(root)
    event_name = payload.get("hook_event_name")

    if event_name == "UserPromptSubmit":
        if events:
            emit(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": event_context(events),
                    }
                }
            )
        else:
            emit({})
        return 0

    if event_name == "Stop":
        if payload.get("stop_hook_active"):
            emit({})
            return 0
        if events:
            emit(
                {
                    "decision": "block",
                    "reason": event_context(events),
                }
            )
            return 0
        running, fresh = job_state(root)
        if running:
            action = (
                "Call jobEvents.wait_for_job_event and keep this turn active."
                if fresh
                else "The watcher state is stale; inspect the watcher before ending this turn."
            )
            emit(
                {
                    "decision": "block",
                    "reason": (
                        f"Monitored jobs are still marked running: {', '.join(running)}. "
                        f"{action} Surface any non-empty MCP result visibly with the "
                        "[MCP jobEvents] prefix before acknowledging it."
                    ),
                }
            )
            return 0

    emit({})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
