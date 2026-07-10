#!/usr/bin/env python3
"""Verify watcher transitions and terminal-event deduplication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import watcher


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="job-watcher-smoke-") as temporary:
        root = Path(temporary)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"status": "complete"}) + "\n")
        config = {
            "jobs": [
                {
                    "id": "still-running",
                    "label": "Still running smoke job",
                    "kind": "local_manifest",
                    "pid": os.getpid(),
                    "cmd_contains": "watcher_smoke.py",
                    "manifest_path": str(manifest),
                },
                {
                    "id": "just-completed",
                    "label": "Completed smoke job",
                    "kind": "local_manifest",
                    "pid": 999_999_999,
                    "cmd_contains": "cannot-match",
                    "manifest_path": str(manifest),
                },
            ]
        }
        state_path = root / "state.json"
        first = watcher.run_once(config, state_path, root)
        if first["jobs"]["still-running"]["state"] != "running":
            raise AssertionError("live process was not classified as running")
        if first["jobs"]["just-completed"]["state"] != "completed":
            raise AssertionError("finished manifest was not classified as completed")
        events = sorted((root / "events").glob("*.json"))
        if events:
            raise AssertionError("terminal event was emitted without confirmation")

        second = watcher.run_once(config, state_path, root)
        if len(list((root / "events").glob("*.json"))) != 1:
            raise AssertionError("confirmed terminal event was not emitted")
        if not second["jobs"]["just-completed"].get("terminal_event_id"):
            raise AssertionError("terminal event id was not retained")

        watcher.run_once(config, state_path, root)
        if len(list((root / "events").glob("*.json"))) != 1:
            raise AssertionError("terminal event was emitted more than once")

        state_path.unlink()
        watcher.run_once(config, state_path, root)
        watcher.run_once(config, state_path, root)
        if len(list((root / "events").glob("*.json"))) != 1:
            raise AssertionError("state loss created a duplicate terminal event")

        manifest.write_text(json.dumps({"status": "failed"}) + "\n")
        watcher.run_once(config, state_path, root)
        corrected = watcher.run_once(config, state_path, root)
        if len(list((root / "events").glob("*.json"))) != 2:
            raise AssertionError("terminal state change did not create a correction event")
        if corrected["jobs"]["just-completed"]["state"] != "failed":
            raise AssertionError("corrected terminal state was not retained")
    print(json.dumps({"ok": True, "checks": 8}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
