#!/usr/bin/env python3
"""Exercise the MCP server over its real newline-delimited stdio protocol."""

from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
SERVER = HERE / "dist" / "server.js"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="job-events-smoke-") as temporary:
        root = Path(temporary)
        process = subprocess.Popen(
            ["node", str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={"PATH": "/usr/bin:/bin", "JOB_EVENTS_ROOT": str(root)},
        )
        assert process.stdin is not None
        assert process.stdout is not None
        next_id = 1
        notifications: list[dict[str, Any]] = []

        def send(message: dict[str, Any]) -> None:
            process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            process.stdin.flush()

        def request(method: str, params: dict[str, Any] | None = None) -> Any:
            nonlocal next_id
            request_id = next_id
            next_id += 1
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            deadline = time.monotonic() + 7
            while time.monotonic() < deadline:
                ready, _, _ = select.select([process.stdout], [], [], 0.25)
                if not ready:
                    continue
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f"server exited during {method}: {process.poll()}")
                response = json.loads(line)
                if response.get("id") is None:
                    notifications.append(response)
                    continue
                if response.get("id") != request_id:
                    continue
                if "error" in response:
                    raise RuntimeError(f"{method} failed: {response['error']}")
                return response["result"]
            raise TimeoutError(f"request timed out: {method}")

        try:
            request(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "job-events-smoke", "version": "0.1.0"},
                },
            )
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )

            tools = request("tools/list")["tools"]
            names = sorted(tool["name"] for tool in tools)
            expected = {
                "ack_job_event",
                "job_watch_status",
                "list_job_events",
                "wait_for_job_event",
            }
            if not expected.issubset(names):
                raise AssertionError(f"missing tools: {sorted(expected - set(names))}")

            request("tools/call", {"name": "job_watch_status", "arguments": {}})
            request(
                "tools/call",
                {"name": "wait_for_job_event", "arguments": {"timeout_seconds": 1}},
            )
            request("resources/subscribe", {"uri": "jobs://events/unread"})

            event = {
                "id": "smoke-event",
                "job_id": "smoke-job",
                "status": "completed",
                "observed_at": "2026-07-10T00:00:00+00:00",
                "summary": "Synthetic completion event.",
            }
            events_dir = root / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            (events_dir / "smoke-event.json").write_text(json.dumps(event) + "\n")
            time.sleep(1.2)

            listed = request(
                "tools/call", {"name": "list_job_events", "arguments": {}}
            )
            unread = json.loads(listed["content"][0]["text"])
            if [item["id"] for item in unread] != ["smoke-event"]:
                raise AssertionError(f"unexpected unread events: {unread}")
            if not any(
                item.get("method") == "notifications/resources/updated"
                and item.get("params", {}).get("uri") == "jobs://events/unread"
                for item in notifications
            ):
                raise AssertionError("subscribed resource update was not delivered")

            request(
                "tools/call",
                {
                    "name": "ack_job_event",
                    "arguments": {"event_id": "smoke-event"},
                },
            )
            if not any(
                item.get("method") == "notifications/resources/updated"
                for item in notifications[1:]
            ):
                raise AssertionError("acknowledgement resource update was not delivered")
            listed = request(
                "tools/call", {"name": "list_job_events", "arguments": {}}
            )
            if json.loads(listed["content"][0]["text"]) != []:
                raise AssertionError("acknowledged event remained unread")
            invalid_ack = request(
                "tools/call",
                {
                    "name": "ack_job_event",
                    "arguments": {"event_id": "../state"},
                },
            )
            if not invalid_ack.get("isError"):
                raise AssertionError("unsafe event id was accepted")
            request("resources/unsubscribe", {"uri": "jobs://events/unread"})

            print(json.dumps({"ok": True, "tools": names}, indent=2))
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
