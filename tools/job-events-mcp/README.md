# Qwen3.6 A6B Job Events MCP

`watcher.py` polls the exact local/SSH process identities in `jobs.json` and emits
one durable JSON event when a job completes or fails. `server.ts` exposes those
events to Codex through MCP.

Tools:

- `wait_for_job_event`: block without polling until an unread event arrives;
- `list_job_events`: list queued events;
- `job_watch_status`: inspect all armed jobs;
- `ack_job_event`: mark an inspected event handled.

MCP notifications do not independently start model inference. Immediate handling
is achieved by keeping a Codex turn active in `wait_for_job_event`; otherwise the
event remains durable and is returned on the next turn. This avoids concurrent
writes caused by launching `codex exec resume` against an already-open TUI thread.

The project hooks add two delivery guarantees on top of the durable queue:

- `UserPromptSubmit` injects unread event summaries into the next turn, even if
  the model forgot to call `list_job_events` first;
- `Stop` keeps a turn alive once when unread events exist or monitored jobs are
  still running, directing Codex to surface the MCP result and wait.

Codex must mirror every non-empty MCP result into visible commentary using the
`[MCP jobEvents]` prefix before acknowledging it. The hook context itself is not
user-visible; this explicit relay is what makes delivery visible in the chat.

## Runtime

- Watcher service: `qwen36-a6b-job-events-watcher.service`
- Job definitions: untracked `jobs.json` (start from `jobs.example.json`)
- Durable queue: `../../reports/job-events/events/`
- Acknowledgements: `../../reports/job-events/acks/`
- Latest state: `../../reports/job-events/state.json`
- Codex MCP name: `jobEvents`

The live configuration is machine-specific and must remain untracked. Create it
with `cp jobs.example.json jobs.json`, then add each long-running job using its
exact process identity, command substring, and positive completion artifacts.
An absent process without the configured artifacts is a failure, not a completion.
Terminal states require two consecutive probes, so delivery can lag process exit
by up to roughly 60 seconds. Failed and completed are separate correction-capable
events. Event IDs are deterministic, which prevents duplicates if watcher state is
lost between event and state writes.

Job IDs are durable run identities and must never be reused. Remote jobs should
set `success_not_before_epoch` and structured `success_json_checks` so stale or
partial artifacts cannot satisfy completion.

## Build and verification

```bash
bun install
bun run smoke
systemctl --user daemon-reload
systemctl --user enable --now qwen36-a6b-job-events-watcher.service
```

After editing `jobs.json`, apply it with:

```bash
systemctl --user restart qwen36-a6b-job-events-watcher.service
```

`bun run smoke` exercises MCP initialization, all four tools, timeout behavior,
event acknowledgement, watcher process detection, and terminal-event
deduplication, plus the Codex delivery hooks. TypeScript is bundled for Node because the MCP SDK's stdio transport
uses Node stream semantics; Bun is used only for dependency management and build.

Codex is registered globally with a 14,430-second MCP tool timeout. A Codex process
that was already open when the server was registered must be restarted once to
load the new MCP tool surface.
