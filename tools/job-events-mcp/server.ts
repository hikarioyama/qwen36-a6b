#!/usr/bin/env bun

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  SubscribeRequestSchema,
  UnsubscribeRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(
  process.env.JOB_EVENTS_ROOT ?? "reports/job-events",
);
const EVENTS_DIR = path.join(ROOT, "events");
const ACKS_DIR = path.join(ROOT, "acks");
const STATE_PATH = path.join(ROOT, "state.json");
const UNREAD_URI = "jobs://events/unread";
const subscriptions = new Set<string>();
const SAFE_EVENT_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/;

type JobEvent = {
  id: string;
  job_id: string;
  status: "completed" | "failed";
  observed_at: string;
  summary: string;
  snapshot?: unknown;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function ensureLayout(): Promise<void> {
  await mkdir(EVENTS_DIR, { recursive: true });
  await mkdir(ACKS_DIR, { recursive: true });
}

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(file, "utf8")) as T;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return null;
    throw error;
  }
}

async function eventFiles(): Promise<string[]> {
  await ensureLayout();
  return (await readdir(EVENTS_DIR))
    .filter((name) => name.endsWith(".json"))
    .sort();
}

async function isAcknowledged(id: string): Promise<boolean> {
  return (await readJson(path.join(ACKS_DIR, `${id}.json`))) !== null;
}

async function listEvents(includeAcknowledged = false): Promise<JobEvent[]> {
  const events: JobEvent[] = [];
  for (const name of await eventFiles()) {
    const event = await readJson<JobEvent>(path.join(EVENTS_DIR, name));
    if (!event) continue;
    if (!includeAcknowledged && (await isAcknowledged(event.id))) continue;
    events.push(event);
  }
  return events;
}

async function atomicJson(file: string, value: unknown): Promise<void> {
  const temp = `${file}.${process.pid}.tmp`;
  await writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temp, file);
}

function textResult(value: unknown) {
  return {
    content: [
      {
        type: "text" as const,
        text: typeof value === "string" ? value : JSON.stringify(value, null, 2),
      },
    ],
  };
}

const server = new Server(
  { name: "qwen36-a6b-job-events", version: "0.1.0" },
  {
    capabilities: { resources: { subscribe: true }, tools: {} },
    instructions:
      "This server reports durable completion/failure events for long-running Qwen3.6 A6B GPU jobs. " +
      "Use wait_for_job_event while actively monitoring. Acknowledge an event only after its " +
      "result has been inspected and the next action has been recorded.",
  },
);

const TOOLS = [
  {
    name: "list_job_events",
    description: "List durable job completion/failure events, unread by default.",
    inputSchema: {
      type: "object" as const,
      properties: {
        include_acknowledged: { type: "boolean" as const, default: false },
      },
    },
  },
  {
    name: "wait_for_job_event",
    description:
      "Block until an unread job event arrives or the timeout expires. Use this instead of polling.",
    inputSchema: {
      type: "object" as const,
      properties: {
        timeout_seconds: {
          type: "integer" as const,
          minimum: 1,
          maximum: 14400,
          default: 3600,
        },
      },
    },
  },
  {
    name: "ack_job_event",
    description: "Acknowledge one event after Codex has processed it.",
    inputSchema: {
      type: "object" as const,
      properties: { event_id: { type: "string" as const } },
      required: ["event_id"],
    },
  },
  {
    name: "job_watch_status",
    description: "Read the latest watcher state for all armed jobs.",
    inputSchema: { type: "object" as const, properties: {} },
  },
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const args = (request.params.arguments ?? {}) as Record<string, unknown>;
  switch (request.params.name) {
    case "list_job_events":
      return textResult(
        await listEvents(Boolean(args.include_acknowledged ?? false)),
      );
    case "wait_for_job_event": {
      const timeoutSeconds = Math.max(
        1,
        Math.min(14400, Number(args.timeout_seconds ?? 3600)),
      );
      const deadline = Date.now() + timeoutSeconds * 1000;
      while (Date.now() < deadline) {
        const events = await listEvents(false);
        if (events.length > 0) return textResult(events);
        await sleep(500);
      }
      return textResult({ events: [], timed_out: true, timeout_seconds: timeoutSeconds });
    }
    case "ack_job_event": {
      const eventId = String(args.event_id ?? "");
      if (!SAFE_EVENT_ID.test(eventId)) {
        return {
          ...textResult(`Invalid event id: ${eventId}`),
          isError: true,
        };
      }
      const event = await readJson<JobEvent>(path.join(EVENTS_DIR, `${eventId}.json`));
      if (!event || event.id !== eventId) {
        return {
          ...textResult(`Unknown event: ${eventId}`),
          isError: true,
        };
      }
      await atomicJson(path.join(ACKS_DIR, `${eventId}.json`), {
        event_id: eventId,
        acknowledged_at: new Date().toISOString(),
      });
      if (subscriptions.has(UNREAD_URI)) {
        await server.notification({
          method: "notifications/resources/updated",
          params: { uri: UNREAD_URI },
        });
      }
      return textResult({ acknowledged: eventId });
    }
    case "job_watch_status":
      return textResult((await readJson(STATE_PATH)) ?? { jobs: {}, status: "not_started" });
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [
    {
      uri: UNREAD_URI,
      name: "Unread Qwen3.6 A6B job events",
      description: "Durable completion and failure events not yet acknowledged by Codex.",
      mimeType: "application/json",
    },
    {
      uri: "jobs://status",
      name: "Qwen3.6 A6B watcher status",
      description: "Latest process state observed for each armed job.",
      mimeType: "application/json",
    },
  ],
}));

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;
  const value =
    uri === UNREAD_URI
      ? await listEvents(false)
      : uri === "jobs://status"
        ? ((await readJson(STATE_PATH)) ?? { jobs: {}, status: "not_started" })
        : null;
  if (value === null) throw new Error(`Unknown resource: ${uri}`);
  return {
    contents: [
      {
        uri,
        mimeType: "application/json",
        text: `${JSON.stringify(value, null, 2)}\n`,
      },
    ],
  };
});

server.setRequestHandler(SubscribeRequestSchema, async (request) => {
  subscriptions.add(request.params.uri);
  return {};
});

server.setRequestHandler(UnsubscribeRequestSchema, async (request) => {
  subscriptions.delete(request.params.uri);
  return {};
});

async function main(): Promise<void> {
  console.error("[job-events-mcp] starting");
  await ensureLayout();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[job-events-mcp] connected");
  process.stdin.on("end", () => setTimeout(() => process.exit(0), 1000));
  let known = new Set(await eventFiles());
  setInterval(async () => {
    try {
      const current = new Set(await eventFiles());
      const hasNewEvent = [...current].some((name) => !known.has(name));
      if (hasNewEvent && subscriptions.has(UNREAD_URI)) {
        await server.notification({
          method: "notifications/resources/updated",
          params: { uri: UNREAD_URI },
        });
      }
      known = current;
    } catch (error) {
      console.error(`[job-events-mcp] ${String(error)}`);
    }
  }, 1000);
}

main().catch((error) => {
  console.error(`[job-events-mcp] fatal: ${String(error)}`);
  process.exit(1);
});
