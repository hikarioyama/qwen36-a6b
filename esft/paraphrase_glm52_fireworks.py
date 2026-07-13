#!/usr/bin/env python3
"""GLM-5.2 (Fireworks) paraphrase driver for the intent-level tool-call selfgen.

Reads paraphrase_batch.jsonl and produces the writeback JSONL
({"seed_id", "natural_request"}) that selfgen_toolcall_intent_v1.py
ingest-paraphrase expects.

- Teacher: GLM-5.2 via Fireworks — the clean-lineage (A group) paraphraser
  named in reports/DATA_QUALITY_STRATEGY_20260711.md §4.1.  GPT/Grok text
  must not enter training data, so they are not usable here.
- Local gate: every string in value_literals must appear verbatim in the
  paraphrase (same substring check as value_occurrences in
  selfgen_toolcall_intent_v1.py).  Failures get up to --retries feedback
  rounds listing the missing literals before we give up on the row.
- Rate limits: Fireworks returns 403/429 on bursts (recovers ~8 s), so the
  driver uses bounded concurrency plus exponential backoff.
- Resume: seed_ids already present in the output file are skipped, so the
  driver can be re-run after interruption.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/glm-5p2"

SYSTEM_PROMPT = """You rewrite formal tool-call instructions into natural user requests.
Rules:
1. Write ONE natural request the way a real user would ask (chat or email style, fluent English, first person).
2. NEVER mention internal tool names (like mock_warehouse_001_inspect_1) or internal parameter names (like field_1_1).
3. Every literal value listed by the user MUST appear VERBATIM in your text, character-for-character, including quotes, brackets and commas (e.g. ["tag-3","tag-4"] must appear exactly as ["tag-3","tag-4"]). If a literal is shown wrapped in double quotes (like "UNAVAILABLE" or "item-29"), the double quotes are part of the literal and must appear in your text.
4. Preserve the operational structure: what can run in parallel, what depends on an earlier result, and any instruction about recovering from an error.
5. Do not invent new requirements, values, or constraints.
Return ONLY a JSON object: {"natural_request": "..."}"""


def build_user_prompt(row: dict, missing: list[str] | None) -> str:
    literals = "\n".join(f"- {v}" for v in row["value_literals"])
    msg = (
        "Rewrite the following instruction as a natural user request.\n\n"
        f"INSTRUCTION:\n{row['transcription_request']}\n\n"
        f"LITERALS THAT MUST APPEAR VERBATIM:\n{literals}\n\n"
        "Remember: no tool names, no parameter names, all literals verbatim. JSON only."
    )
    if missing:
        msg += (
            "\n\nYour previous attempt failed these checks — literals listed must "
            "appear exactly as written; <remove ...> items must not appear at all: "
            + ", ".join(missing)
        )
    return msg


_JSON_RE = re.compile(r"\{[^{}]*\"natural_request\"\s*:.*\}", re.DOTALL)


def extract_natural_request(content: str) -> str | None:
    content = content.strip()
    # Strip a fenced block if present.
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", content).strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("natural_request"), str):
            return obj["natural_request"]
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(content)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("natural_request"), str):
                return obj["natural_request"]
        except json.JSONDecodeError:
            return None
    return None


class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def add(self, usage: dict | None) -> None:
        if not usage:
            return
        with self.lock:
            self.prompt += usage.get("prompt_tokens", 0)
            self.completion += usage.get("completion_tokens", 0)
            self.calls += 1


def call_api(api_key: str, model: str, messages: list[dict], usage: Usage,
             max_tokens: int, temperature: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # GLM-5.2 on Fireworks streams CoT into content by default; "none"
        # suppresses it (measured: 7 vs 60+ completion tokens for a tiny JSON).
        "reasoning_effort": "none",
    }).encode()
    backoff = 8.0
    for attempt in range(7):
        req = urllib.request.Request(API_URL, data=payload, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare blocks the default Python-urllib signature (403 code 1010).
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/126.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.load(resp)
            usage.add(body.get("usage"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503, 529):
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 90)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 90)
    raise RuntimeError("api retries exhausted")


def paraphrase_row(row: dict, api_key: str, model: str, usage: Usage,
                   retries: int, temperature: float) -> dict:
    missing: list[str] | None = None
    last_text = None
    for round_no in range(retries + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, missing)},
        ]
        content = call_api(api_key, model, messages, usage,
                           max_tokens=1400, temperature=temperature)
        text = extract_natural_request(content)
        if text is None:
            missing = None  # parse failure: plain retry
            continue
        last_text = text
        missing = [lit for lit in row["value_literals"] if lit not in text]
        # Leakage gate: internal tool/param names must not surface in the request.
        leaks = [w for w in ("mock_", "field_") if w in text]
        if leaks:
            missing = (missing or []) + [f"<remove internal name fragment: {w}>" for w in leaks]
        if not missing:
            return {"seed_id": row["seed_id"], "natural_request": text,
                    "tier": row.get("tier"), "rounds": round_no + 1, "ok": True}
    return {"seed_id": row["seed_id"], "natural_request": last_text,
            "tier": row.get("tier"), "rounds": retries + 1, "ok": False,
            "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="paraphrase_batch.jsonl")
    ap.add_argument("--out", required=True, help="writeback JSONL (appended, resumable)")
    ap.add_argument("--fail-out", default=None, help="rows that never passed the literal gate")
    ap.add_argument("--n", type=int, default=0, help="limit rows (0 = all)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        sys.exit("FIREWORKS_API_KEY not set")

    rows = [json.loads(l) for l in open(args.batch)]
    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                done_ids.add(json.loads(l)["seed_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in rows if r["seed_id"] not in done_ids]
    if args.n:
        todo = todo[: args.n]
    print(f"[driver] batch={len(rows)} done={len(done_ids)} todo={len(todo)} "
          f"model={args.model} conc={args.concurrency}", flush=True)

    usage = Usage()
    fail_path = Path(args.fail_out) if args.fail_out else out_path.with_suffix(".failures.jsonl")
    ok = fail = 0
    write_lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
            open(out_path, "a") as fh_ok, open(fail_path, "a") as fh_fail:
        futs = {pool.submit(paraphrase_row, r, api_key, args.model, usage,
                            args.retries, args.temperature): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 — record and continue
                res = {"seed_id": row["seed_id"], "ok": False, "error": str(e)}
            with write_lock:
                if res.get("ok"):
                    fh_ok.write(json.dumps({"seed_id": res["seed_id"],
                                            "natural_request": res["natural_request"]},
                                           ensure_ascii=False) + "\n")
                    fh_ok.flush()
                    ok += 1
                else:
                    fh_fail.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fh_fail.flush()
                    fail += 1
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[driver] {i}/{len(todo)} ok={ok} fail={fail} "
                      f"tok(p/c)={usage.prompt}/{usage.completion} "
                      f"calls={usage.calls} {el:.0f}s", flush=True)
    print(f"PARAPHRASE_DRIVER_DONE ok={ok} fail={fail} "
          f"prompt_tok={usage.prompt} completion_tok={usage.completion}", flush=True)


if __name__ == "__main__":
    main()
