#!/usr/bin/env python
"""INC-0 rollout generation on the aux-host, F_user_prefill configuration.

Gate-selected config (probe @ max_tokens 10000, n=64/arm):
  F_user_prefill won every axis -- fmt_ok 10.9% (only arm emitting the true
  SWE-RL envelope), strict_bo4 -0.388, lenient_mean -0.064, lenient_bo4 +0.574,
  paired Δlenient vs plain +0.394 (95% CI [+0.148,+0.647]), trunc 6%.

Per prompt: append the explicit envelope-spec instruction to the last user
message AND prefill the assistant turn with "<think>\\n" (the one tag the
terminus-2 SFT reliably omits) via continue_final_message. The completion is
stitched back to prefill+gen so the scorer and any downstream SFT target see the
whole envelope. We record the prefill string so the SFT target can be rebuilt as
full_assistant = prefill + completion.

Reward: strict (reward.score_record, SWE-RL envelope) AND lenient (capability-only,
format-agnostic) are both recorded per completion. INC-0's working reward is
lenient; strict is kept for the ~11% that pass the envelope.

Resume-safe: skips instance_ids already present in --out. Selection matches
rollout_gen.py exactly (Random(seed).shuffle then first --limit) so INC-0 is
same-condition with the probe's first 16.
"""
import argparse, asyncio, json, os, random, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as reward_mod

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp (uv pip install --python <venv> aiohttp)", file=sys.stderr); raise

import re as _re
_FENCE = _re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", _re.DOTALL)
_HUNK_AT = _re.compile(r"^@@ .* @@", _re.M)

FORCE_INSTRUCTION = """\

--- OUTPUT FORMAT (STRICT — follow exactly) ---
Your entire response MUST be exactly one <think> block followed by exactly one
<solution> block, and nothing outside them:

<think>
your reasoning about where the bug is and how to fix it
</think>
<solution>
```python
### path/to/file.py
<<<<<<< SEARCH
verbatim original lines to find (indentation-exact)
=======
replacement lines
>>>>>>> REPLACE
```
</solution>

Rules:
- Put the *entire* final answer inside <solution>...</solution>.
- Each edit is a fenced code block. The first line inside the fence is `### `
  followed by the file path. Then `<<<<<<< SEARCH`, the exact original lines,
  `=======`, the replacement lines, `>>>>>>> REPLACE`.
- The SEARCH text must be copied verbatim from the file (exact indentation) so
  it can be found. Emit one fenced block per edit; multiple edits are allowed.

Minimal example (illustration only — do NOT use this content):
<think>
The greeting says "Hi" but the issue asks for "Hello"; fix the return line.
</think>
<solution>
```python
### src/greeter.py
<<<<<<< SEARCH
def greet(name):
    return "Hi " + name
=======
def greet(name):
    return "Hello " + name
>>>>>>> REPLACE
```
</solution>
"""

PREFILL_TEXT = "<think>\n"


def build_prompt(prompt_messages):
    """F_user_prefill: force-instruction on the last user msg + assistant prefill."""
    msgs = [dict(m) for m in prompt_messages]
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] = m["content"] + "\n" + FORCE_INSTRUCTION
            break
    else:
        msgs.append({"role": "user", "content": FORCE_INSTRUCTION.strip()})
    return msgs, PREFILL_TEXT


def _recover_bare_hunks(text, code_ctx):
    if len(code_ctx) != 1:
        return {}
    path = next(iter(code_ctx))
    for block in _FENCE.findall(text):
        lines = block.splitlines()
        if sum(1 for ln in lines if ln[:1] in "+-") < 1:
            continue
        body = block if _HUNK_AT.search(block) else "@@ -1 +1 @@\n" + block
        diff = f"--- a/{path}\n+++ b/{path}\n{body}"
        try:
            sr = reward_mod.diff_to_search_replace(diff)
            if sr:
                return sr
        except Exception:
            continue
    return {}


def lenient_reward(record, completion):
    """Capability-only reward: parse SEARCH/REPLACE from anywhere (no envelope
    required), apply, score with the same math. -1.0 only if nothing recoverable."""
    try:
        code_ctx = reward_mod._record_code_context(record)
        oracle_new = reward_mod._record_oracle_new(record, code_ctx)
        text = reward_mod._THINK_BLOCK_RE.sub("", completion)
        sr = reward_mod.parse_search_replace(text)
        if not sr:
            diff = reward_mod.extract_patch(completion)
            if diff:
                sr = reward_mod.diff_to_search_replace(diff)
        if not sr:
            sr = _recover_bare_hunks(text, code_ctx)
        if not sr:
            return -1.0
        pred_new = reward_mod.apply_code_change(code_ctx, sr)
        r, _ = reward_mod.calculate_reward(code_ctx, oracle_new, pred_new, normalize=True)
        return r
    except Exception:
        return -1.0


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


async def gen_one(session, api, model, messages, prefill, temp, max_tokens, seed, sem):
    msgs = list(messages) + [{"role": "assistant", "content": prefill}]
    body = {"model": model, "messages": msgs, "temperature": temp,
            "max_tokens": max_tokens, "top_p": 0.95, "seed": seed,
            "continue_final_message": True, "add_generation_prompt": False}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(f"{api}/chat/completions", json=body,
                                        timeout=aiohttp.ClientTimeout(total=1800)) as r:
                    j = await r.json()
                if "choices" not in j:
                    raise RuntimeError(str(j)[:200])
                c = j["choices"][0]
                gen = c["message"]["content"] or ""
                return {"text": prefill + gen,  # stitch prefill back for scoring
                        "gen_only": gen,
                        "finish": c.get("finish_reason"),
                        "tokens": j.get("usage", {}).get("completion_tokens")}
            except Exception as e:
                if attempt == 2:
                    return {"text": prefill, "gen_only": "", "finish": f"error:{e}", "tokens": 0}
                await asyncio.sleep(5 * (attempt + 1))


def cap_max_tokens(prompt_tokens, want, ctx, margin=600):
    """Keep prompt + completion within the serve context window (avoids 400s on
    the few very long prompts). Chat template + prefill add a little, hence margin."""
    room = ctx - int(prompt_tokens or 0) - margin
    return max(512, min(want, room))


async def main_async(args):
    records = list(load_jsonl(args.data))
    sidecar = {r["instance_id"]: r for r in load_jsonl(args.sidecar)}
    done = set()
    if os.path.exists(args.out):
        done = {r["instance_id"] for r in load_jsonl(args.out)}
        print(f"[resume] {len(done)} instances already in {args.out}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    todo = [r for r in records[:args.limit] if r["instance_id"] not in done]
    print(f"[plan] {len(todo)} prompts x {args.n} rollouts (max_tok={args.max_tokens} "
          f"cap@ctx{args.ctx}) conc={args.concurrency} model={args.model}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    S = {"fmt_ok": 0, "err": 0, "total": 0, "s_sum": 0.0, "s_best": 0.0,
         "l_sum": 0.0, "l_best": 0.0}
    async with aiohttp.ClientSession() as session:
        with open(args.out, "a") as fout:
            for i, rec in enumerate(todo):
                iid = rec["instance_id"]
                files = sidecar.get(iid)
                if files is None:
                    print(f"[skip] {iid}: no sidecar", flush=True); continue
                msgs, prefill = build_prompt(rec["prompt_messages"])
                mt = cap_max_tokens(rec.get("prompt_tokens"), args.max_tokens, args.ctx)
                tasks = [gen_one(session, args.api, args.model, msgs, prefill,
                                 args.temp, mt, args.seed + i * 100 + s, sem)
                         for s in range(args.n)]
                outs = await asyncio.gather(*tasks)
                merged = {**rec, "repo_files": files["repo_files"],
                          "oracle_new_files": files["oracle_new_files"]}
                comps = []
                for o in outs:
                    res = reward_mod.score_record(merged, o["text"])
                    lr = lenient_reward(merged, o["text"])
                    comps.append({"text": o["text"], "prefill": prefill,
                                  "finish": o["finish"], "tokens": o["tokens"],
                                  "strict": res["reward"],
                                  "format_valid": bool(res.get("format_valid")),
                                  "method": res.get("method"),
                                  "lenient": lr})
                s_r = [c["strict"] for c in comps]
                l_r = [c["lenient"] for c in comps]
                S["total"] += len(comps)
                S["fmt_ok"] += sum(c["format_valid"] for c in comps)
                S["err"] += sum(str(c["finish"]).startswith("error:") for c in comps)
                S["s_sum"] += sum(s_r); S["s_best"] += max(s_r)
                S["l_sum"] += sum(l_r); S["l_best"] += max(l_r)
                fout.write(json.dumps({
                    "instance_id": iid, "n": args.n, "prefill": prefill,
                    "max_tokens": mt, "strict": s_r, "lenient": l_r,
                    "strict_best": max(s_r), "lenient_best": max(l_r),
                    "fmt_ok": sum(c["format_valid"] for c in comps),
                    "completions": comps}, ensure_ascii=False) + "\n")
                fout.flush()
                if (i + 1) % 10 == 0 or i == 0:
                    el = time.time() - t0; T = max(1, S["total"]); P = i + 1
                    print(f"[{i+1}/{len(todo)}] lenient_mean={S['l_sum']/T:.3f} "
                          f"lenient_best8={S['l_best']/P:.3f} strict_mean={S['s_sum']/T:.3f} "
                          f"fmt_ok={S['fmt_ok']/T:.1%} err={S['err']/T:.2%} "
                          f"{el/P:.1f}s/prompt eta={el/P*(len(todo)-P)/3600:.1f}h", flush=True)
    el = time.time() - t0; T = max(1, S["total"]); P = max(1, len(todo))
    print(f"DONE prompts={len(todo)} gens={S['total']} "
          f"lenient_mean={S['l_sum']/T:.4f} lenient_best{args.n}={S['l_best']/P:.4f} "
          f"strict_mean={S['s_sum']/T:.4f} strict_best{args.n}={S['s_best']/P:.4f} "
          f"fmt_ok={S['fmt_ok']/T:.2%} err_rate={S['err']/T:.2%} wall={el/60:.1f}min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--data", default=os.path.expanduser("~/esft/data/rl/grpo_prompts.jsonl"))
    ap.add_argument("--sidecar", default=os.path.expanduser("~/esft/data/rl/grpo_prompts_files.jsonl"))
    ap.add_argument("--out", default=os.path.expanduser("~/esft/rollouts/inc0_prefill.jsonl"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=384)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5934875)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
