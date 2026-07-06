#!/usr/bin/env python
"""SWE-RL format-forcing probe.

Question: how far does an explicit envelope-spec instruction lift the SWE-RL
``format_ok`` rate (score_record's format_valid) on the ESFT patch@k32 model,
and does forcing the envelope degrade the underlying edit quality?

Design (seed-fixed, paired):
  * Same 16 prompts selected exactly like rollout_gen (Random(seed).shuffle then
    first --limit), so this probe is same-condition with INC-0's first 16.
  * Three prompt arms, all sharing the per-(prompt,sample) sampling seed:
      P        : prompt_messages verbatim
      F_system : + envelope-spec instruction appended to the system message
      F_user   : + envelope-spec instruction appended to the last user message
  * n samples/prompt/arm, temp 1.0, top_p 0.95, max_tokens 4096.
  * Each completion scored strict (reward.score_record) and lenient
    (rollout_gen.lenient_reward). Method distribution tracked to separate
    "emitted the true SWE-RL envelope" (search_replace) from "diff-fallback
    rescued it" and from format_fail.

The forced instruction carries a MADE-UP dummy example (not from the data) to
avoid any leakage into the scored task.
"""
import argparse, asyncio, json, os, random, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as reward_mod
from rollout_gen import lenient_reward, load_jsonl

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp", file=sys.stderr); raise


# The exact envelope the strict parser wants, with a self-made dummy example
# (unrelated to any repo in the data -> no leakage). Spells out the three things
# the default prompt omits: the <solution> wrapper, the fenced code block, and
# the "### path" prefix on each SEARCH/REPLACE block.
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


PREFILL_TEXT = "<think>\n"  # seed the opening tag the SFT model won't emit


def _append_force(msgs, role):
    for m in (reversed(msgs) if role == "user" else msgs):
        if m["role"] == role:
            m["content"] = m["content"] + "\n" + FORCE_INSTRUCTION
            return msgs
    slot = len(msgs) if role == "user" else 0
    msgs.insert(slot, {"role": role, "content": FORCE_INSTRUCTION.strip()})
    return msgs


def build_messages(base_messages, arm):
    """Return (messages, prefill) for the given arm.

    prefill is a string prepended to the assistant turn (via continue_final_message)
    and stitched back onto the completion before scoring, or None.
    """
    msgs = [dict(m) for m in base_messages]
    if arm == "P":
        return msgs, None
    if arm == "F_system":
        return _append_force(msgs, "system"), None
    if arm in ("F_user", "F_user_prefill"):
        _append_force(msgs, "user")
        return msgs, (PREFILL_TEXT if arm == "F_user_prefill" else None)
    raise ValueError(arm)


async def gen_one(session, api, model, messages, temp, max_tokens, seed, sem,
                  prefill=None):
    msgs = list(messages)
    body = {"model": model, "temperature": temp,
            "max_tokens": max_tokens, "top_p": 0.95, "seed": seed}
    if prefill:
        msgs = msgs + [{"role": "assistant", "content": prefill}]
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    body["messages"] = msgs
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(f"{api}/chat/completions", json=body,
                                        timeout=aiohttp.ClientTimeout(total=1800)) as r:
                    j = await r.json()
                c = j["choices"][0]
                gen = c["message"]["content"] or ""
                # stitch prefill back so the scorer sees the full envelope
                text = (prefill + gen) if prefill else gen
                return {"text": text,
                        "finish": c.get("finish_reason"),
                        "tokens": j.get("usage", {}).get("completion_tokens")}
            except Exception as e:
                if attempt == 2:
                    return {"text": "", "finish": f"error:{e}", "tokens": 0}
                await asyncio.sleep(5 * (attempt + 1))


def diag_flags(text):
    """Tag-level diagnostics: where exactly does the strict envelope break?"""
    n_to = text.count("<think>"); n_tc = text.count("</think>")
    n_so = text.count("<solution>"); n_sc = text.count("</solution>")
    envelope_ok = (n_to == 1 and n_tc == 1 and n_so == 1 and n_sc == 1)
    return {
        "n_think_open": n_to, "n_think_close": n_tc,
        "n_sol_open": n_so, "n_sol_close": n_sc,
        "has_sr_marker": "<<<<<<< SEARCH" in text,
        "envelope_tags_ok": envelope_ok,  # all four tags exactly once
    }


def score(merged, text):
    res = reward_mod.score_record(merged, text)
    return {
        "strict": res["reward"],
        "format_valid": bool(res.get("format_valid")),
        "method": res.get("method", "format_fail"),
        "lenient": lenient_reward(merged, text),
        **diag_flags(text),
    }


async def main_async(args):
    records = list(load_jsonl(args.data))
    sidecar = {r["instance_id"]: r for r in load_jsonl(args.sidecar)}
    rng = random.Random(args.seed)
    rng.shuffle(records)
    sel = [r for r in records[:args.limit] if r["instance_id"] in sidecar]
    arms = args.arms.split(",")
    print(f"[plan] {len(sel)} prompts x {args.n} samples x {len(arms)} arms {arms} "
          f"= {len(sel)*args.n*len(arms)} gens, max_tok={args.max_tokens} "
          f"conc={args.concurrency}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    # build the full job list; paired sampling seed across arms per (pi, si)
    jobs = []
    for pi, rec in enumerate(sel):
        for si in range(args.n):
            samp_seed = args.seed + pi * 100 + si
            for arm in arms:
                msgs, prefill = build_messages(rec["prompt_messages"], arm)
                jobs.append((pi, rec, si, arm, samp_seed, msgs, prefill))

    t0 = time.time()
    done = 0
    async with aiohttp.ClientSession() as session:
        async def run(job):
            nonlocal done
            pi, rec, si, arm, samp_seed, msgs, prefill = job
            o = await gen_one(session, args.api, args.model, msgs,
                              args.temp, args.max_tokens, samp_seed, sem, prefill)
            files = sidecar[rec["instance_id"]]
            merged = {**rec, "repo_files": files["repo_files"],
                      "oracle_new_files": files["oracle_new_files"]}
            sc = score(merged, o["text"])
            row = {"pi": pi, "instance_id": rec["instance_id"], "si": si,
                   "arm": arm, "seed": samp_seed, "finish": o["finish"],
                   "tokens": o["tokens"], "text_len": len(o["text"]),
                   "text": o["text"], **sc}
            done += 1
            if done <= 6 or done % 16 == 0:
                began = row["n_sol_open"] >= 1 or row["has_sr_marker"]
                print(f"[{done}/{len(jobs)}] arm={arm} finish={o['finish']} "
                      f"tok={o['tokens']} fmt_ok={row['format_valid']} began={began} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
            return row

        results = await asyncio.gather(*(run(j) for j in jobs))

    # persist raw
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # aggregate
    by_arm = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)

    el = time.time() - t0
    print(f"\n[done] {len(results)} gens in {el/60:.1f} min\n", flush=True)
    print(f"{'arm':<9} {'n':>4} {'fmt_ok%':>8} {'env%':>6} {'fb%':>5} {'fail%':>6} "
          f"{'strict_mean':>12} {'strict_bo'+str(args.n):>11} "
          f"{'lenient_mean':>13} {'lenient_bo'+str(args.n):>12}")
    summary = {}
    for arm in arms:
        rs = by_arm[arm]
        n = len(rs)
        fmt_ok = sum(r["format_valid"] for r in rs) / n
        env = sum(r["method"] == "search_replace" for r in rs) / n
        fb = sum(r["method"] == "diff_fallback" for r in rs) / n
        fail = sum(r["method"] == "format_fail" for r in rs) / n
        strict_mean = sum(r["strict"] for r in rs) / n
        lenient_mean = sum(r["lenient"] for r in rs) / n
        # best-of-n per prompt
        by_pi_s = defaultdict(list); by_pi_l = defaultdict(list)
        for r in rs:
            by_pi_s[r["pi"]].append(r["strict"])
            by_pi_l[r["pi"]].append(r["lenient"])
        strict_bo = sum(max(v) for v in by_pi_s.values()) / len(by_pi_s)
        lenient_bo = sum(max(v) for v in by_pi_l.values()) / len(by_pi_l)
        trunc = sum(r["finish"] == "length" for r in rs) / n
        summary[arm] = dict(n=n, fmt_ok=fmt_ok, env=env, fb=fb, fail=fail,
                            trunc=trunc,
                            strict_mean=strict_mean, strict_bo=strict_bo,
                            lenient_mean=lenient_mean, lenient_bo=lenient_bo)
        print(f"{arm:<9} {n:>4} {fmt_ok*100:>7.1f}% {env*100:>5.1f}% {fb*100:>4.1f}% "
              f"{fail*100:>5.1f}% {strict_mean:>12.4f} {strict_bo:>11.4f} "
              f"{lenient_mean:>13.4f} {lenient_bo:>12.4f}  trunc={trunc*100:.0f}%")

    # --- mechanism decomposition: WHY does strict fail? ---
    print(f"\n{'arm':<9} {'trunc%':>7} {'open<think>%':>12} {'tags_ok%':>9} "
          f"{'SRmark%':>8} {'tags_ok&stop%':>14} {'tags_ok&stop&valid%':>20}")
    for arm in arms:
        rs = by_arm[arm]; n = len(rs)
        trunc = sum(r["finish"] == "length" for r in rs) / n
        open_think = sum(r["n_think_open"] >= 1 for r in rs) / n
        tags_ok = sum(r["envelope_tags_ok"] for r in rs) / n
        srmark = sum(r["has_sr_marker"] for r in rs) / n
        stop = [r for r in rs if r["finish"] == "stop"]
        tags_ok_stop = (sum(r["envelope_tags_ok"] for r in stop) / n)
        # of the non-truncated + tags_ok, how many actually score as format_valid
        ready = [r for r in stop if r["envelope_tags_ok"]]
        ready_valid = (sum(r["format_valid"] for r in ready) / n)
        summary[arm].update(open_think=open_think, tags_ok=tags_ok,
                            srmark=srmark, n_stop=len(stop), n_ready=len(ready))
        print(f"{arm:<9} {trunc*100:>6.1f}% {open_think*100:>11.1f}% {tags_ok*100:>8.1f}% "
              f"{srmark*100:>7.1f}% {tags_ok_stop*100:>13.1f}% {ready_valid*100:>19.1f}%")

    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[raw]     {args.out}")
    print(f"[summary] {args.out.replace('.jsonl', '_summary.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--data", default="data/grpo_prompts.jsonl")
    ap.add_argument("--sidecar", default="data/grpo_prompts_files.jsonl")
    ap.add_argument("--out", default="probe_out/format_probe.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5934875)
    ap.add_argument("--arms", default="P,F_system,F_user",
                    help="comma list: P,F_system,F_user,F_user_prefill")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
