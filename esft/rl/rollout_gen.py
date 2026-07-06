#!/usr/bin/env python
"""INC-0 / GRPO rollout generation against an OpenAI-compatible vLLM server.

Reads grpo_prompts.jsonl (+ sidecar for rewards), samples N completions per
prompt, scores with reward.score_record, writes rollouts.jsonl incrementally
(resume-safe: skips instance_ids already present in the output).

Usage:
  python rollout_gen.py --api http://127.0.0.1:18000/v1 --model <served-name> \
      --data data/grpo_prompts.jsonl --sidecar data/grpo_prompts_files.jsonl \
      --out rollouts/inc0.jsonl --n 8 --limit 384 --temp 1.0 --max-tokens 4096 \
      --concurrency 8
"""
import argparse, asyncio, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward as reward_mod

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp", file=sys.stderr); raise


import re as _re
_FENCE = _re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", _re.DOTALL)
_HUNK_AT = _re.compile(r"^@@ .* @@", _re.M)


def _recover_bare_hunks(text, code_ctx):
    """Recover edits from the SFT model's native format: unified-diff hunks
    (lines starting with ' ', '+', '-') inside a plain fence, WITHOUT a
    ``--- a/ +++ b/`` git header. We synthesize the header from the single
    target file (the common case: one file per instance) and hand it to the
    existing diff parser. Returns {} if not confidently recoverable.
    """
    if len(code_ctx) != 1:
        return {}  # multi-file bare hunks are ambiguous without a path — skip
    path = next(iter(code_ctx))
    for block in _FENCE.findall(text):
        lines = block.splitlines()
        pm = sum(1 for ln in lines if ln[:1] in "+-")
        if pm < 1:
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
    """Capability-only reward: parse SEARCH/REPLACE blocks from ANYWHERE in the
    completion (no <think>/<solution> envelope required), apply, and score with the
    same reward math. Measures whether the *edit* is right, decoupled from whether
    the model followed the SWE-RL output format (which our terminus-2 SFT never
    taught). Returns -1.0 only if no applicable edit is recoverable at all.
    """
    try:
        code_ctx = reward_mod._record_code_context(record)
        oracle_new = reward_mod._record_oracle_new(record, code_ctx)
        # 1) SEARCH/REPLACE anywhere (strip think spans first to avoid reasoning noise)
        text = reward_mod._THINK_BLOCK_RE.sub("", completion)
        sr = reward_mod.parse_search_replace(text)
        if not sr:  # 2) raw unified diff (git header or fenced ```diff)
            diff = reward_mod.extract_patch(completion)
            if diff:
                sr = reward_mod.diff_to_search_replace(diff)
        if not sr:  # 3) our SFT model's native format: bare +/- hunks in a fence,
            #        no git header. Synthesize a header using the (usually single)
            #        target file path so diff_to_search_replace can parse it.
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


async def gen_one(session, api, model, messages, temp, max_tokens, sem):
    body = {"model": model, "messages": messages, "temperature": temp,
            "max_tokens": max_tokens, "top_p": 0.95}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(f"{api}/chat/completions", json=body,
                                        timeout=aiohttp.ClientTimeout(total=1800)) as r:
                    j = await r.json()
                c = j["choices"][0]
                return {"text": c["message"]["content"],
                        "finish": c.get("finish_reason"),
                        "tokens": j.get("usage", {}).get("completion_tokens")}
            except Exception as e:
                if attempt == 2:
                    return {"text": "", "finish": f"error:{e}", "tokens": 0}
                await asyncio.sleep(5 * (attempt + 1))


async def main_async(args):
    records = list(load_jsonl(args.data))
    sidecar = {r["instance_id"]: r for r in load_jsonl(args.sidecar)}
    done = set()
    if os.path.exists(args.out):
        done = {r["instance_id"] for r in load_jsonl(args.out)}
        print(f"[resume] {len(done)} instances already in {args.out}")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    todo = [r for r in records if r["instance_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[plan] {len(todo)} prompts x {args.n} rollouts, conc={args.concurrency}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    stats = {"fmt_ok": 0, "total": 0, "reward_sum": 0.0, "best_sum": 0.0, "len_sum": 0.0, "len_best_sum": 0.0}
    async with aiohttp.ClientSession() as session:
        with open(args.out, "a") as fout:
            for i, rec in enumerate(todo):
                iid = rec["instance_id"]
                files = sidecar.get(iid)
                if files is None:
                    print(f"[skip] {iid}: no sidecar row"); continue
                tasks = [gen_one(session, args.api, args.model,
                                 rec["prompt_messages"], args.temp,
                                 args.max_tokens, sem) for _ in range(args.n)]
                outs = await asyncio.gather(*tasks)
                scored = []
                for o in outs:
                    merged = {**rec, "repo_files": files["repo_files"],
                              "oracle_new_files": files["oracle_new_files"]}
                    res = reward_mod.score_record(merged, o["text"])
                    len_r = lenient_reward(merged, o["text"])
                    scored.append({**o,
                                   "reward": res["reward"], "format_valid": res.get("format_valid", False),
                                   "reward_lenient": len_r})
                rewards = [s["reward"] for s in scored]
                rewards_len = [s["reward_lenient"] for s in scored]
                stats["total"] += len(rewards)
                stats["fmt_ok"] += sum(1 for s in scored if s.get("format_valid"))
                stats["reward_sum"] += sum(rewards)
                stats["best_sum"] += max(rewards)
                stats["len_sum"] += sum(rewards_len)
                stats["len_best_sum"] += max(rewards_len)
                fout.write(json.dumps({
                    "instance_id": iid, "n": args.n, "temp": args.temp,
                    "rewards": rewards, "rewards_lenient": rewards_len,
                    "best": max(rewards), "best_lenient": max(rewards_len),
                    "completions": scored}, ensure_ascii=False) + "\n")
                fout.flush()
                if (i + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"[{i+1}/{len(todo)}] strict_mean={stats['reward_sum']/stats['total']:.3f} "
                          f"lenient_mean={stats['len_sum']/stats['total']:.3f} "
                          f"lenient_best={stats['len_best_sum']/(i+1):.3f} "
                          f"fmt_ok={stats['fmt_ok']/stats['total']:.1%} "
                          f"{el/(i+1):.1f}s/prompt", flush=True)
    el = time.time() - t0
    T = max(1, stats["total"]); P = max(1, len(todo))
    print(f"DONE prompts={len(todo)} "
          f"strict_mean={stats['reward_sum']/T:.4f} strict_best_of_{args.n}={stats['best_sum']/P:.4f} "
          f"lenient_mean={stats['len_sum']/T:.4f} lenient_best_of_{args.n}={stats['len_best_sum']/P:.4f} "
          f"format_ok_rate={stats['fmt_ok']/T:.2%} wall={el/60:.1f}min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5934875)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
