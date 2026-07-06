#!/usr/bin/env python
"""
Build GRPO (SWE-RL style) RL prompt set from SWE-smith trajectories.

Data integrity note (load-bearing):
  The local SWE-bench_SWE-smith-trajectories parquet copy has a CORRUPT `patch`
  column: it is shuffled relative to (instance_id, messages) across every config
  (tool/xml/ticks/train), verified at ~2% repo-consistency (random chance).
  Therefore the oracle patch is NOT taken from the trajectories; it is joined by
  instance_id from the authoritative SWE-bench/SWE-smith base dataset, whose
  `patch` column is perfectly aligned (300/300 repo-consistency check).

  - prompt_messages : from trajectories `train` config (system + first user msg).
                      These ARE aligned with instance_id (channels traj <-> channels id).
  - oracle_patch    : gold patch joined by instance_id from SWE-smith base.
  - repo/base_commit: reconstructed from the SWE-smith instance_id encoding
                      "{owner}__{repo}.{base_commit8}.{strategy}__{hash}".
  - resolved=True   : quality gate kept from trajectories.

Decontam (hard gate): drop any instance matching SWE-bench_Verified by
  instance_id OR (repo, base_commit-prefix), or matching a Terminal-Bench-2 task.
"""
import glob, json, re, sys, time
from collections import Counter, defaultdict
import pyarrow.parquet as pq

TRAJ_DIR = "~/esft/data/hf/SWE-bench_SWE-smith-trajectories/data"
BASE_DIR = "~/esft/data/hf/SWE-bench_SWE-smith_base/data"
VERIFIED_CACHE = "~/.cache/huggingface/datasets/princeton-nlp___swe-bench_verified"
DECONTAM_TERMINAL = "~/esft/decontam_terminal_bench.json"
TOKENIZER_PATH = "~/esft-work/models/Qwen3.6-35B-A3B"
OUT_JSONL = "~/esft/data/rl/grpo_prompts.jsonl"
OUT_STATS = "~/esft/data/rl/grpo_prompts_stats.json"
SOURCE = "SWE-bench/SWE-smith-trajectories"
MAX_PATCH_BYTES = 100 * 1024

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def parse_instance(iid):
    """owner__repo.base_commit8.strategy__hash -> (repo 'owner/repo', base_commit8, strategy)."""
    head = iid.split(".")[0]              # owner__repo
    parts = iid.split(".")
    commit = parts[1] if len(parts) > 1 else ""
    strat = parts[2] if len(parts) > 2 else ""
    strat = strat.split("__")[0].split("_")[0]  # e.g. lm, combine, func, pr, or numeric
    if "__" in head:
        owner, repo = head.split("__", 1)
        repo_full = f"{owner}/{repo}"
    else:
        owner, repo, repo_full = "", head, head
    # strategy family label for stats
    m = re.search(r"\.[0-9a-f]{6,}\.([a-zA-Z_]+)", iid)
    strat_label = m.group(1).rstrip("_") if m else (strat if strat else "other")
    return repo_full, commit, strat_label

def is_binary_only(patch):
    if "GIT binary patch" in patch:
        return "@@" not in patch
    if "Binary files" in patch and "@@" not in patch:
        return True
    return False

def load_verified():
    import pyarrow as pa, pyarrow.ipc as ipc
    files = glob.glob(VERIFIED_CACHE + "/**/*.arrow", recursive=True) + \
            glob.glob(VERIFIED_CACHE + "/**/*.parquet", recursive=True)
    ids = set()
    repo_commits = defaultdict(set)  # repo -> {full base_commit}
    for f in files:
        if f.endswith(".arrow"):
            with pa.memory_map(f) as src:
                try: t = ipc.open_stream(src).read_all()
                except Exception: t = ipc.open_file(src).read_all()
        else:
            t = pq.read_table(f)
        for r in t.select(["instance_id", "repo", "base_commit"]).to_pylist():
            ids.add(r["instance_id"])
            repo_commits[r["repo"]].add(r["base_commit"])
    return ids, repo_commits

def main():
    t0 = time.time()
    log("loading SWE-bench_Verified decontam keys...")
    ver_ids, ver_repo_commits = load_verified()
    log(f"  verified: {len(ver_ids)} instances, {len(ver_repo_commits)} repos")

    term = json.load(open(DECONTAM_TERMINAL))
    term_tasks = set(term.get("tasks", []))
    log(f"  terminal-bench tasks: {len(term_tasks)}")

    # Pass A: resolved=True instance ids from train config
    log("pass A: scanning trajectories for resolved=True instances...")
    tfiles = sorted(glob.glob(TRAJ_DIR + "/train-*.parquet"))
    resolved_ids = set()
    for tf in tfiles:
        for r in pq.read_table(tf, columns=["instance_id", "resolved"]).to_pylist():
            if r["resolved"]:
                resolved_ids.add(r["instance_id"])
    log(f"  resolved=True unique instances: {len(resolved_ids)}")

    # load base gold patches only for needed ids (memory-safe)
    log("loading SWE-smith base gold patches (filtered)...")
    bfiles = sorted(glob.glob(BASE_DIR + "/*.parquet"))
    gold = {}
    for bf in bfiles:
        for r in pq.read_table(bf, columns=["instance_id", "patch"]).to_pylist():
            iid = r["instance_id"]
            if iid in resolved_ids and iid not in gold:
                gold[iid] = r["patch"] or ""
    log(f"  gold patches available for {len(gold)}/{len(resolved_ids)} resolved instances")

    log("loading Qwen3.6 tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    # Pass B: build records
    log("pass B: building records...")
    drops = Counter()
    by_strategy_kept = Counter()
    by_strategy_drop = Counter()
    seen = set()
    tok_lens = []
    samples = []
    n_written = 0
    fout = open(OUT_JSONL, "w")
    for tf in tfiles:
        tbl = pq.read_table(tf, columns=["instance_id", "resolved", "messages"]).to_pylist()
        for r in tbl:
            iid = r["instance_id"]
            if not r["resolved"]:
                continue
            if iid in seen:
                drops["dup_instance"] += 1
                continue
            repo, commit8, strat = parse_instance(iid)
            # decontam: instance_id
            if iid in ver_ids:
                drops["decontam_instance_id"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            # decontam: terminal-bench
            if iid in term_tasks or repo.split("/")[-1] in term_tasks:
                drops["decontam_terminal"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            # decontam: (repo, base_commit-prefix)
            hit = False
            if repo in ver_repo_commits and commit8:
                for bc in ver_repo_commits[repo]:
                    if bc.startswith(commit8) or commit8.startswith(bc[:len(commit8)]):
                        hit = True; break
            if hit:
                drops["decontam_repo_commit"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            # oracle patch join
            patch = gold.get(iid)
            if patch is None:
                drops["no_oracle_in_base"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            if not patch.strip():
                drops["empty_patch"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
                drops["patch_too_large"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            if is_binary_only(patch):
                drops["binary_only"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            # prompt: system + first user
            msgs = r["messages"]
            pm = []
            for m in msgs:
                if m["role"] == "assistant":
                    break
                pm.append({"role": m["role"], "content": m["content"]})
            if not (len(pm) >= 2 and pm[0]["role"] == "system" and pm[1]["role"] == "user"):
                drops["bad_prompt_shape"] += 1; by_strategy_drop[strat] += 1; seen.add(iid); continue
            # token length via chat template
            try:
                ids_ = tok.apply_chat_template(pm, add_generation_prompt=True, tokenize=True)
                tlen = len(ids_)
            except Exception:
                tlen = len(tok(pm[0]["content"] + pm[1]["content"]).input_ids)
            tok_lens.append(tlen)
            rec = {
                "prompt_messages": pm,
                "oracle_patch": patch,
                "repo": repo,
                "base_commit": commit8,
                "instance_id": iid,
                "source": SOURCE,
                "strategy": strat,
                "prompt_tokens": tlen,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            by_strategy_kept[strat] += 1
            seen.add(iid)
            if len(samples) < 3:
                samples.append({
                    "instance_id": iid, "repo": repo, "base_commit": commit8,
                    "prompt_tokens": tlen,
                    "prompt_head_500": (pm[0]["content"] + "\n\n[USER]\n" + pm[1]["content"])[:500],
                    "patch_head_300": patch[:300],
                })
    fout.close()

    tok_lens.sort()
    def pct(p):
        if not tok_lens: return 0
        return tok_lens[min(len(tok_lens) - 1, int(p * len(tok_lens)))]
    stats = {
        "source": SOURCE,
        "oracle_patch_source": "SWE-bench/SWE-smith (base, joined by instance_id; trajectory patch column was corrupt)",
        "total_written": n_written,
        "resolved_true_instances": len(resolved_ids),
        "gold_join_coverage": f"{len(gold)}/{len(resolved_ids)}",
        "drops": dict(drops),
        "drops_total": sum(drops.values()),
        "kept_by_strategy": dict(by_strategy_kept),
        "dropped_by_strategy": dict(by_strategy_drop),
        "prompt_token_len": {
            "n": len(tok_lens),
            "p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95),
            "p99": pct(0.99), "max": tok_lens[-1] if tok_lens else 0,
            "min": tok_lens[0] if tok_lens else 0,
            "mean": round(sum(tok_lens) / len(tok_lens), 1) if tok_lens else 0,
            "gt_32768": sum(1 for x in tok_lens if x > 32768),
        },
        "tokenizer": TOKENIZER_PATH,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump({"stats": stats, "samples": samples}, open(OUT_STATS, "w"), ensure_ascii=False, indent=2)

    log("=== DONE ===")
    print(json.dumps(stats, indent=2))
    print("\n=== SAMPLES ===")
    for s in samples:
        print(f"\n--- {s['instance_id']} ({s['repo']} @ {s['base_commit']}, {s['prompt_tokens']} tok) ---")
        print("PROMPT[:500]:", s["prompt_head_500"])
        print("PATCH[:300]:", s["patch_head_300"])

if __name__ == "__main__":
    main()
