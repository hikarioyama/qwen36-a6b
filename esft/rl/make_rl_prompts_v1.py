#!/usr/bin/env python
"""
Rebuild GRPO RL prompts in SWE-RL (arXiv:2502.18449) agentless-repair format.

Change from v0 (per team-lead gate 2):
  v0 used the SWE-agent [system, first-user] messages -> emits tool calls,
  no code context. Our GRPO is single-shot patch generation with no execution
  env, so we must give the model (a) a SEARCH/REPLACE instruction system prompt
  and (b) the actual code of the files the oracle patch touches, at the BUGGY
  state the agent starts from. That buggy state is the branch named exactly
  after the instance_id in the swesmith mirror `swesmith/{owner}__{repo}.{commit8}`.

Prompt (verbatim SWE-RL templates from facebookresearch/swe-rl core/prompts.py):
  system = THINKING_SYSTEM
  user   = AGENTLESS_REPAIR.format(problem_statement=<base issue>,
             content=<concat of "### {path}\n{file_content}" for oracle-touched files>)

Selection (unchanged, approved): resolved=True, gold-patch join from SWE-smith
  base, decontam vs SWE-bench_Verified + Terminal-Bench, dedup 1 instance = 1.

Code context = files touched by the oracle patch, fetched at buggy state via
  raw.githubusercontent.com (no clone; ~0 disk). Oracle patch retained for the
  reward function (SEARCH/REPLACE similarity, computed by another agent).
"""
import glob, json, os, re, sys, time, argparse, subprocess, tempfile, urllib.request, urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq

TRAJ_DIR = "~/esft/data/hf/SWE-bench_SWE-smith-trajectories/data"
BASE_DIR = "~/esft/data/hf/SWE-bench_SWE-smith_base/data"
VERIFIED_CACHE = "~/.cache/huggingface/datasets/princeton-nlp___swe-bench_verified"
DECONTAM_TERMINAL = "~/esft/decontam_terminal_bench.json"
TOKENIZER_PATH = "~/esft-work/models/Qwen3.6-35B-A3B"
SOURCE = "SWE-bench/SWE-smith-trajectories"
MAX_PATCH_BYTES = 100 * 1024
MAX_PROMPT_TOKENS = 24000

# ---- verbatim SWE-RL templates (facebookresearch/swe-rl, core/prompts.py) ----
THINKING_SYSTEM = (
    "A user will ask you to solve a task. You should first draft your thinking "
    "process (inner monologue). Then, generate the solution.\n\nYour response "
    "format must follow the template below:\n<think>\nYour thoughts or/and draft, "
    "like working through an exercise on scratch paper. Be as casual and as long "
    "as you want until you are confident to generate a correct solution.\n</think>"
    "\n<solution>\nFinal solution presented to the user.\n</solution>"
)
AGENTLESS_REPAIR = (
    "We are currently solving the following issue within our repository. Here is "
    "the issue text:\n--- BEGIN ISSUE ---\n{problem_statement}\n--- END ISSUE ---"
    "\n\nBelow are some code segments, each from a relevant file. One or more of "
    "these files may contain bugs.\n\n--- BEGIN FILE ---\n```\n{content}\n```\n"
    "--- END FILE ---\n\nPlease first localize the bug based on the issue "
    "statement, and then generate *SEARCH/REPLACE* edits to fix the issue.\n\n"
    "Every *SEARCH/REPLACE* edit must use this format:\n1. The file path\n2. The "
    "start of search block: <<<<<<< SEARCH\n3. A contiguous chunk of lines to "
    "search for in the existing source code\n4. The dividing line: =======\n5. "
    "The lines to replace into the source code\n6. The end of the replace block: "
    ">>>>>>> REPLACE"
)
CODE_FILE = "### {path}\n{content}"

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def parse_instance(iid):
    parts = iid.split(".")
    head = parts[0]                       # owner__repo
    commit = parts[1] if len(parts) > 1 else ""
    owner, repo = (head.split("__", 1) + [""])[:2] if "__" in head else ("", head)
    repo_full = f"{owner}/{repo}" if owner else head
    mirror = f"{head}.{commit}"           # swesmith mirror repo name
    tail = parts[2] if len(parts) > 2 else ""
    strat = tail.rsplit("__", 1)[0] if "__" in tail else ("".join(c for c in tail if not c.isdigit()) or "numeric")
    return repo_full, commit, mirror, strat

def touched_files(patch):
    return re.findall(r'diff --git a/(\S+) b/', patch)

def is_binary_only(patch):
    if "GIT binary patch" in patch: return "@@" not in patch
    return "Binary files" in patch and "@@" not in patch

def load_verified():
    import pyarrow as pa, pyarrow.ipc as ipc
    files = glob.glob(VERIFIED_CACHE + "/**/*.arrow", recursive=True) + \
            glob.glob(VERIFIED_CACHE + "/**/*.parquet", recursive=True)
    ids, repo_commits = set(), defaultdict(set)
    for f in files:
        if f.endswith(".arrow"):
            with pa.memory_map(f) as src:
                try: t = ipc.open_stream(src).read_all()
                except Exception: t = ipc.open_file(src).read_all()
        else:
            t = pq.read_table(f)
        for r in t.select(["instance_id", "repo", "base_commit"]).to_pylist():
            ids.add(r["instance_id"]); repo_commits[r["repo"]].add(r["base_commit"])
    return ids, repo_commits

def fetch_file(mirror, branch, path, retries=5):
    url = f"https://raw.githubusercontent.com/swesmith/{mirror}/{branch}/{path}"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404: return None            # file genuinely absent
            back = (8.0 * (i + 1)) if e.code == 429 else (1.5 * (i + 1))  # throttle harder on 429
            last = e; time.sleep(back)
        except Exception as e:
            last = e; time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)     # 0 = full
    ap.add_argument("--out", default="~/esft/data/rl/grpo_prompts.jsonl")
    ap.add_argument("--stats", default="~/esft/data/rl/grpo_prompts_stats.json")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()

    log("loading decontam keys...")
    ver_ids, ver_repo_commits = load_verified()
    term_tasks = set(json.load(open(DECONTAM_TERMINAL)).get("tasks", []))
    log(f"  verified {len(ver_ids)} / terminal {len(term_tasks)}")

    log("pass A: resolved=True instances...")
    resolved_ids = set()
    for tf in sorted(glob.glob(TRAJ_DIR + "/train-*.parquet")):
        for r in pq.read_table(tf, columns=["instance_id", "resolved"]).to_pylist():
            if r["resolved"]: resolved_ids.add(r["instance_id"])
    log(f"  resolved unique {len(resolved_ids)}")

    log("loading base gold patch + problem_statement...")
    base = {}
    for bf in sorted(glob.glob(BASE_DIR + "/*.parquet")):
        for r in pq.read_table(bf, columns=["instance_id", "patch", "problem_statement"]).to_pylist():
            iid = r["instance_id"]
            if iid in resolved_ids and iid not in base:
                base[iid] = (r["patch"] or "", r["problem_statement"] or "")
    log(f"  base join {len(base)}/{len(resolved_ids)}")

    # selection (approved logic), dedup, decontam, quality
    drops = Counter(); seen = set(); selected = []
    for tf in sorted(glob.glob(TRAJ_DIR + "/train-*.parquet")):
        for r in pq.read_table(tf, columns=["instance_id", "resolved"]).to_pylist():
            if not r["resolved"]: continue
            iid = r["instance_id"]
            if iid in seen: drops["dup_instance"] += 1; continue
            seen.add(iid)
            repo, commit8, mirror, strat = parse_instance(iid)
            if iid in ver_ids: drops["decontam_instance_id"] += 1; continue
            if iid in term_tasks or repo.split("/")[-1] in term_tasks: drops["decontam_terminal"] += 1; continue
            if repo in ver_repo_commits and commit8 and any(bc.startswith(commit8) for bc in ver_repo_commits[repo]):
                drops["decontam_repo_commit"] += 1; continue
            if iid not in base: drops["no_oracle_in_base"] += 1; continue
            patch, problem = base[iid]
            if not patch.strip(): drops["empty_patch"] += 1; continue
            if len(patch.encode()) > MAX_PATCH_BYTES: drops["patch_too_large"] += 1; continue
            if is_binary_only(patch): drops["binary_only"] += 1; continue
            if not problem.strip(): drops["empty_problem_statement"] += 1; continue
            tfiles = touched_files(patch)
            if not tfiles: drops["no_touched_files"] += 1; continue
            selected.append({"iid": iid, "repo": repo, "commit8": commit8, "mirror": mirror,
                             "strat": strat, "patch": patch, "problem": problem, "files": tfiles})
    selected_pre_fetch = len(selected)
    log(f"  selected before fetch: {selected_pre_fetch}")
    if args.limit: selected = selected[:args.limit]; log(f"  LIMIT -> {len(selected)}")
    n_fetch_input = len(selected)

    # concurrent fetch of touched files at buggy branch (=instance_id), and
    # materialize the GOLD FIX = reverse of base bug-introduction patch.
    # (SWE-smith base `patch` is the perturbation fixed->buggy; the agent must
    #  produce buggy->fixed, so oracle = reverse, generated via `git apply -R`.)
    log(f"fetching touched files ({sum(len(s['files']) for s in selected)} files, {args.workers} workers)...")
    def build_one(s):
        contents = {}
        for fp in s["files"]:
            c = fetch_file(s["mirror"], s["iid"], fp)   # branch = instance_id (buggy state)
            if c is None: return ("fetch_404", s["iid"], fp)
            contents[fp] = c
        # code context shown to the model = buggy files
        s["content"] = "\n\n".join(CODE_FILE.format(path=fp, content=contents[fp]) for fp in s["files"])
        s["repo_files"] = contents                     # pre-fix (buggy) = what the model sees
        # gold fix + post-fix files via temp git repo: write buggy, reverse-apply
        # the SWE-smith bug patch (== apply the fix), diff, and read fixed files.
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
                   "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            for fp, c in contents.items():
                full = os.path.join(d, fp); os.makedirs(os.path.dirname(full) or d, exist_ok=True)
                open(full, "w").write(c)
            subprocess.run(["git", "add", "-A"], cwd=d, check=True, env=env)
            subprocess.run(["git", "commit", "-qm", "buggy"], cwd=d, check=True, env=env)
            pf = os.path.join(d, "bug.patch"); open(pf, "w").write(s["patch"])
            rr = subprocess.run(["git", "apply", "-R", "-p1", pf], cwd=d, capture_output=True, text=True)
            if rr.returncode != 0:
                return ("bug_patch_reverse_fail", s["iid"], rr.stderr.strip()[:100])
            fix = subprocess.run(["git", "diff"], cwd=d, capture_output=True, text=True).stdout
            if not fix.strip():
                return ("empty_gold_fix", s["iid"], None)
            s["gold_fix"] = fix
            s["bug_patch"] = s["patch"]                 # audit: original bug-introduction diff
            # post-fix content of the touched files (working tree is now fixed)
            newf = {}
            for fp in s["files"]:
                full = os.path.join(d, fp)
                newf[fp] = open(full).read() if os.path.exists(full) else ""
            s["oracle_new_files"] = newf
            # HARD GATE: gold fix must forward-apply to the buggy files (what reward relies on).
            subprocess.run(["git", "checkout", "-q", "--", "."], cwd=d, check=True, env=env)  # restore buggy (HEAD)
            gp = os.path.join(d, "gold.patch"); open(gp, "w").write(fix)
            ck = subprocess.run(["git", "apply", "--check", "-p1", gp], cwd=d, capture_output=True, text=True)
            if ck.returncode != 0:
                return ("oracle_apply_fail", s["iid"], ck.stderr.strip()[:100])
        return ("ok", s, None)
    built = []
    fetch_fail = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, s): s for s in selected}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                fetch_fail["fetch_error"] += 1; continue
            done += 1
            if done % 500 == 0: log(f"  fetched {done}/{len(selected)}")
            if res[0] == "ok": built.append(res[1])
            else: fetch_fail[res[0]] += 1
    log(f"  built {len(built)}, fetch drops {dict(fetch_fail)}")
    for k, v in fetch_fail.items(): drops[k] += v

    log("tokenizing + prompt assembly...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    sidecar_path = args.out.replace(".jsonl", "_files.jsonl")
    tok_lens = []; by_strat = Counter(); n_written = 0; samples = []
    inv_violation = 0
    fout = open(args.out, "w")
    fside = open(sidecar_path, "w")
    for s in built:
        # prompt code context built from the SAME dict stored as repo_files (single source)
        content = "\n\n".join(CODE_FILE.format(path=fp, content=s["repo_files"][fp]) for fp in s["files"])
        user = AGENTLESS_REPAIR.format(problem_statement=s["problem"], content=content)
        pm = [{"role": "system", "content": THINKING_SYSTEM},
              {"role": "user", "content": user}]
        rendered = tok.apply_chat_template(pm, add_generation_prompt=True, tokenize=False)
        tlen = len(tok(rendered, add_special_tokens=False).input_ids)
        if tlen > MAX_PROMPT_TOKENS:
            drops["prompt_gt_24k"] += 1; continue
        # HARD INVARIANT: prompt must NOT contain the gold fix nor any post-fix file content
        blob = pm[0]["content"] + "\n" + pm[1]["content"]
        if s["gold_fix"] in blob or any(nf and nf in blob for nf in s["oracle_new_files"].values()):
            inv_violation += 1; drops["invariant_leak"] += 1; continue
        rec = {"prompt_messages": pm, "oracle_patch": s["gold_fix"], "repo": s["repo"],
               "base_commit": s["commit8"], "instance_id": s["iid"], "source": SOURCE,
               "strategy": s["strat"], "prompt_tokens": tlen,
               "oracle_files": s["files"],
               "bug_patch": s["bug_patch"],
               "oracle_orientation": "buggy->fixed (reverse of SWE-smith base bug patch); forward-applies to repo_files"}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # sidecar: file contents for the reward function (pre = repo_files, post = oracle_new_files)
        fside.write(json.dumps({"instance_id": s["iid"], "repo_files": s["repo_files"],
                                "oracle_new_files": s["oracle_new_files"]}, ensure_ascii=False) + "\n")
        n_written += 1; by_strat[s["strat"]] += 1; tok_lens.append(tlen)
        if len(samples) < 2:
            samples.append({"instance_id": s["iid"], "repo": s["repo"], "prompt_tokens": tlen,
                            "system_head_400": pm[0]["content"][:400],
                            "user_head_400": pm[1]["content"][:400],
                            "gold_fix_head_300": s["gold_fix"][:300]})
    fout.close(); fside.close()

    tok_lens.sort()
    pct = lambda p: tok_lens[min(len(tok_lens) - 1, int(p * len(tok_lens)))] if tok_lens else 0
    main_mb = round(os.path.getsize(args.out) / 1e6, 1)
    side_mb = round(os.path.getsize(sidecar_path) / 1e6, 1) if os.path.exists(sidecar_path) else 0
    stats = {
        "mode": "SAMPLE" if args.limit else "FULL",
        "source": SOURCE,
        "prompt_format": "SWE-RL agentless-repair (arXiv:2502.18449)",
        "data_contract": {
            "main_jsonl": args.out + f" ({main_mb} MB): prompt_messages[system,user], oracle_patch(gold fix), oracle_files, repo, base_commit, instance_id, source, strategy, prompt_tokens",
            "sidecar_jsonl": sidecar_path + f" ({side_mb} MB): instance_id, repo_files{{path:pre-fix content}}, oracle_new_files{{path:post-fix content}}",
            "reward_pair": "reward uses (repo_files, oracle_new_files) per instance_id; oracle_patch is the equivalent buggy->fixed diff",
            "invariant": "prompt (system+user) contains ONLY pre-fix code (repo_files) + issue; never gold fix / post-fix content. Enforced; violations dropped.",
            "invariant_violations_dropped": inv_violation,
        },
        "oracle_patch_source": "SWE-bench/SWE-smith base (join by instance_id; trajectory patch column corrupt)",
        "code_context_source": "raw.githubusercontent.com/swesmith/{mirror}/{instance_id-branch}/{file} (buggy state, no clone)",
        "search_replace_spec": {
            "search_start": "<<<<<<< SEARCH",
            "divider": "=======",
            "replace_end": ">>>>>>> REPLACE",
            "file_path_header": "### {path}",
            "solution_wrapper": "<think>...</think>\\n<solution>\\n```\\n### path\\n<blocks>\\n```\\n</solution>",
            "note": "verbatim from facebookresearch/swe-rl core/prompts.py; reward agent applies blocks to oracle_files content and scores diff similarity vs oracle_patch",
        },
        "reconciliation": {
            "unique_resolved_instances": len(resolved_ids),
            "minus_selection_drops": {k: drops[k] for k in
                ("decontam_instance_id", "decontam_terminal", "decontam_repo_commit",
                 "no_oracle_in_base", "empty_patch", "patch_too_large", "binary_only",
                 "empty_problem_statement", "no_touched_files") if drops.get(k)},
            "equals_selected_pre_fetch": selected_pre_fetch,
            "fetch_input (after --limit)": n_fetch_input,
            "minus_fetch_drops": {k: drops[k] for k in
                ("fetch_404", "fetch_error", "bug_patch_reverse_fail", "empty_gold_fix",
                 "oracle_apply_fail") if drops.get(k)},
            "equals_built": len(built),
            "minus_prompt_gt_24k": drops.get("prompt_gt_24k", 0),
            "equals_written": n_written,
            "dup_instance_rows_collapsed": drops.get("dup_instance", 0),
        },
        "drops": dict(drops),
        "kept_by_strategy": dict(by_strat.most_common()),
        "prompt_token_len": {
            "n": len(tok_lens), "min": tok_lens[0] if tok_lens else 0, "p50": pct(.5),
            "p90": pct(.9), "p95": pct(.95), "p99": pct(.99),
            "max": tok_lens[-1] if tok_lens else 0,
            "mean": round(sum(tok_lens) / len(tok_lens), 1) if tok_lens else 0,
            "gt_16384": sum(1 for x in tok_lens if x > 16384),
            "gt_8192": sum(1 for x in tok_lens if x > 8192),
        },
        "max_prompt_tokens_cap": MAX_PROMPT_TOKENS,
        "tokenizer": TOKENIZER_PATH,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump({"stats": stats, "samples": samples}, open(args.stats, "w"), ensure_ascii=False, indent=2)
    log("=== DONE ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    for s in samples:
        print(f"\n--- {s['instance_id']} ({s['repo']}, {s['prompt_tokens']} tok) ---")
        print("SYSTEM[:400]:", s["system_head_400"])
        print("USER[:400]:", s["user_head_400"])
        print("GOLD_FIX[:300]:", s["gold_fix_head_300"])

if __name__ == "__main__":
    main()
