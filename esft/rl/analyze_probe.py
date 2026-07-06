#!/usr/bin/env python
"""Decompose a format-probe raw jsonl into mechanism + paired-delta tables.

Usage: analyze_probe.py probe_out/probe_4096.jsonl [more.jsonl ...]

For each arm reports fmt_ok / envelope-tags / truncation / near-miss and strict &
lenient (mean, best-of-n). Then paired P-vs-F deltas (same prompt+sample seed) on
lenient reward with a paired cluster bootstrap CI, so the "does forcing hurt the
edit?" question gets a same-condition answer, not an across-arm eyeball.
"""
import json, sys, random
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        for l in open(p):
            l = l.strip()
            if l:
                r = json.loads(l); r["_src"] = p; rows.append(r)
    return rows


def near_miss_missing_think(r):
    """Envelope otherwise complete but the opening <think> is absent — the single
    cheapest-to-recover strict failure (a 1-line parser relax would rescue it)."""
    return (r["n_think_open"] == 0 and r["n_think_close"] == 1
            and r["n_sol_open"] == 1 and r["n_sol_close"] == 1
            and r["has_sr_marker"] and r["finish"] == "stop")


def pct(x): return f"{x*100:.1f}%"


def summarize(rows):
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    n_per = max(r["si"] for r in rows) + 1

    print(f"\n{'arm':<15}{'n':>4}{'fmt_ok':>8}{'env(SR)':>8}{'fb':>6}"
          f"{'strict_m':>10}{'strict_bo':>10}{'lenient_m':>10}{'lenient_bo':>11}")
    for arm in arms:
        rs = by[arm]; n = len(rs)
        fmt_ok = sum(r["format_valid"] for r in rs) / n
        env = sum(r["method"] == "search_replace" for r in rs) / n
        fb = sum(r["method"] == "diff_fallback" for r in rs) / n
        sm = sum(r["strict"] for r in rs) / n
        lm = sum(r["lenient"] for r in rs) / n
        bs = defaultdict(list); bl = defaultdict(list)
        for r in rs:
            bs[r["pi"]].append(r["strict"]); bl[r["pi"]].append(r["lenient"])
        sbo = sum(max(v) for v in bs.values()) / len(bs)
        lbo = sum(max(v) for v in bl.values()) / len(bl)
        print(f"{arm:<15}{n:>4}{pct(fmt_ok):>8}{pct(env):>8}{pct(fb):>6}"
              f"{sm:>10.4f}{sbo:>10.4f}{lm:>10.4f}{lbo:>11.4f}")

    # trunc-conditioned + start-marker (envelope INTENT, robust to truncation:
    # <solution> or a SEARCH marker appears even if the tail is cut off).
    print(f"\n{'arm':<15}{'trunc':>8}{'began_env':>10}{'sol_open':>9}"
          f"{'sol_close':>10}{'fmt_ok':>8}{'fmt_ok|nontrunc':>16}{'near-miss':>10}")
    for arm in arms:
        rs = by[arm]; n = len(rs)
        trunc = sum(r["finish"] == "length" for r in rs) / n
        began = sum((r["n_sol_open"] >= 1 or r["has_sr_marker"]) for r in rs) / n
        so = sum(r["n_sol_open"] >= 1 for r in rs) / n
        sc = sum(r["n_sol_close"] >= 1 for r in rs) / n
        fmt = sum(r["format_valid"] for r in rs) / n
        nt = [r for r in rs if r["finish"] != "length"]
        fmt_nt = (sum(r["format_valid"] for r in nt) / len(nt)) if nt else 0.0
        nm = sum(near_miss_missing_think(r) for r in rs) / n
        print(f"{arm:<15}{pct(trunc):>8}{pct(began):>10}{pct(so):>9}"
              f"{pct(sc):>10}{pct(fmt):>8}{pct(fmt_nt):>13}({len(nt)}){pct(nm):>10}")

    # paired P vs each F on lenient (key: same prompt pi + sample si => same seed)
    base = "P"
    if base in by:
        keyed = defaultdict(dict)
        for r in rows:
            keyed[(r["pi"], r["si"])][r["arm"]] = r
        print(f"\npaired Δlenient vs P (same prompt+seed), cluster bootstrap 95% CI:")
        for arm in arms:
            if arm == base:
                continue
            pairs = [(k[0], kv[base]["lenient"], kv[arm]["lenient"])
                     for k, kv in keyed.items() if base in kv and arm in kv]
            if not pairs:
                continue
            deltas = [b - a for _, a, b in pairs]
            mean_d = sum(deltas) / len(deltas)
            lo, hi = cluster_bootstrap([(pi, b - a) for pi, a, b in pairs])
            win = sum(d > 1e-9 for d in deltas); tie = sum(abs(d) <= 1e-9 for d in deltas)
            loss = sum(d < -1e-9 for d in deltas)
            print(f"  {arm:<15} Δ={mean_d:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
                  f"n_pairs={len(pairs)}  win/tie/loss={win}/{tie}/{loss}")


def cluster_bootstrap(pi_deltas, B=5000, seed=0):
    """Cluster (by prompt) bootstrap CI on the mean paired delta."""
    by_pi = defaultdict(list)
    for pi, d in pi_deltas:
        by_pi[pi].append(d)
    clusters = list(by_pi.values())
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        samp = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [d for c in samp for d in c]
        means.append(sum(flat) / len(flat))
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    summarize(rows)
