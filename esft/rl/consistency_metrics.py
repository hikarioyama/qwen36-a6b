#!/usr/bin/env python3
"""Consistency metrics for SWE-RL layer-2 rollouts (CPU-only, numpy-only).

Given one rollout jsonl per arm (schema below), compute *within-prompt*
consistency of the sampled completions and compare arms paired on instance_id.

Rollout jsonl (1 line = 1 prompt):
  {"instance_id", "n", "temp",
   "rewards":[n strict], "rewards_lenient":[n],
   "best", "best_lenient",
   "completions":[{"text","finish","tokens","reward","format_valid","reward_lenient"}...]}

Reward semantics (see reward.py): reward in {-1.0} U [0,1].
  -1.0  == FormatError (completion did not parse/apply into a valid patch)
  [0,1] == mean per-file diff similarity vs oracle (1.0 = exact).
  "lenient" adds a raw-diff fallback on top of strict SEARCH/REPLACE.
  => reward_lenient > -1  <=>  format-valid (lenient).  reward_lenient > 0.5 == "success".

Infra noise: a completion whose finish reason starts with "error" (e.g. a serve
"Connection reset by peer") is NOT a model output. We call the error-free subset
"clean" and compute the model-behavior metrics on it; the "all" variant keeps
the errored samples (they score -1) so the contamination is visible, not hidden.

All CIs are 95% percentile bootstrap that RESAMPLES INSTANCES (cluster bootstrap:
the prompt is the cluster, its n samples move together), because samples inside a
prompt are correlated -- a per-sample bootstrap is optimistic.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Callable, Optional

import numpy as np

FLOOR = -1.0            # format-fail reward
SUCCESS = 0.5           # reward_lenient > SUCCESS == "success"
RNG = np.random.default_rng(0)
B = 10000               # bootstrap resamples


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def arm_name(path: str) -> str:
    b = os.path.basename(path)
    for ext in (".jsonl", ".json"):
        if b.endswith(ext):
            b = b[: -len(ext)]
    if b.startswith("l2_"):
        b = b[3:]
    return b


def is_error(c: dict) -> bool:
    """Infra failure (not a model output): finish reason marked 'error...'."""
    return str(c.get("finish", "")).startswith("error")


def load_arm(path: str) -> dict[str, dict]:
    """{instance_id: record}. Each record's completions get derived subsets."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["instance_id"]] = d
    return out


# --------------------------------------------------------------------------- #
# per-prompt scalar extractors  (return None when undefined for that prompt)
# --------------------------------------------------------------------------- #
def _rl(comps: list[dict]) -> np.ndarray:
    return np.array([c["reward_lenient"] for c in comps], dtype=float)


def _tokens(comps: list[dict]) -> np.ndarray:
    return np.array([c["tokens"] for c in comps], dtype=float)


def clean_comps(rec: dict) -> list[dict]:
    return [c for c in rec["completions"] if not is_error(c)]


def m_reward_std(comps: list[dict]) -> Optional[float]:
    """Sample std (ddof=1) of reward_lenient across a prompt's completions."""
    r = _rl(comps)
    if r.size < 2:
        return None
    return float(np.std(r, ddof=1))


def m_passk_gap(comps: list[dict]) -> Optional[float]:
    """best - mean of reward_lenient (pass@k concentration; small = uniform)."""
    r = _rl(comps)
    if r.size < 1:
        return None
    return float(r.max() - r.mean())


def m_len_cv(comps: list[dict]) -> Optional[float]:
    """Coefficient of variation of completion tokens (std/mean, ddof=1)."""
    t = _tokens(comps)
    t = t[t > 0]                      # drop 0-token stubs
    if t.size < 2 or t.mean() == 0:
        return None
    return float(np.std(t, ddof=1) / t.mean())


def m_valid_frac(comps: list[dict]) -> Optional[float]:
    """Fraction of a prompt's completions that are format-valid (> -1)."""
    r = _rl(comps)
    if r.size < 1:
        return None
    return float(np.mean(r > FLOOR))


def m_success_frac(comps: list[dict]) -> Optional[float]:
    """Among prompts with >=1 success, fraction of completions that succeed."""
    r = _rl(comps)
    if r.size < 1 or not np.any(r > SUCCESS):
        return None                  # only defined on prompts that ever succeed
    return float(np.mean(r > SUCCESS))


def m_valid_repro(comps: list[dict]) -> Optional[float]:
    """Among prompts with >=1 format-valid, fraction valid (reproducibility)."""
    r = _rl(comps)
    if r.size < 1 or not np.any(r > FLOOR):
        return None
    return float(np.mean(r > FLOOR))


def m_trunc_frac(comps: list[dict]) -> Optional[float]:
    """Fraction of a prompt's completions truncated at the length cap."""
    if not comps:
        return None
    return float(np.mean([c.get("finish") == "length" for c in comps]))


# --------------------------------------------------------------------------- #
# distribution + bootstrap helpers
# --------------------------------------------------------------------------- #
def describe(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def boot_mean_ci(vals: list[Optional[float]], b: int = B) -> Optional[tuple]:
    """95% percentile-bootstrap CI for the mean of a per-instance scalar.

    Resamples the instances themselves (cluster bootstrap). None entries are
    dropped first; the resample is over the defined instances.
    """
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size < 2:
        return None
    idx = RNG.integers(0, a.size, size=(b, a.size))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_paired_ci(pairs: list[tuple[float, float]], b: int = B) -> Optional[dict]:
    """Paired diff (arm - base) mean + 95% CI, resampling instance pairs."""
    p = np.array([(x, y) for x, y in pairs if x is not None and y is not None], dtype=float)
    if p.shape[0] < 2:
        return None
    diff = p[:, 0] - p[:, 1]
    idx = RNG.integers(0, diff.size, size=(b, diff.size))
    md = diff[idx].mean(axis=1)
    return {
        "n_pairs": int(diff.size),
        "mean_diff": float(diff.mean()),
        "ci": [float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))],
        "base_mean": float(p[:, 1].mean()),
        "arm_mean": float(p[:, 0].mean()),
    }


def cluster_rate_ci(recs: list[dict], numer: Callable, denom: Callable, b: int = B) -> dict:
    """Pooled rate (sum numer / sum denom over prompts) + cluster-bootstrap CI."""
    ns = np.array([numer(r) for r in recs], dtype=float)
    ds = np.array([denom(r) for r in recs], dtype=float)
    tot = ds.sum()
    rate = float(ns.sum() / tot) if tot else float("nan")
    idx = RNG.integers(0, len(recs), size=(b, len(recs)))
    bn = ns[idx].sum(axis=1)
    bd = ds[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        br = bn / bd
    br = br[np.isfinite(br)]
    ci = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))] if br.size else [float("nan")] * 2
    return {"rate": rate, "ci": ci, "numer": int(ns.sum()), "denom": int(tot)}


# --------------------------------------------------------------------------- #
# per-arm computation
# --------------------------------------------------------------------------- #
METRICS = {
    "reward_std": (m_reward_std, "M1 reward_lenient std (spread; ddof=1, n>=2)"),
    "passk_gap": (m_passk_gap, "M2 best-mean gap (pass@k concentration)"),
    "len_cv": (m_len_cv, "M4 token CV (std/mean)"),
    "valid_frac": (m_valid_frac, "M6 format-valid fraction"),
    "trunc_frac": (m_trunc_frac, "M5 length-truncation fraction"),
    "success_frac": (m_success_frac, "M3 success reproducibility (prompts w/ >=1 success)"),
    "valid_repro": (m_valid_repro, "M6b format-valid reproducibility (prompts w/ >=1 valid)"),
}


def per_arm(recs_by_id: dict[str, dict]) -> dict:
    recs = list(recs_by_id.values())
    all_comp = [c for r in recs for c in r["completions"]]
    n_comp = len(all_comp)

    finish = Counter(
        "error" if is_error(c) else c.get("finish", "?") for c in all_comp
    )
    n_err = sum(v for k, v in finish.items() if k == "error")

    out = {
        "n_prompts": len(recs),
        "n_completions": n_comp,
        "finish": dict(finish),
        "error_rate": n_err / n_comp if n_comp else 0.0,
        "trunc_rate_all": finish.get("length", 0) / n_comp if n_comp else 0.0,
        "metrics": {},
    }

    # per-prompt scalars on two subsets: clean (model outputs) and all
    for key, (fn, _desc) in METRICS.items():
        clean_vals = [fn(clean_comps(r)) for r in recs]
        all_vals = [fn(r["completions"]) for r in recs]
        out["metrics"][key] = {
            "clean": {"dist": describe(clean_vals), "boot": boot_mean_ci(clean_vals)},
            "all": {"dist": describe(all_vals), "boot": boot_mean_ci(all_vals)},
            "_clean_vals": clean_vals,   # kept for pairing (stripped before JSON dump)
            "_all_vals": all_vals,
        }

    # pooled rates with cluster CI (denominator = clean completions)
    out["valid_rate_clean"] = cluster_rate_ci(
        recs,
        numer=lambda r: sum(c["reward_lenient"] > FLOOR for c in clean_comps(r)),
        denom=lambda r: len(clean_comps(r)),
    )
    out["success_rate_clean"] = cluster_rate_ci(
        recs,
        numer=lambda r: sum(c["reward_lenient"] > SUCCESS for c in clean_comps(r)),
        denom=lambda r: len(clean_comps(r)),
    )
    out["_recs"] = recs_by_id
    return out


def strip_internal(d: dict) -> dict:
    """Remove non-serializable / bulky internals for JSON dump."""
    clean = {}
    for arm, res in d.items():
        r = {k: v for k, v in res.items() if k != "_recs"}
        r["metrics"] = {
            mk: {sk: sv for sk, sv in mv.items() if not sk.startswith("_")}
            for mk, mv in res["metrics"].items()
        }
        clean[arm] = r
    return clean


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def fmt_ci(ci) -> str:
    if ci is None:
        return "n/a"
    if isinstance(ci, tuple):  # (mean, lo, hi)
        return f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report(arms: dict, base_key: Optional[str]) -> str:
    L = []
    L.append("# consistency metrics -- layer-2 SWE-RL rollouts\n")
    for arm, res in arms.items():
        L.append(f"## arm: {arm}")
        L.append(f"- prompts={res['n_prompts']}  completions={res['n_completions']}")
        L.append(f"- finish={res['finish']}")
        L.append(f"- error_rate(infra)={res['error_rate']:.3f}  trunc_rate(all)={res['trunc_rate_all']:.3f}")
        vr, sr = res["valid_rate_clean"], res["success_rate_clean"]
        L.append(f"- format-valid rate (clean) = {vr['rate']:.3f} {fmt_ci(vr['ci'])}  ({vr['numer']}/{vr['denom']})")
        L.append(f"- success rate    (clean) = {sr['rate']:.3f} {fmt_ci(sr['ci'])}  ({sr['numer']}/{sr['denom']})")
        for key, (_, desc) in METRICS.items():
            m = res["metrics"][key]["clean"]
            d = m["dist"]
            if d["n"] == 0:
                L.append(f"  - {key:12s} clean: n=0 (undefined for all prompts)")
                continue
            L.append(
                f"  - {key:12s} clean: n={d['n']} mean={d['mean']:.3f} med={d['median']:.3f} "
                f"p10={d['p10']:.3f} p90={d['p90']:.3f}  boot95={fmt_ci(m['boot'])}"
            )
        L.append("")

    if base_key and len(arms) >= 2:
        for arm, res in arms.items():
            if arm == base_key:
                continue
            L.append(f"## paired diff: {arm} - {base_key}  (per-instance, cluster bootstrap)")
            base = arms[base_key]
            ids = [i for i in res["_recs"] if i in base["_recs"]]
            L.append(f"- shared instances = {len(ids)}")
            for key, (_, desc) in METRICS.items():
                for subset in ("clean",):
                    a_vals = res["metrics"][key][f"_{subset}_vals"]
                    b_vals = base["metrics"][key][f"_{subset}_vals"]
                    a_map = dict(zip(res["_recs"].keys(), a_vals))
                    b_map = dict(zip(base["_recs"].keys(), b_vals))
                    pairs = [(a_map[i], b_map[i]) for i in ids]
                    pc = boot_paired_ci(pairs)
                    if pc is None:
                        L.append(f"  - {key:12s} ({subset}): too few paired ({desc})")
                    else:
                        star = "" if pc["ci"][0] <= 0 <= pc["ci"][1] else "  *"
                        L.append(
                            f"  - {key:12s} ({subset}): base={pc['base_mean']:.3f} "
                            f"{arm}={pc['arm_mean']:.3f} diff={pc['mean_diff']:+.3f} "
                            f"CI[{pc['ci'][0]:+.3f},{pc['ci'][1]:+.3f}] n={pc['n_pairs']}{star}"
                        )
            L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="rollout jsonl per arm")
    ap.add_argument("--base", default=None, help="arm name to use as paired reference")
    ap.add_argument("--json-out", default=None, help="write metric summary json here")
    args = ap.parse_args()

    arms: dict[str, dict] = {}
    for path in args.files:
        arms[arm_name(path)] = per_arm(load_arm(path))

    base_key = args.base
    if base_key is None:
        for k in arms:
            if "base" in k:
                base_key = k
                break
        if base_key is None:
            base_key = list(arms)[0]

    text = report(arms, base_key)
    print(text)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(strip_internal(arms), f, indent=2)
        print(f"\n[wrote {args.json_out}]")


if __name__ == "__main__":
    main()
