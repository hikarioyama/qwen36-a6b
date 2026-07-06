#!/usr/bin/env python
"""Validate that the oracle patch cleanly applies to the fetched buggy-state files.
This proves the code context in the prompt is the exact state the reward function
will apply model edits to."""
import json, os, re, subprocess, tempfile, urllib.request, urllib.error

SAMPLE = "~/esft/data/rl/sample_v1.jsonl"

def fetch(mirror, branch, path):
    url = f"https://raw.githubusercontent.com/swesmith/{mirror}/{branch}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

def main():
    recs = [json.loads(l) for l in open(SAMPLE)]
    ok = 0; fail = 0; fails = []
    for r in recs:
        iid = r["instance_id"]
        parts = iid.split("."); mirror = parts[0] + "." + parts[1]
        patch = r["oracle_patch"]; files = r["oracle_files"]
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            good = True
            for fp in files:
                try:
                    c = fetch(mirror, iid, fp)
                except Exception:
                    good = False; break
                full = os.path.join(d, fp)
                os.makedirs(os.path.dirname(full) or d, exist_ok=True)
                open(full, "w").write(c)
            if not good:
                fail += 1; fails.append((iid, "fetch")); continue
            pf = os.path.join(d, "o.patch"); open(pf, "w").write(patch)
            # --check: does patch apply to current (fetched buggy) state?
            res = subprocess.run(["git", "apply", "--check", "-p1", pf], cwd=d,
                                 capture_output=True, text=True)
            if res.returncode == 0:
                ok += 1
            else:
                fail += 1; fails.append((iid, res.stderr.strip()[:120]))
    print(f"oracle patch applies to fetched buggy state: {ok}/{ok+fail}")
    for iid, err in fails[:8]:
        print("  FAIL", iid, "::", err)

if __name__ == "__main__":
    main()
