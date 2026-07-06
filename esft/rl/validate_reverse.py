import json, os, subprocess, tempfile, urllib.request
recs = [json.loads(l) for l in open("~/esft/data/rl/sample_v1.jsonl")]
def fetch(mirror, branch, path):
    url = f"https://raw.githubusercontent.com/swesmith/{mirror}/{branch}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent":"curl/8"})
    try: return urllib.request.urlopen(req, timeout=30).read().decode("utf-8","replace")
    except Exception: return None

def gold_fix(mirror, iid, files, bug_patch):
    """buggy(branch) + reverse(bug_patch) -> fixed; return git diff buggy->fixed."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git","init","-q"], cwd=d, check=True)
        subprocess.run(["git","config","user.email","x@x"], cwd=d)
        subprocess.run(["git","config","user.name","x"], cwd=d)
        for fp in files:
            c = fetch(mirror, iid, fp)
            if c is None: return None, "fetch"
            full=os.path.join(d,fp); os.makedirs(os.path.dirname(full) or d, exist_ok=True)
            open(full,"w").write(c)
        subprocess.run(["git","add","-A"], cwd=d, check=True)
        subprocess.run(["git","commit","-qm","buggy"], cwd=d, check=True)
        pf=os.path.join(d,"bug.patch"); open(pf,"w").write(bug_patch)
        r=subprocess.run(["git","apply","-R","-p1",pf], cwd=d, capture_output=True, text=True)
        if r.returncode!=0: return None, "reverse_apply:"+r.stderr.strip()[:80]
        diff=subprocess.run(["git","diff"], cwd=d, capture_output=True, text=True).stdout
        return diff, None

def forward_ok(mirror, iid, files, fix_patch):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git","init","-q"], cwd=d, check=True)
        for fp in files:
            c=fetch(mirror,iid,fp)
            full=os.path.join(d,fp); os.makedirs(os.path.dirname(full) or d, exist_ok=True)
            open(full,"w").write(c)
        pf=os.path.join(d,"f.patch"); open(pf,"w").write(fix_patch)
        return subprocess.run(["git","apply","--check","-p1",pf],cwd=d,capture_output=True).returncode==0

ok=0; fwd_ok=0; fail=[]
for r in recs:
    iid=r["instance_id"]; parts=iid.split("."); mirror=parts[0]+"."+parts[1]
    fix, err = gold_fix(mirror, iid, r["oracle_files"], r["oracle_patch"])
    if fix is None: fail.append((iid,err)); continue
    ok+=1
    if fix.strip() and forward_ok(mirror, iid, r["oracle_files"], fix): fwd_ok+=1
    else: fail.append((iid,"fwd_fail"))
print(f"gold-fix generated (reverse applied): {ok}/{len(recs)}")
print(f"gold-fix forward-applies to buggy prompt code: {fwd_ok}/{len(recs)}")
for iid,e in fail[:8]: print("  ",iid,"::",e)
# show one gold fix
for r in recs[:1]:
    iid=r["instance_id"]; parts=iid.split("."); mirror=parts[0]+"."+parts[1]
    fix,_=gold_fix(mirror,iid,r["oracle_files"],r["oracle_patch"])
    print("\n=== SAMPLE GOLD FIX (reverse of base patch) ==="); print(fix)
