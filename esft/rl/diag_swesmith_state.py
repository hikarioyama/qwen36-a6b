import json, urllib.request
recs = [json.loads(l) for l in open("~/esft/data/rl/sample_v1.jsonl")]
def fetch(mirror, branch, path):
    url = f"https://raw.githubusercontent.com/swesmith/{mirror}/{branch}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8","replace")
r = recs[0]
iid = r["instance_id"]; parts=iid.split("."); mirror=parts[0]+"."+parts[1]
fp = r["oracle_files"][0]
print("instance:", iid, "file:", fp)
main = fetch(mirror,"main",fp).splitlines()
br = fetch(mirror,iid,fp).splitlines()
print(f"main lines={len(main)} branch lines={len(br)}")
print("\n=== GOLD PATCH (full) ===")
print(r["oracle_patch"])
print("\n=== MAIN around 44-60 ===")
for i in range(43,60):
    if i < len(main): print(f"{i+1:4} {main[i]}")
print("\n=== BRANCH(iid) around 44-60 ===")
for i in range(43,60):
    if i < len(br): print(f"{i+1:4} {br[i]}")
