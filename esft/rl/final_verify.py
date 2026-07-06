import json, os, subprocess, tempfile, random
MAIN="~/esft/data/rl/grpo_prompts.jsonl"
SIDE="~/esft/data/rl/grpo_prompts_files.jsonl"
main=[json.loads(l) for l in open(MAIN)]
side=[json.loads(l) for l in open(SIDE)]
print("main lines:",len(main),"side lines:",len(side))
# alignment
aladr = all(m["instance_id"]==s["instance_id"] for m,s in zip(main,side))
print("instance_id aligned row-by-row:", aladr)
# invariant on written records
inv_bad=0
for m,s in zip(main[:1000],side[:1000]):
    blob=m["prompt_messages"][0]["content"]+"\n"+m["prompt_messages"][1]["content"]
    if m["oracle_patch"] in blob or any(v and v in blob for v in s["oracle_new_files"].values()):
        inv_bad+=1
print("invariant violations in first 1000 written:", inv_bad)
# end-to-end: apply gold fix to repo_files -> must equal oracle_new_files (offline, no net)
random.seed(0)
idx=random.sample(range(len(main)),40)
ok_apply=0; match_new=0
for i in idx:
    m,s=main[i],side[i]
    with tempfile.TemporaryDirectory() as d:
        for fp,c in s["repo_files"].items():
            full=os.path.join(d,fp); os.makedirs(os.path.dirname(full) or d,exist_ok=True); open(full,"w").write(c)
        subprocess.run(["git","init","-q"],cwd=d,check=True)
        pf=os.path.join(d,"g.patch"); open(pf,"w").write(m["oracle_patch"])
        r=subprocess.run(["git","apply","-p1",pf],cwd=d,capture_output=True,text=True)
        if r.returncode!=0: continue
        ok_apply+=1
        allmatch=True
        for fp,expect in s["oracle_new_files"].items():
            got=open(os.path.join(d,fp)).read()
            if got!=expect: allmatch=False; break
        if allmatch: match_new+=1
print(f"gold fix applies to repo_files: {ok_apply}/40")
print(f"applied result == oracle_new_files: {match_new}/40")
# prompt contains buggy (repo_files) content
buggy_in=sum(1 for m,s in zip(main[:200],side[:200]) if all(v in m["prompt_messages"][1]["content"] for v in s["repo_files"].values()))
print(f"repo_files content present in prompt (first 200): {buggy_in}/200")
