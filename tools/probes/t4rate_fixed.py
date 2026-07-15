import json, os, sys
sys.path.insert(0,".")
os.environ["SELFGEN_NOTHINK"]="1"; os.environ["SELFGEN_STAGE_HINT"]="1"
import selfgen_toolcall_intent_v1 as intent
import selfgen_toolcall_v1 as v1
run=sys.argv[1]; ep=sys.argv[2]; n=int(sys.argv[3])
seeds=json.load(open(f"data/selfgen_toolcall_intent_v1/{run}/seeds.json"))["seeds"]
cli=intent._VLLMClient(ep,model="Qwen/Qwen3.6-35B-A3B")
spec=v1.GenerationSpec(20260714,0.7,2048,4,0)
t4s=[s for s in seeds if s["tier"]=="T4"][:n]
npass=0; stage_fail={}
for t4 in t4s:
    tool_map={t["name"]:t for t in t4["tools"]}
    prior=[]; ok=True   # 本物 worker と同じく累積
    for st,exp in enumerate(t4["expected_stages"]):
        calls,reasons,_,_=intent._select_candidate_details(cli.generate(intent._vllm_prompt(t4,st,prior),spec),exp,tool_map)
        if calls is None:
            ok=False; stage_fail[st]=stage_fail.get(st,0)+1; break
        prior.extend([intent.mock_execute(c,st,t4["pattern"]) for c in calls])  # ← extend (置換でなく)
    npass+=ok
print(f"{run} T4 (n={n}, best_of=4, PRIOR-EXTEND): {npass}/{n} = {npass/n:.2f}  stage_fail={stage_fail}", flush=True)
