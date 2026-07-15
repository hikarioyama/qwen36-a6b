import json, os, sys
sys.path.insert(0,".")
os.environ["SELFGEN_NOTHINK"]="1"; os.environ["SELFGEN_STAGE_HINT"]="1"
import selfgen_toolcall_intent_v1 as intent
import selfgen_toolcall_v1 as v1
run=sys.argv[1]; ep=sys.argv[2]
seeds=json.load(open(f"data/selfgen_toolcall_intent_v1/{run}/seeds.json"))["seeds"]
cli=intent._VLLMClient(ep,model="Qwen/Qwen3.6-35B-A3B")
spec=v1.GenerationSpec(20260714,0.0,2048,1,0)
t4=[s for s in seeds if s["tier"]=="T4"][int(sys.argv[3]) if len(sys.argv)>3 else 0]
tool_map={t["name"]:t for t in t4["tools"]}
print(f"RUN={run} EP={ep} seed={t4['seed_id']}")
prior=[]
npass=0
for st,exp in enumerate(t4["expected_stages"]):
    raw=cli.generate(intent._vllm_prompt(t4,st,prior),spec)[0]
    calls,reason=v1.parse_model_turn(raw,tool_map)
    match = calls is not None and intent.canonical(calls)==intent.canonical(exp)
    npass+=match
    print(f"  stage{st}: {'MATCH' if match else 'MISS'} exp={[c['name'] for c in exp]} got={[c['name'] for c in calls] if calls else reason}")
    if not match:
        print(f"    exp_full={json.dumps(exp,ensure_ascii=False)}")
        print(f"    got_full={json.dumps(calls,ensure_ascii=False) if calls else None}")
        print(f"    raw={raw[:1500]!r}")
    prior.extend([intent.mock_execute(c,st,t4["pattern"]) for c in exp])  # extend (置換禁止)
print(f"stages matched: {npass}/{len(t4['expected_stages'])}")
