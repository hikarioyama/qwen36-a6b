"""T4 probe: FEW_SHOTS に multi-turn 継続例を注入して測る (seed/request は無改変)。
cx 対立仮説 (a): single-turn few-shot による継続プロトコル未学習、の直接検証。
scaffold は input-only なので効けば訓練データ無変更で T4 が直る。
usage: python t4rate_fewshot.py <run> <endpoint> <n> [natural|trans]
"""
import copy, json, os, sys
sys.path.insert(0, ".")
os.environ["SELFGEN_NOTHINK"] = "1"; os.environ["SELFGEN_STAGE_HINT"] = "1"
import selfgen_toolcall_intent_v1 as intent
import selfgen_toolcall_v1 as v1

# 継続プロトコルの実演: diverse 名、turn2 (2 receipt 複写)、turn3 (先行 call を反復しない)
CONTINUATION_SHOTS = [
    {"user": ("Schema: billing.accounts.open requires owner_tag; ledger-server-verify_entry requires "
              "audit_flag; postFinalNotice requires summary_ref, first_receipt, second_receipt. "
              "Request: open with \"item-5\" and verify with true at the same time; once both are done, "
              "post the final notice with \"item-8\" and the two receipts. Turn: 2. "
              "prior_tool_results: [{\"call\": \"billing.accounts.open\", \"ok\": true, \"result\": "
              "{\"receipt\": \"mock-aa11\", \"stage\": 1}}, {\"call\": \"ledger-server-verify_entry\", "
              "\"ok\": true, \"result\": {\"receipt\": \"mock-bb22\", \"stage\": 1}}]. "
              "Instruction: Turns 1..1 are already executed. Emit ONLY the new tool-call JSON for turn 2. "
              "Never repeat a call that was already executed."),
     "assistant": {"calls": [{"name": "postFinalNotice", "arguments": {
         "summary_ref": "item-8", "first_receipt": "mock-aa11", "second_receipt": "mock-bb22"}}]}},
]


def _patched_scaffold(seed, stage, tool_results):
    orig = v1.FEW_SHOTS
    v1.FEW_SHOTS = orig + CONTINUATION_SHOTS
    try:
        return _orig_scaffold(seed, stage, tool_results)
    finally:
        v1.FEW_SHOTS = orig


_orig_scaffold = v1.scaffold_prompt

if __name__ == "__main__":
    run = sys.argv[1]; ep = sys.argv[2]; n = int(sys.argv[3])
    style = sys.argv[4] if len(sys.argv) > 4 else "natural"
    v1.scaffold_prompt = _patched_scaffold
    seeds = json.load(open(f"data/selfgen_toolcall_intent_v1/{run}/seeds.json"))["seeds"]
    cli = intent._VLLMClient(ep, model="Qwen/Qwen3.6-35B-A3B")
    spec = v1.GenerationSpec(20260714, 0.7, 2048, 4, 0)
    t4s = []
    for s in seeds:
        if s["tier"] != "T4":
            continue
        s = copy.deepcopy(s)
        if style == "trans":
            s["user_request"] = s["transcription_request"]
        t4s.append(s)
        if len(t4s) >= n:
            break
    npass = 0; stage_fail = {}
    for t4 in t4s:
        tool_map = {t["name"]: t for t in t4["tools"]}
        prior = []; ok = True
        for st, exp in enumerate(t4["expected_stages"]):
            calls, reasons, _, _ = intent._select_candidate_details(
                cli.generate(intent._vllm_prompt(t4, st, prior), spec), exp, tool_map)
            if calls is None:
                ok = False; stage_fail[st] = stage_fail.get(st, 0) + 1; break
            prior.extend([intent.mock_execute(c, st, t4["pattern"]) for c in calls])
        npass += ok
    print(f"{run}+FEWSHOT-MT+{style.upper()} T4 (n={len(t4s)}, best_of=4, PRIOR-EXTEND): "
          f"{npass}/{len(t4s)} = {npass/len(t4s):.2f}  stage_fail={stage_fail}", flush=True)
