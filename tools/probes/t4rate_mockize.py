"""T4 probe: desc2 seeds を生成用に r1-parity へ全単射リネームして測る (本走行の go/no-go ゲート)。
- tool 名 → mock_<domain>_<verb>_<ordinal> (r1 と同形式)
- 可視 arg 名 → field_<toolOrdinal>_<k> (r1 と同形式)  [--keepargs で無効化]
- derived 名 → first_receipt/second_receipt/prior_receipt/recovery_code (r1 と同一)
- few-shot 継続例注入、request は natural のまま (paraphrase はツール名を含まないので整合)
usage: python t4rate_mockize.py <run> <endpoint> <n> [keepargs]
"""
import copy, json, os, sys
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["SELFGEN_NOTHINK"] = "1"; os.environ["SELFGEN_STAGE_HINT"] = "1"
import selfgen_toolcall_intent_v1 as intent
import selfgen_toolcall_v1 as v1
import t4rate_fewshot as fs

VERBS = ["inspect", "list", "reserve", "update", "search", "check", "fetch", "apply"]

import re as _re


def recompute_derived(seed):
    """リネーム後の gold calls を stage 順に mock 実行し直し、derived 値 (receipt/error code) を
    焼き直す。mock_execute の receipt は call 内容のハッシュなのでリネームで必ず変わる。"""
    per_stage_results = []
    for st, stage in enumerate(seed["expected_stages"]):
        for c in stage:
            for d in seed["derived_values"]:
                if d["stage"] != st or d["field"] not in c["arguments"]:
                    continue
                m = _re.fullmatch(r"stage(\d+)\.call(\d+)\.(result|error)\.([A-Za-z0-9_]+)", d["source"])
                if not m:
                    raise ValueError(f"bad derived source: {d['source']}")
                K, J, kind, fld = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
                c["arguments"][d["field"]] = per_stage_results[K][J][kind][fld]
        per_stage_results.append([intent.mock_execute(c, st, seed["pattern"]) for c in stage])
    return seed


def mockize(seed, keep_visible_args=False):
    """生成用の r1-parity リネーム。name_map/field_maps を返すので逆写像可能。"""
    seed = copy.deepcopy(seed)
    domain = seed["domain"]
    plan = [c["name"] for st in seed["expected_stages"] for c in st]
    name_map = {}
    for i, nm in enumerate(plan):
        if nm not in name_map:
            name_map[nm] = f"mock_{domain}_001_{VERBS[i % len(VERBS)]}_{i + 1}"
    # derived field の r1-parity 名
    per_stage = {}
    for d in seed["derived_values"]:
        per_stage.setdefault(d["stage"], []).append(d)
    field_renames = {}  # (stage, old) -> new
    for st, ds in per_stage.items():
        recs = sorted([d for d in ds if d["source"].endswith("result.receipt")], key=lambda x: x["source"])
        if len(recs) >= 2:
            for k, d in enumerate(recs):
                field_renames[(st, d["field"])] = ["first_receipt", "second_receipt", "third_receipt"][k]
        elif len(recs) == 1:
            field_renames[(st, recs[0]["field"])] = "prior_receipt"
        for d in ds:
            if d["source"].endswith("error.code"):
                field_renames[(st, d["field"])] = "recovery_code"
    for d in seed["derived_values"]:
        key = (d["stage"], d["field"])
        if key in field_renames:
            d["field"] = field_renames[key]
    # 可視 arg 名 → field_<toolOrdinal>_<k>
    ord_of = {nm: i + 1 for i, nm in enumerate(plan)}
    vis_renames = {}  # tool_old_name -> {old_field: new_field}
    for st, stage in enumerate(seed["expected_stages"]):
        for c in stage:
            tool = next(t for t in seed["tools"] if t["name"] == c["name"])
            props = tool["parameters"]["properties"]
            m = {}
            k = 0
            for f in list(props.keys()):
                if (st, f) in field_renames or f in ("first_receipt", "second_receipt", "third_receipt",
                                                     "prior_receipt", "recovery_code"):
                    dr = field_renames.get((st, f))
                    if dr:
                        m[f] = dr
                    continue
                if not keep_visible_args:
                    k += 1
                    m[f] = f"field_{ord_of[c['name']]}_{k}"
            vis_renames[c["name"]] = m
    # tools (本体+distractor) へ適用
    for tool in seed["tools"]:
        base = tool["name"]
        is_distr = base not in name_map
        src = None
        if is_distr:
            for real in name_map:
                if base.startswith(real) or real in base:
                    src = real
                    break
        else:
            src = base
        m = dict(vis_renames.get(src) or {})
        # derived renames はスキーマにも適用 (stage 不問で field 名一致で引く)
        for (st, old), new in field_renames.items():
            m.setdefault(old, new)
        props = tool["parameters"]["properties"]
        for old, new in m.items():
            variants = [old, old + "_routing"]
            for ov in variants:
                if ov in props:
                    nv = new + ("_routing" if ov.endswith("_routing") else "")
                    props[nv] = props.pop(ov)
                    tool["parameters"]["required"] = [nv if r == ov else r
                                                      for r in tool["parameters"]["required"]]
        if is_distr and src:
            tool["name"] = name_map[src] + "_assistant_1"
        elif not is_distr:
            tool["name"] = name_map[base]
    # expected_stages へ適用
    for st, stage in enumerate(seed["expected_stages"]):
        for c in stage:
            m = dict(vis_renames.get(c["name"]) or {})
            for (rst, old), new in field_renames.items():
                if rst == st:
                    m.setdefault(old, new)
            c["arguments"] = {m.get(f, f): v for f, v in c["arguments"].items()}
            c["name"] = name_map[c["name"]]
    return recompute_derived(seed)


def r1_style_request(seed):
    """mockize 済み seed から r1 骨格の request を再構成する (long_chain 専用)。
    動詞 anchor + per-call 値グループ (props 順) + receipt 定型句。"""
    derived = {(d["stage"], d["field"]) for d in seed["derived_values"]}
    def verb(call):
        return call["name"].rsplit("_", 2)[-2]
    def vals(st, call):
        tool = next(t for t in seed["tools"] if t["name"] == call["name"])
        ordered = [f for f in tool["parameters"]["properties"] if (st, f) not in derived
                   and f not in ("first_receipt", "second_receipt", "prior_receipt", "recovery_code")
                   and f in call["arguments"]]
        return ", ".join(intent._literal(call["arguments"][f]) for f in ordered)
    s0 = seed["expected_stages"][0]
    s1, s2, s3 = (seed["expected_stages"][i][0] for i in (1, 2, 3))
    return (f"I need to process a {seed['domain']} request. First, please run two things at the "
            f"same time: {verb(s0[0])} with {vals(0, s0[0])}; and {verb(s0[1])} with {vals(0, s0[1])}. "
            f"Once both of those come back, go ahead and {verb(s1)} using {vals(1, s1)}, setting the "
            f"receipt fields to the two receipts that were returned. After that, {verb(s2)} with "
            f"{vals(2, s2)}, using the receipt from the previous step. If that last step reports an "
            f"error, recover by running {verb(s3)} with {vals(3, s3)}, passing the reported error "
            f"code and the earlier receipt.")


if __name__ == "__main__":
    run = sys.argv[1]; ep = sys.argv[2]; n = int(sys.argv[3])
    keep = len(sys.argv) > 4 and sys.argv[4] == "keepargs"
    r1req = len(sys.argv) > 4 and sys.argv[4] == "r1req"
    if os.environ.get("SKIP_MT_FEWSHOT") != "1":
        v1.scaffold_prompt = fs._patched_scaffold
    seeds = json.load(open(f"data/selfgen_toolcall_intent_v1/{run}/seeds.json"))["seeds"]
    cli = intent._VLLMClient(ep, model="Qwen/Qwen3.6-35B-A3B")
    spec = v1.GenerationSpec(20260714, 0.7, 2048, 4, 0)
    t4s = []
    for s in seeds:
        if s["tier"] != "T4":
            continue
        m = mockize(s, keep_visible_args=keep)
        if r1req:
            m["user_request"] = r1_style_request(m)
        t4s.append(m)
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
    tag = "MOCKIZE-KEEPARGS" if keep else ("MOCKIZE-FULL+R1REQ" if r1req else "MOCKIZE-FULL")
    print(f"{run}+{tag}+FEWSHOT+NATURAL T4 (n={len(t4s)}, best_of=4, PRIOR-EXTEND): "
          f"{npass}/{len(t4s)} = {npass/len(t4s):.2f}  stage_fail={stage_fail}", flush=True)
