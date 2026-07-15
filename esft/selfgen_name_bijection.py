"""Reversible mock-name projection for self-generated tool-call seeds.

Generation uses deliberately plain names that are easier for the base model to
continue across turns.  This module records the inverse projection so accepted
traces can be returned to their diverse training namespace without changing
requests, values, or trace structure.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Any

import selfgen_toolcall_intent_v1 as intent


VERBS = ["inspect", "list", "reserve", "update", "search", "check", "fetch", "apply"]
_DERIVED_NAMES = {"first_receipt", "second_receipt", "prior_receipt", "recovery_code"}
_DERIVED_SOURCE = re.compile(r"stage(\d+)\.call(\d+)\.(result|error)\.([A-Za-z0-9_]+)")
_ORIGINAL_REQUESTS = "_bijection_original_requests"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plan(seed: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    stages = seed.get("expected_stages")
    _require(isinstance(stages, list), "seed is missing expected_stages")
    out = []
    for stage_no, stage in enumerate(stages):
        _require(isinstance(stage, list), f"expected_stages[{stage_no}] is not a list")
        for call in stage:
            _require(isinstance(call, dict) and isinstance(call.get("name"), str),
                     f"invalid expected call at stage {stage_no}")
            out.append((stage_no, call))
    return out


def _source_name(name: str, candidates: list[str]) -> str | None:
    """Return the unique original tool whose spelling contains this distractor."""
    matches = [candidate for candidate in candidates
               if name.startswith(candidate) or candidate in name]
    if len(matches) > 1:
        raise ValueError(f"ambiguous distractor source for {name!r}: {matches!r}")
    return matches[0] if matches else None


def _derived_renames(seed: dict[str, Any]) -> dict[tuple[int, str], str]:
    per_stage: dict[int, list[dict[str, Any]]] = {}
    for item in seed.get("derived_values", []):
        _require(isinstance(item, dict), "derived_values contains a non-object")
        stage, field, source = item.get("stage"), item.get("field"), item.get("source")
        _require(isinstance(stage, int) and isinstance(field, str) and isinstance(source, str),
                 f"invalid derived value: {item!r}")
        per_stage.setdefault(stage, []).append(item)
    renamed: dict[tuple[int, str], str] = {}
    for stage, items in per_stage.items():
        receipts = [item for item in items if item["source"].endswith("result.receipt")]
        if len(receipts) == 1:
            renamed[(stage, receipts[0]["field"])] = "prior_receipt"
        elif len(receipts) == 2:
            # Source call order, rather than lexical spelling, is the contract.
            receipts.sort(key=lambda item: item["source"])
            renamed[(stage, receipts[0]["field"])] = "first_receipt"
            renamed[(stage, receipts[1]["field"])] = "second_receipt"
        elif len(receipts) > 2:
            raise ValueError(f"stage {stage} has more than two receipt-derived fields")
        for item in items:
            if item["source"].endswith("error.code"):
                renamed[(stage, item["field"])] = "recovery_code"
    return renamed


def _schema(tool: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    parameters = tool.get("parameters")
    _require(isinstance(parameters, dict), f"tool {tool.get('name')!r} lacks parameters")
    props, required = parameters.get("properties"), parameters.get("required")
    _require(isinstance(props, dict) and isinstance(required, list),
             f"tool {tool.get('name')!r} has invalid parameters")
    _require(all(isinstance(key, str) for key in props) and all(isinstance(key, str) for key in required),
             f"tool {tool.get('name')!r} has non-string fields")
    _require(set(required) <= set(props), f"tool {tool.get('name')!r} required is not a property subset")
    return props, required


def recompute_derived(seed: dict[str, Any]) -> dict[str, Any]:
    """Rebuild result-derived arguments after a reversible namespace change.

    Receipts are hashes of canonical calls, so changing a tool or field spelling
    changes every downstream receipt.  Execute each gold stage only after its
    declared inputs have been filled from earlier deterministic mock results.
    """
    stages = seed.get("expected_stages")
    pattern = seed.get("pattern")
    derived_values = seed.get("derived_values", [])
    _require(isinstance(stages, list), "seed is missing expected_stages")
    _require(isinstance(pattern, str), "seed is missing pattern")
    _require(isinstance(derived_values, list), "derived_values must be a list")

    by_target_stage: dict[int, list[tuple[str, int, int, str, str]]] = {}
    for item in derived_values:
        _require(isinstance(item, dict), "derived_values contains a non-object")
        target_stage, target_field, source = item.get("stage"), item.get("field"), item.get("source")
        _require(isinstance(target_stage, int) and isinstance(target_field, str) and isinstance(source, str),
                 f"invalid derived value: {item!r}")
        match = _DERIVED_SOURCE.fullmatch(source)
        _require(match is not None, f"invalid derived source {source!r}")
        source_stage, source_call = int(match.group(1)), int(match.group(2))
        by_target_stage.setdefault(target_stage, []).append(
            (target_field, source_stage, source_call, match.group(3), match.group(4)))

    stage_results: list[list[dict[str, Any]]] = []
    for stage_no, stage in enumerate(stages):
        _require(isinstance(stage, list), f"expected_stages[{stage_no}] is not a list")
        for field, source_stage, source_call, result_kind, result_field in by_target_stage.get(stage_no, []):
            _require(source_stage < stage_no,
                     f"derived source for stage {stage_no} must refer to an earlier stage")
            _require(source_stage < len(stage_results) and source_call < len(stage_results[source_stage]),
                     f"derived source stage{source_stage}.call{source_call} is unavailable")
            source_result = stage_results[source_stage][source_call]
            source_object = source_result.get(result_kind)
            _require(isinstance(source_object, dict) and result_field in source_object,
                     f"derived source lacks {result_kind}.{result_field}: {source_result!r}")
            matched_calls = [call for call in stage
                             if isinstance(call, dict) and isinstance(call.get("arguments"), dict)
                             and field in call["arguments"]]
            _require(len(matched_calls) == 1,
                     f"derived field {field!r} in stage {stage_no} must occur in exactly one call")
            matched_calls[0]["arguments"][field] = source_object[result_field]
        stage_results.append([intent.mock_execute(call, stage_no, pattern) for call in stage])
    return seed


def _call_verb(call: dict[str, Any]) -> str:
    name = call.get("name")
    _require(isinstance(name, str), "call has no name")
    parts = name.rsplit("_", 2)
    _require(len(parts) == 3 and parts[0].startswith("mock_") and parts[2].isdigit(),
             f"call does not have a mock generation name: {name!r}")
    return parts[1]


def _call_values(seed: dict[str, Any], stage_no: int, call: dict[str, Any],
                 derived: set[tuple[int, str]]) -> str:
    name, arguments = call.get("name"), call.get("arguments")
    _require(isinstance(name, str) and isinstance(arguments, dict), "invalid expected call")
    tool = next((tool for tool in seed.get("tools", []) if tool.get("name") == name), None)
    _require(isinstance(tool, dict), f"expected tool {name!r} is absent")
    props, _ = _schema(tool)
    values = [intent._literal(arguments[field]) for field in props
              if field in arguments and (stage_no, field) not in derived]
    return ", ".join(values) if values else "no additional values"


def generation_request(seed: dict[str, Any]) -> str:
    """Render a plain r1-style request for a mockized seed.

    Each call is anchored by the verb encoded in its mock name.  Visible values
    remain grouped per call in schema-property order; result-derived values are
    described by their provenance instead of being leaked before they exist.
    """
    stages, domain, pattern = seed.get("expected_stages"), seed.get("domain"), seed.get("pattern")
    _require(isinstance(stages, list) and isinstance(domain, str) and isinstance(pattern, str),
             "seed lacks generation-request fields")
    derived = {(item["stage"], item["field"]) for item in seed.get("derived_values", [])
               if isinstance(item, dict) and isinstance(item.get("stage"), int)
               and isinstance(item.get("field"), str)}

    def step(stage_no: int, call: dict[str, Any]) -> str:
        return f"{_call_verb(call)} with {_call_values(seed, stage_no, call, derived)}"

    _require(all(isinstance(stage, list) for stage in stages), "expected_stages contains a non-list stage")
    if pattern == "single":
        _require(len(stages) == 1 and len(stages[0]) == 1, "single pattern has invalid stages")
        return f"I need to process a {domain} request. Please {step(0, stages[0][0])}."
    if pattern == "parallel":
        _require(len(stages) == 1 and len(stages[0]) == 2, "parallel pattern has invalid stages")
        return (f"I need to process a {domain} request. First, please run two things at the same time: "
                f"{step(0, stages[0][0])}; and {step(0, stages[0][1])}. Return both results.")
    if pattern == "multi_turn":
        _require(len(stages) == 2 and all(len(stage) == 1 for stage in stages),
                 "multi_turn pattern has invalid stages")
        return (f"I need to process a {domain} request. First, please {step(0, stages[0][0])}. "
                f"Once that comes back, {step(1, stages[1][0])}, using the receipt returned by the first step.")
    if pattern == "error_recovery":
        _require(len(stages) == 2 and all(len(stage) == 1 for stage in stages),
                 "error_recovery pattern has invalid stages")
        return (f"I need to process a {domain} request. First, please {step(0, stages[0][0])}. "
                f"If that reports an error, recover by running {step(1, stages[1][0])}, "
                "passing the reported error code.")
    if pattern == "long_chain":
        _require(len(stages) == 4 and len(stages[0]) == 2 and all(len(stage) == 1 for stage in stages[1:]),
                 "long_chain pattern has invalid stages")
        return (f"I need to process a {domain} request. First, please run two things at the same time: "
                f"{step(0, stages[0][0])}; and {step(0, stages[0][1])}. Once both of those come back, "
                f"go ahead and {step(1, stages[1][0])}, setting the receipt fields to the two receipts "
                f"that were returned. After that, {step(2, stages[2][0])}, using the receipt from the "
                f"previous step. If that last step reports an error, recover by running "
                f"{step(3, stages[3][0])}, passing the reported error code and the earlier receipt.")
    raise ValueError(f"unknown pattern {pattern!r}")


def mockize_seed(seed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make a seed generation-friendly and return its complete inverse map."""
    out = copy.deepcopy(seed)
    _require(isinstance(out.get("domain"), str) and out["domain"], "seed is missing domain")
    tools = out.get("tools")
    _require(isinstance(tools, list), "seed is missing tools")
    tools_by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        _require(isinstance(tool, dict) and isinstance(tool.get("name"), str), "invalid tool schema")
        _require(tool["name"] not in tools_by_name, f"duplicate tool name {tool['name']!r}")
        _schema(tool)
        tools_by_name[tool["name"]] = tool

    planned = _plan(out)
    planned_names: list[str] = []
    for _, call in planned:
        _require(call["name"] in tools_by_name, f"expected tool {call['name']!r} is absent")
        if call["name"] not in planned_names:
            planned_names.append(call["name"])
    # A tool may recur in a plan, but its first plan appearance owns its ordinal.
    primary_names = list(planned_names)
    unplanned = [name for name in tools_by_name if name not in planned_names]
    # Most distractors are spelling variants of a planned tool.  Some source data
    # also contains an unplanned primary plus its variant; retain that pair rather
    # than silently losing either schema.
    unresolved = list(unplanned)
    while unresolved:
        name = unresolved.pop(0)
        source = _source_name(name, primary_names)
        if source is None:
            primary_names.append(name)
        # Otherwise it becomes that primary's assistant variant below.

    tool_new: dict[str, str] = {}
    for ordinal, old in enumerate(primary_names, 1):
        tool_new[old] = f"mock_{out['domain']}_001_{VERBS[(ordinal - 1) % len(VERBS)]}_{ordinal}"
    assistant_count: dict[str, int] = {name: 0 for name in primary_names}
    for old in tools_by_name:
        if old in tool_new:
            continue
        source = _source_name(old, primary_names)
        _require(source is not None, f"cannot identify distractor source for {old!r}")
        assistant_count[source] += 1
        tool_new[old] = f"{tool_new[source]}_assistant_{assistant_count[source]}"
    _require(len(set(tool_new.values())) == len(tool_new), "mock tool names are not unique")

    derived = _derived_renames(out)
    plan_stage_by_tool: dict[str, int] = {}
    for stage, call in planned:
        if call["name"] in plan_stage_by_tool and plan_stage_by_tool[call["name"]] != stage:
            raise ValueError(f"tool {call['name']!r} is reused across stages; field map is ambiguous")
        plan_stage_by_tool[call["name"]] = stage

    fields_old_to_new: dict[str, dict[str, str]] = {}
    for ordinal, old in enumerate(primary_names, 1):
        props, _ = _schema(tools_by_name[old])
        stage = plan_stage_by_tool.get(old)
        field_map: dict[str, str] = {}
        visible = 0
        for field in props:
            new = derived.get((stage, field)) if stage is not None else None
            if new is None:
                visible += 1
                new = f"field_{ordinal}_{visible}"
            _require(new not in field_map.values(), f"duplicate mock field {new!r} for {old!r}")
            field_map[field] = new
        fields_old_to_new[old] = field_map

    for old in tools_by_name:
        if old in fields_old_to_new:
            continue
        source = _source_name(old, primary_names)
        _require(source is not None, f"cannot identify field source for distractor {old!r}")
        source_fields = fields_old_to_new[source]
        props, _ = _schema(tools_by_name[old])
        local: dict[str, str] = {}
        for field in props:
            core = field[:-8] if field.endswith("_routing") else field
            _require(core in source_fields, f"distractor field {field!r} has no source field in {source!r}")
            local[field] = source_fields[core] + ("_routing" if field.endswith("_routing") else "")
        _require(len(set(local.values())) == len(local), f"duplicate mock fields for {old!r}")
        fields_old_to_new[old] = local

    mapping = {"tools": {}, "fields": {}, "pattern": out["pattern"],
               "derived_values": copy.deepcopy(out.get("derived_values", [])),
               "original_requests": {key: copy.deepcopy(out[key])
                                     for key in ("user_request", "natural_request", "transcription_request")
                                     if key in out}}
    for tool in tools:
        old = tool["name"]
        props, required = _schema(tool)
        fields = fields_old_to_new[old]
        _require(set(fields) == set(props), f"incomplete field map for {old!r}")
        tool["name"] = tool_new[old]
        tool["parameters"]["properties"] = {fields[field]: value for field, value in props.items()}
        tool["parameters"]["required"] = [fields[field] for field in required]
        mapping["tools"][tool_new[old]] = old
        mapping["fields"][tool_new[old]] = {new: field for field, new in fields.items()}

    for stage, calls in enumerate(out["expected_stages"]):
        for call in calls:
            old = call["name"]
            _require(old in tool_new, f"expected tool {old!r} is absent from map")
            fields = fields_old_to_new[old]
            _require(set(call.get("arguments", {})) <= set(fields), f"unknown argument on {old!r}")
            call["name"] = tool_new[old]
            call["arguments"] = {fields[field]: value for field, value in call["arguments"].items()}
    for item in out.get("derived_values", []):
        key = (item["stage"], item["field"])
        _require(key in derived, f"derived value has no rename: {item!r}")
        item["field"] = derived[key]
    # The separately stored distractor list contains distinct objects after JSON
    # loading, so keep it consistent with tools when present.
    if isinstance(out.get("distractor_tools"), list):
        by_old = {original: mock for mock, original in mapping["tools"].items()}
        for tool in out["distractor_tools"]:
            _require(isinstance(tool, dict) and isinstance(tool.get("name"), str), "invalid distractor tool")
            old = tool["name"]
            _require(old in by_old, f"distractor {old!r} is absent from tools")
            # Reuse the already-mutated canonical schema to guarantee exact parity.
            canonical_tool = next(item for item in out["tools"] if item["name"] == by_old[old])
            tool.clear()
            tool.update(copy.deepcopy(canonical_tool))
    recompute_derived(out)
    return out, mapping


def _inverse(mapping: dict[str, Any], mock_tool: str) -> tuple[str, dict[str, str]]:
    tools, fields = mapping.get("tools"), mapping.get("fields")
    _require(isinstance(tools, dict) and isinstance(fields, dict), "mapping must contain tools and fields")
    _require(mock_tool in tools and mock_tool in fields, f"unknown mock tool {mock_tool!r}")
    _require(isinstance(tools[mock_tool], str) and isinstance(fields[mock_tool], dict),
             f"invalid mapping entry for {mock_tool!r}")
    return tools[mock_tool], fields[mock_tool]


def _restore_call(call: dict[str, Any], mapping: dict[str, Any]) -> None:
    _require(isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict), "invalid call")
    old_name, fields = _inverse(mapping, call["name"])
    _require(set(call["arguments"]) <= set(fields), f"unknown mock argument for {call['name']!r}")
    call["name"] = old_name
    call["arguments"] = {fields[field]: value for field, value in call["arguments"].items()}


def _restore_schema(tool: dict[str, Any], mapping: dict[str, Any]) -> None:
    _require(isinstance(tool.get("name"), str), "invalid tool schema")
    old_name, fields = _inverse(mapping, tool["name"])
    props, required = _schema(tool)
    _require(set(props) <= set(fields) and set(required) <= set(fields),
             f"unknown mock schema field for {tool['name']!r}")
    tool["name"] = old_name
    tool["parameters"]["properties"] = {fields[field]: value for field, value in props.items()}
    tool["parameters"]["required"] = [fields[field] for field in required]


def _recompute_rendered_messages(messages: list[dict[str, Any]], mapping: dict[str, Any]) -> None:
    """Restore receipts in an already-rendered, accepted gold conversation."""
    pattern, derived_values = mapping.get("pattern"), mapping.get("derived_values", [])
    if not isinstance(pattern, str) or not isinstance(derived_values, list):
        return
    stages: list[list[dict[str, Any]]] = []
    assistant_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant" or "tool_calls" not in message:
            continue
        calls = []
        for tool_call in message["tool_calls"]:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            _require(isinstance(function, dict) and isinstance(function.get("name"), str),
                     "invalid assistant tool_call")
            arguments = function.get("arguments")
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
            _require(isinstance(decoded, dict), "assistant arguments must be a JSON object")
            calls.append({"name": function["name"], "arguments": decoded})
        stages.append(calls)
        assistant_messages.append(message)
    if not stages:
        return
    projection = {"expected_stages": stages, "derived_values": copy.deepcopy(derived_values),
                  "pattern": pattern}
    recompute_derived(projection)
    results_by_call: dict[tuple[int, int], dict[str, Any]] = {}
    for stage_no, calls in enumerate(stages):
        for call_no, call in enumerate(calls):
            results_by_call[(stage_no, call_no)] = intent.mock_execute(call, stage_no, pattern)
            function = assistant_messages[stage_no]["tool_calls"][call_no]["function"]
            original_arguments = function["arguments"]
            function["arguments"] = (json.dumps(call["arguments"], separators=(",", ":"), sort_keys=True)
                                     if isinstance(original_arguments, str) else copy.deepcopy(call["arguments"]))
    stage_no, call_no = -1, 0
    for message in messages:
        if message.get("role") == "assistant" and "tool_calls" in message:
            stage_no, call_no = stage_no + 1, 0
        elif message.get("role") == "tool" and stage_no >= 0:
            result = results_by_call.get((stage_no, call_no))
            _require(result is not None, "tool message has no matching assistant call")
            message["content"] = json.dumps(result, separators=(",", ":"), sort_keys=True)
            call_no += 1


def restore_record(record: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    """Restore a checkpoint record or training row without changing non-name data."""
    out = copy.deepcopy(record)
    _require(isinstance(out, dict), "record must be an object")
    if isinstance(out.get("seed"), dict):
        out["seed"] = restore_record(out["seed"], mapping)
    if isinstance(out.get("tools"), list):
        for tool in out["tools"]:
            _restore_schema(tool, mapping)
    if isinstance(out.get("distractor_tools"), list):
        for tool in out["distractor_tools"]:
            _restore_schema(tool, mapping)
    # A derived field belongs to a particular expected-stage schema.  Resolve it
    # through that schema rather than globally: standard names such as
    # ``prior_receipt`` intentionally occur in more than one tool.
    stage_field_maps: dict[int, dict[str, str]] = {}
    if isinstance(out.get("expected_stages"), list):
        for stage_no, stage in enumerate(out["expected_stages"]):
            _require(isinstance(stage, list), "expected_stages contains a non-list stage")
            for call in stage:
                _require(isinstance(call, dict) and isinstance(call.get("name"), str), "invalid expected call")
                _, fields = _inverse(mapping, call["name"])
                stage_field_maps.setdefault(stage_no, {}).update(fields)
    if isinstance(out.get("derived_values"), list):
        for item in out["derived_values"]:
            _require(isinstance(item, dict) and isinstance(item.get("stage"), int)
                     and isinstance(item.get("field"), str), "invalid derived value")
            fields = stage_field_maps.get(item["stage"])
            _require(fields is not None and item["field"] in fields,
                     f"derived field has no stage schema: {item!r}")
            item["field"] = fields[item["field"]]
    for key in ("expected_stages", "selected"):
        if key in out:
            _require(isinstance(out[key], list), f"{key} must be a list")
            for stage in out[key]:
                _require(isinstance(stage, list), f"{key} contains a non-list stage")
                for call in stage:
                    _restore_call(call, mapping)
    if isinstance(out.get("expected_stages"), list) and isinstance(out.get("pattern"), str):
        recompute_derived(out)
    if isinstance(out.get("calls"), list):
        for call in out["calls"]:
            _restore_call(call, mapping)
    if isinstance(out.get("results"), list):
        for stage in out["results"]:
            _require(isinstance(stage, list), "results contains a non-list stage")
            for result in stage:
                if isinstance(result, dict) and isinstance(result.get("call"), str):
                    result["call"] = _inverse(mapping, result["call"])[0]
        if isinstance(out.get("expected_stages"), list) and isinstance(out.get("pattern"), str):
            _require(len(out["results"]) == len(out["expected_stages"]),
                     "results and expected_stages have different lengths")
            out["results"] = [[intent.mock_execute(call, stage_no, out["pattern"])
                               for call in stage]
                              for stage_no, stage in enumerate(out["expected_stages"])]
    if isinstance(out.get("messages"), list):
        for message in out["messages"]:
            _require(isinstance(message, dict), "message is not an object")
            if message.get("role") == "assistant" and "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function") if isinstance(tool_call, dict) else None
                    _require(isinstance(function, dict) and isinstance(function.get("name"), str),
                             "invalid assistant tool_call")
                    old_name, fields = _inverse(mapping, function["name"])
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        decoded = json.loads(arguments)
                        _require(isinstance(decoded, dict) and set(decoded) <= set(fields),
                                 f"unknown mock argument for {function['name']!r}")
                        function["arguments"] = json.dumps({fields[key]: value for key, value in decoded.items()},
                                                           separators=(",", ":"), sort_keys=True)
                    else:
                        _require(isinstance(arguments, dict) and set(arguments) <= set(fields),
                                 "assistant arguments must be an object or JSON object")
                        function["arguments"] = {fields[key]: value for key, value in arguments.items()}
                    function["name"] = old_name
            elif message.get("role") == "tool" and isinstance(message.get("name"), str):
                message["name"] = _inverse(mapping, message["name"])[0]
                if isinstance(message.get("content"), str):
                    result = json.loads(message["content"])
                    if isinstance(result, dict) and isinstance(result.get("call"), str):
                        result["call"] = _inverse(mapping, result["call"])[0]
                    message["content"] = json.dumps(result, separators=(",", ":"), sort_keys=True)
        _recompute_rendered_messages(out["messages"], mapping)
    originals = out.pop(_ORIGINAL_REQUESTS, mapping.get("original_requests"))
    if originals is not None:
        _require(isinstance(originals, dict) and all(isinstance(key, str) for key in originals),
                 "invalid preserved request fields")
        if "user_request" in out:
            out.update(originals)
        elif isinstance(out.get("messages"), list) and isinstance(originals.get("user_request"), str):
            for message in out["messages"]:
                if message.get("role") == "user":
                    message["content"] = originals["user_request"]
                    break
    return out


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _cmd_mockize(input_path: Path, output_dir: Path) -> None:
    data = _load_json(input_path)
    _require(isinstance(data, dict) and isinstance(data.get("seeds"), list), "input must be {'seeds': [...]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds, maps = [], {}
    for seed in data["seeds"]:
        mocked, mapping = mockize_seed(seed)
        _require(_ORIGINAL_REQUESTS not in mocked, "seed already contains preserved request fields")
        mocked[_ORIGINAL_REQUESTS] = {key: copy.deepcopy(mocked[key])
                                      for key in ("user_request", "natural_request", "transcription_request")
                                      if key in mocked}
        mocked["user_request"] = generation_request(mocked)
        seed_id = mocked.get("seed_id")
        _require(isinstance(seed_id, str) and seed_id not in maps, "missing or duplicate seed_id")
        seeds.append(mocked)
        maps[seed_id] = mapping
    with (output_dir / "seeds.json").open("w", encoding="utf-8") as handle:
        json.dump({**data, "seeds": seeds}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "bijection_maps.json").open("w", encoding="utf-8") as handle:
        json.dump(maps, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _cmd_restore(records_path: Path, maps_path: Path, output_path: Path) -> None:
    maps = _load_json(maps_path)
    _require(isinstance(maps, dict), "bijection maps must be an object")
    with records_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(source, 1):
            _require(line.strip(), f"blank JSONL line {line_no}")
            record = json.loads(line)
            seed = record.get("seed") if isinstance(record, dict) else None
            metadata = record.get("metadata") if isinstance(record, dict) else None
            container = seed if isinstance(seed, dict) else record
            seed_id = container.get("seed_id") if isinstance(container, dict) else None
            # render_training() identifies rows through metadata.id rather than
            # repeating the private seed object.
            if seed_id is None and isinstance(metadata, dict):
                seed_id = metadata.get("id")
            _require(isinstance(seed_id, str) and seed_id in maps, f"line {line_no} has no known seed_id")
            target.write(json.dumps(restore_record(record, maps[seed_id]), ensure_ascii=False,
                                    separators=(",", ":"), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mock = sub.add_parser("mockize")
    mock.add_argument("input_seeds")
    mock.add_argument("output_dir")
    restore = sub.add_parser("restore")
    restore.add_argument("records")
    restore.add_argument("bijection_maps")
    restore.add_argument("output")
    args = parser.parse_args()
    if args.command == "mockize":
        _cmd_mockize(Path(args.input_seeds), Path(args.output_dir))
    else:
        _cmd_restore(Path(args.records), Path(args.bijection_maps), Path(args.output))


if __name__ == "__main__":
    main()
