#!/usr/bin/env python3
"""Stream Toucan/selfgen tool-call records into the trainer's message JSONL.

The default output is compatible with the legacy incremental trainer: it puts
Qwen's tool-schema text at the start of the first user message.  With
``--tools-mode native`` it instead preserves normal system messages and stores
the schema in the record's native ``tools`` field for the trainer's offsets
tokenisation mode.  It also normalises the four observed Toucan representations
to Qwen's native ``assistant.tool_calls`` and ``tool`` roles.

This program is CPU/IO only.  Parquet is read batch by batch; it never reads a
whole file or corpus into memory.  A parquet shard still being written is not
read unless a matching completion marker exists or it is older than the
configured grace period.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


TOOL_INSTRUCTION_SUFFIX = """\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


class ConversionError(ValueError):
    """An input row cannot be safely represented by the Qwen template."""


def parse_json(value: Any, *, want: type | tuple[type, ...] | None = None) -> Any:
    """Decode the JSON-in-string columns, including an occasional second layer."""
    result = value
    for _ in range(2):
        if not isinstance(result, str):
            break
        stripped = result.strip()
        if not stripped or stripped[0] not in "[{":
            break
        try:
            result = json.loads(stripped)
        except json.JSONDecodeError:
            break
    if want is not None and not isinstance(result, want):
        raise ConversionError(f"expected {want}, got {type(result).__name__}")
    return result


def _arguments(value: Any) -> dict[str, Any]:
    value = parse_json(value)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConversionError("tool-call arguments are not a JSON object")
    return value


def normalize_tool_call(value: Any) -> dict[str, Any]:
    """Return the OpenAI-style shape required by this snapshot's template."""
    value = parse_json(value, want=dict)
    function = value.get("function", value)
    if not isinstance(function, dict) or not function.get("name"):
        raise ConversionError("tool call has no function name")
    return {
        "type": "function",
        "function": {
            "name": str(function["name"]),
            "arguments": _arguments(function.get("arguments", {})),
        },
    }


def _calls_from_message(message: dict[str, Any], *, content_fallback: bool = False) -> list[dict[str, Any]]:
    raw = message.get("tool_calls")
    if raw is not None:
        raw = parse_json(raw, want=list)
        return [normalize_tool_call(call) for call in raw]
    for key in ("function_call", "function", "tool_call"):
        if message.get(key) is not None:
            return [normalize_tool_call(message[key])]
    if not content_fallback:
        return []
    content = message.get("content")
    parsed = parse_json(content)
    if isinstance(parsed, list):
        return [normalize_tool_call(call) for call in parsed]
    if isinstance(parsed, dict):
        return [normalize_tool_call(parsed)]
    return []


def _tool_system_content(tools: list[dict[str, Any]]) -> str:
    """Exact content emitted by the snapshot's ``if tools`` template branch.

    The surrounding ``<|im_start|>system`` / ``<|im_end|>`` is intentionally
    omitted: the normal message branch supplies it after this string is made a
    real leading system message.  ``ensure_ascii=False`` matches Jinja's
    observed ``tojson`` output under this tokenizer/template environment.
    """
    rendered = "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in tools)
    return "# Tools\n\nYou have access to the following functions:\n\n<tools>" + rendered + "\n</tools>" + TOOL_INSTRUCTION_SUFFIX


def _legacy_tool_declaration(content: str) -> list[dict[str, Any]] | None:
    """Extract Toucan's embedded legacy system ``tool_declare`` list, if any."""
    prefix = "<|im_system|>tool_declare<|im_middle|>"
    suffix = "<|im_end|>"
    if not isinstance(content, str) or not content.startswith(prefix):
        return None
    payload = content[len(prefix):]
    if payload.endswith(suffix):
        payload = payload[:-len(suffix)]
    try:
        return parse_json(payload, want=list)
    except ConversionError:
        return None


def _normalize_messages_and_tools(raw_messages: Any, raw_tools: Any = None, *,
                                  tools_mode: str = "preamble") -> tuple[list[dict[str, Any]], Any]:
    """Normalise selfgen, Toucan legacy, and Toucan SFT messages.

    The output contains only roles accepted by the Qwen 3.6 chat template.
    ``preamble`` retains the legacy behavior: a leading system message and tool
    schemas are folded into the first user turn because the incremental trainer
    renders system-only prefixes.  ``native`` keeps a normal system message and
    returns the tool schemas separately for the trainer to pass as ``tools=``.
    """
    if tools_mode not in {"preamble", "native"}:
        raise ValueError(f"unknown tools mode: {tools_mode!r}")
    raw_messages = parse_json(raw_messages, want=list)
    tools = parse_json(raw_tools) if raw_tools is not None else None
    if tools is not None and not isinstance(tools, list):
        raise ConversionError("tools column is not a JSON list")

    source_system: str | None = None
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ConversionError(f"message {index} is not an object")
        role = raw.get("role")
        content = raw.get("content")
        content = "" if content is None else str(content)
        if role == "system":
            legacy_tools = _legacy_tool_declaration(content)
            if legacy_tools is not None:
                if tools is None:
                    tools = legacy_tools
                continue
            if source_system is not None:
                raise ConversionError("multiple non-tool system messages")
            source_system = content
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            out = {"role": "assistant", "content": content}
            calls = _calls_from_message(raw)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
        elif role in ("tool", "function", "tool_response"):
            out = {"role": "tool", "content": content}
            if raw.get("name") is not None:
                out["name"] = str(raw["name"])
            messages.append(out)
        elif role == "tool_call":
            calls = _calls_from_message(raw, content_fallback=True)
            if not calls:
                raise ConversionError("tool_call message contains no call")
            # Some SFT rows carry a natural-language preface in content; retain
            # it only when it was not the JSON object used to encode the call.
            parsed_content = parse_json(content)
            preface = content if not isinstance(parsed_content, (dict, list)) else ""
            messages.append({"role": "assistant", "content": preface, "tool_calls": calls})
        else:
            raise ConversionError(f"unsupported message role {role!r}")

    if tools_mode == "preamble":
        preamble_parts = []
        if tools:
            preamble_parts.append(_tool_system_content(tools))
        if source_system and source_system.strip():
            preamble_parts.append(source_system.strip())
        if preamble_parts:
            first_user = next((message for message in messages if message["role"] == "user"), None)
            if first_user is None:
                raise ConversionError("tool/system context has no following user turn")
            first_user["content"] = "\n\n".join(preamble_parts) + "\n\n" + first_user["content"]
    elif source_system and source_system.strip():
        messages.insert(0, {"role": "system", "content": source_system})
    if not messages or not any(msg["role"] == "assistant" for msg in messages):
        raise ConversionError("record has no assistant turn")
    return messages, tools


def normalize_messages(raw_messages: Any, raw_tools: Any = None, *,
                       tools_mode: str = "preamble") -> list[dict[str, Any]]:
    """Public message-only normaliser retained for callers and legacy tests."""
    messages, _ = _normalize_messages_and_tools(
        raw_messages,
        raw_tools,
        tools_mode=tools_mode,
    )
    return messages


def to_trainer_record(row: dict[str, Any], *, source_tag: str, domain: str,
                      tools_mode: str = "preamble") -> dict[str, Any]:
    """Convert one source row, optionally retaining native tool schemas."""
    raw_tools = row.get("tools", row.get("available_tools"))
    messages, tools = _normalize_messages_and_tools(
        row.get("messages"), raw_tools, tools_mode=tools_mode)
    record = {
        "messages": messages,
        "_source": source_tag,
        "_domain": domain,
    }
    if tools_mode == "native" and tools is not None:
        record["tools"] = tools
    return record


def parquet_is_complete(path: Path, min_age_seconds: float) -> bool:
    """Permit a finished shard marker, otherwise require a stable old mtime."""
    markers = (Path(str(path) + ".done"), path.with_suffix(".done"))
    if any(marker.is_file() for marker in markers):
        return True
    # A manifest is only a completion signal when it lives next to the shard.
    if (path.parent / "manifest.json").is_file():
        return True
    return time.time() - path.stat().st_mtime >= min_age_seconds


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConversionError(f"{path}:{line_no}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ConversionError(f"{path}:{line_no}: row is not an object")
                yield row


def input_files(inputs: Iterable[str], min_age_seconds: float) -> Iterator[Path]:
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.parquet"))
        else:
            raise FileNotFoundError(path)
        for candidate in candidates:
            if candidate.suffix == ".parquet" and not parquet_is_complete(candidate, min_age_seconds):
                print(f"SKIP active parquet: {candidate}", file=sys.stderr)
                continue
            yield candidate


def convert(inputs: Iterable[str], output: str, *, source_tag: str, domain: str,
            batch_size: int = 128, min_parquet_age_seconds: float = 300,
            max_records: int = 0, tools_mode: str = "preamble") -> tuple[int, int]:
    """Stream all usable input rows to ``output``.  Returns ``(written, bad)``."""
    written = bad = 0
    with Path(output).open("w", encoding="utf-8") as handle:
        for path in input_files(inputs, min_parquet_age_seconds):
            iterator = iter_parquet_rows(path, batch_size) if path.suffix == ".parquet" else iter_jsonl_rows(path)
            for row_no, row in enumerate(iterator):
                try:
                    record = to_trainer_record(row, source_tag=source_tag, domain=domain,
                                               tools_mode=tools_mode)
                except (ConversionError, TypeError, ValueError) as exc:
                    bad += 1
                    print(f"SKIP {path}:{row_no}: {exc}", file=sys.stderr)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if max_records and written >= max_records:
                    return written, bad
    return written, bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="selfgen JSONL, Toucan parquet, or a directory containing either (repeatable)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-tag", required=True,
                        help="stored in _source on every emitted trainer record")
    parser.add_argument("--domain", default="toolcall", help="stored in _domain (default: toolcall)")
    parser.add_argument("--tools-mode", choices=["preamble", "native"], default="preamble",
                        help="preamble preserves legacy incremental training; native emits "
                             "record.tools for trainer --tokenize-mode offsets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-parquet-age-seconds", type=float, default=300)
    parser.add_argument("--max-records", type=int, default=0, help="test/smoke cap; 0 streams all rows")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.min_parquet_age_seconds < 0 or args.max_records < 0:
        parser.error("batch size must be positive; ages and caps must be non-negative")
    written, bad = convert(args.input, args.output, source_tag=args.source_tag, domain=args.domain,
                           batch_size=args.batch_size,
                           min_parquet_age_seconds=args.min_parquet_age_seconds,
                           max_records=args.max_records,
                           tools_mode=args.tools_mode)
    print(json.dumps({"written": written, "skipped_invalid": bad}, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
