#!/usr/bin/env python3
"""Pilot gate for diverse-name paraphrases: literal fidelity + name-leak check.

The legacy driver leak check only knows the mock_/field_ prefixes; diverse-name
runs must instead reject any internal tool or parameter name from the seed's own
schemas. Reads seeds.json for the run, applies both gates to a paraphrase JSONL
({seed_id, paraphrase} or {seed_id, natural_request}), prints a summary.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent / "data" / "selfgen_toolcall_intent_v1"


def seed_index(run_id: str) -> dict[str, dict]:
    data = json.loads((RUN_ROOT / run_id / "seeds.json").read_text())
    return {s["seed_id"]: s for s in data["seeds"]}


def internal_names(seed: dict) -> set[str]:
    names: set[str] = set()
    for tool in seed.get("tools", []):
        names.add(tool["name"])
        names.update(tool.get("parameters", {}).get("properties", {}).keys())
    return {n for n in names if len(n) >= 4}


def batch_literals(run_id: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(RUN_ROOT / run_id / "paraphrase_batch.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["seed_id"]] = row["value_literals"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--paraphrases", required=True)
    args = ap.parse_args()

    seeds = seed_index(args.run_id)
    literals = batch_literals(args.run_id)
    n = ok = miss_lit = leak = 0
    leak_examples: list[str] = []
    with open(args.paraphrases) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row["seed_id"]
            text = row.get("paraphrase") or row.get("natural_request") or ""
            n += 1
            missing = [lit for lit in literals[sid] if lit not in text]
            leaked = sorted(name for name in internal_names(seeds[sid]) if name in text)
            if missing:
                miss_lit += 1
            if leaked:
                leak += 1
                if len(leak_examples) < 5:
                    leak_examples.append(f"{sid}: {leaked[:3]}")
            if not missing and not leaked:
                ok += 1
    print(json.dumps({"rows": n, "ok": ok, "ok_rate": round(ok / max(n, 1), 4),
                      "literal_missing": miss_lit, "name_leak": leak,
                      "leak_examples": leak_examples}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
