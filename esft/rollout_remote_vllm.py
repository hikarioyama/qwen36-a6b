#!/usr/bin/env python3
"""Thin rejection-sampling runner against an already-serving vLLM endpoint.

``selfgen_toolcall_intent_v1.py execute`` bundles the true-stock fingerprint
check and a local two-GPU launch topology.  When the vLLM server already runs
on another host (single TP=8 endpoint), that identity check keys off a local
snapshot path that does not exist here, and the physical-GPU preflight is moot.
This runner reuses the intent module's generation worker and its mechanical
verifier (``evaluate_records``) verbatim, but drives them against one remote
endpoint.  Verification — the part that actually admits data — is unchanged.

The stock identity is instead pinned by asserting the endpoint's served model
name equals ``--served-model`` (the operator states which model vLLM loaded).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ESFT = Path(__file__).resolve().parent
sys.path.insert(0, str(ESFT))

import selfgen_toolcall_intent_v1 as intent  # noqa: E402
import selfgen_toolcall_v1 as v1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--endpoint", required=True, help="base vLLM URL, e.g. http://localhost:8199")
    ap.add_argument("--served-model", required=True, help="model name vLLM serves; asserted against /v1/models")
    ap.add_argument("--workers", type=int, default=2, choices=[2], help="parallel generation workers over the one endpoint")
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    target, manifest, frozen = intent._load_frozen_seeds(args.run_id)

    served = intent._served_model(args.endpoint)
    if served != args.served_model:
        raise SystemExit(f"endpoint serves {served!r}, expected {args.served_model!r}")

    seeds = frozen["seeds"]
    completed = intent.load_checkpoint_records(target, seeds)
    todo = intent.pending_seeds(seeds, completed, retry_failed=args.retry_failed)
    print(f"[rollout] run={args.run_id} served={served} completed={len(completed)} pending={len(todo)}",
          flush=True)

    if todo:
        nw = max(1, args.workers)
        buckets: list[list[dict]] = [[] for _ in range(nw)]
        for i, seed in enumerate(todo):
            buckets[i % nw].append(seed)
        output: mp.Queue = mp.Queue()
        workers = []
        for w in range(nw):
            spec = v1.GenerationSpec(args.seed, args.temperature, args.max_new, args.best_of, w)
            worker_args = (w, buckets[w], spec, v1.checkpoint_path(target, w), output,
                           args.endpoint, served)
            workers.append(mp.Process(target=intent.vllm_generation_worker, args=worker_args))
        for worker in workers:
            worker.start()
        try:
            v1.wait_for_worker_completion(workers, output)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=30)
            output.close()
            output.join_thread()

    records = list(intent.load_checkpoint_records(target, seeds).values())
    records.sort(key=lambda record: record["seed"]["seed_id"])
    if len(records) != len(seeds):
        raise SystemExit(f"missing generated records: {len(records)}/{len(seeds)}")

    contamination, grams, names = v1.contamination_corpus()
    accepted, rejected, reasons = intent.evaluate_records(records, grams, names)
    intent.atomic_jsonl(target / "train.jsonl", accepted)
    intent.atomic_jsonl(target / "rejected.jsonl", rejected)
    rate = len(accepted) / len(records) if records else 0.0
    print(f"[rollout] generated={len(records)} accepted={len(accepted)} "
          f"rejected={len(rejected)} rate={rate:.3f}", flush=True)
    print(f"[rollout] reasons={dict(reasons)}", flush=True)


if __name__ == "__main__":
    main()
