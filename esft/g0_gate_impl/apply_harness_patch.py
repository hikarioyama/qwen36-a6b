#!/usr/bin/env python
"""Idempotently wire gate_shaping into ~/esft/eval_harness.py.

Inserts a call to ``gate_shaping.maybe_install_from_env(refs, gpu_id)`` inside
``load_subject_model`` -- the one function that runs in every spawned GPU worker,
AFTER the top-k override and BEFORE the patch load. Running it there (rather than
monkeypatching from a parent runner) is required because the harness uses
mp.spawn: each child re-imports eval_harness fresh, so the wiring must live in the
module source, not in a parent-process monkeypatch.

Env vars (G0_SHAPE/G0_PARAM/G0_DEBUG) are inherited by spawn children, so the
per-worker call picks up the sweep's config automatically. With G0_SHAPE unset the
inserted code is a proven no-op (no hook registered).

Usage (on aux-host):  python apply_harness_patch.py [--harness ~/esft/eval_harness.py]
Prints ALREADY-PATCHED and exits 0 if the marker is already present.
"""
import argparse
import os
import shutil

MARKER = "G0 gate-shaping (training-free"

ANCHOR = '    if spec["kind"] == "patched":\n'

INSERT = (
    "    # G0 gate-shaping (training-free router reweighting; no-op unless "
    "G0_SHAPE set).\n"
    "    if is_moe:\n"
    "        try:\n"
    "            import gate_shaping\n"
    "            gate_shaping.maybe_install_from_env(refs, gpu_id=gpu_id)\n"
    "        except Exception as _g0e:\n"
    '            print(f"[gpu{gpu_id}] gate_shaping unavailable: {_g0e}", '
    "flush=True)\n"
    "\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness",
                    default=os.path.expanduser("~/esft/eval_harness.py"))
    args = ap.parse_args()

    with open(args.harness) as f:
        src = f.read()

    if MARKER in src:
        print(f"ALREADY-PATCHED: {args.harness}")
        return

    if ANCHOR not in src:
        raise SystemExit(
            f"anchor not found in {args.harness!r}; expected line:\n{ANCHOR!r}\n"
            "load_subject_model may have changed -- inspect and insert the G0 hook "
            "call manually after the top-k override block.")
    if src.count(ANCHOR) != 1:
        raise SystemExit(
            f"anchor is ambiguous ({src.count(ANCHOR)} matches); refusing to patch.")

    backup = args.harness + ".pre_g0.bak"
    shutil.copy2(args.harness, backup)
    patched = src.replace(ANCHOR, INSERT + ANCHOR, 1)
    with open(args.harness, "w") as f:
        f.write(patched)
    print(f"PATCHED: {args.harness}  (backup at {backup})")


if __name__ == "__main__":
    main()
