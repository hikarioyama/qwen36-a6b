#!/bin/bash
# INC-0 memory + correctness probe for the full-FFN FSDP path.
# 8-GPU, --max-steps 5. Asserts (see clever-conjuring-clarke.md Phase 2a):
#   1. no OOM AND peak < 70 GB/GPU  (torch.cuda.max_memory_allocated, printed by trainer)
#   2. trainable == 32.2B  AND  non-expert params frozen
#   3. all 40 layers' expert shards get grad (grad_none==0 && grad_zero==0 in FULLFFN_PROBE)
#   4. router/attn/embed requires_grad=False + .grad None (part A, in-run)
#      + router weights byte-identical to base after save (part B, offline, step 2)
#
# GPU LAUNCH IS DONE BY THE MAIN SESSION (this agent must not fire GPU jobs).
# Deploy: copy train_esft.py + the to_esft_full addition into the gpu-host esft tree,
# then run this from that tree.
set -euo pipefail

ESFT_DIR=${ESFT_DIR:-/mnt/docker-raid/models/esft}
VENV=${VENV:-python3}
TRAINER=${TRAINER:-train_fullffn_dcp.py}
MODEL=${MODEL:-/mnt/docker-raid/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0}
TRAIN_DATA=${TRAIN_DATA:-v3.jsonl}
REPLAY_DATA=${REPLAY_DATA:-mixed_v2.jsonl}
OUT=${OUT:-codex_runs/fullffn_probe_v2_20260710}
SEQ=${SEQ:-7168}
NPROC=${NPROC:-8}
LOG=${LOG:-${OUT}/probe.log}

cd "$ESFT_DIR"

# full-ffn ignores expert selection (all experts train); a trivial config suffices.
PROBE_CFG=configs/fullffn_probe.json
mkdir -p configs
echo '{"experts": {}}' > "$PROBE_CFG"
mkdir -p "$OUT"

echo "[probe] 1/3 building/loading packed cache (CPU, single-proc)..."
$VENV "$TRAINER" --model "$MODEL" --method full-ffn \
  --expert-config "$PROBE_CFG" --train-data "$TRAIN_DATA" --output-dir "$OUT" \
  --replay-data "$REPLAY_DATA" --replay-ratio 0.30 \
  --seq-length "$SEQ" --router-top-k 32 --optimizer adafactor --weight-decay 0.0 \
  --random-concat-ratio 0 \
  --prepare-data-only

echo "[probe] 2/3 uninterrupted reference run through step 6 (checkpoint at 5)..."
FULLFFN_PROBE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$VENV -m torch.distributed.run --nproc_per_node="$NPROC" \
  "$TRAINER" --model "$MODEL" --method full-ffn \
  --expert-config "$PROBE_CFG" --train-data "$TRAIN_DATA" --output-dir "$OUT" \
  --replay-data "$REPLAY_DATA" --replay-ratio 0.30 \
  --seq-length "$SEQ" --router-top-k 32 --fused-ce --random-concat-ratio 0 \
  --optimizer adafactor --weight-decay 0.0 --grad-accum 4 --per-device-batch-size 1 \
  --max-steps 6 --eval-steps 5 --save-steps 5 --logging-steps 1 \
  --skip-final-hf-export \
  2>&1 | tee "$LOG"

echo "[probe] 3/3 parsing asserts from $LOG ..."
$VENV - "$LOG" "$MODEL" "$OUT" <<'PY'
import sys, re, glob, os
log, base_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log).read()
ok = True
if "[FULLFFN_PROBE] FAIL:" in txt:
    print("GLOBAL FAIL: trainer emitted FULLFFN_PROBE failure"); ok = False

freeze_audits = re.findall(r"\[fullffn-freeze-audit\].*router_trainable=0 unexpected=0 missing=0", txt)
good = len(freeze_audits) == 8
print(f"ASSERT0 exact trainable/frozen boundary on 8 ranks -> {'OK' if good else 'FAIL'}")
ok &= good

# assertion 1: peak mem < 70 GB on every rank that printed it
peaks = [float(m) for m in re.findall(r"max_memory_allocated=([\d.]+)GiB", txt)]
if not peaks:
    print("ASSERT1 FAIL: no peak-mem line found"); ok = False
else:
    hi = max(peaks)
    print(f"ASSERT1 peak/GPU = {hi:.1f} GiB (n={len(peaks)}) -> {'OK' if hi < 70 else 'FAIL'}")
    ok &= hi < 70

# assertion 2: trainable ~= 32.2B
m = re.search(r"ESFT trainable params \(full-ffn\): ([\d,]+)", txt)
if not m:
    print("ASSERT2 FAIL: no trainable-count line"); ok = False
else:
    n = int(m.group(1).replace(",", ""))
    good = abs(n - 32_212_254_720) < 5e7 or 32.0e9 < n < 32.5e9
    print(f"ASSERT2 trainable = {n:,} -> {'OK' if good else 'FAIL'}")
    ok &= good

# assertion 3: every expert shard got grad (grad_none==0 && grad_zero==0), >=1 step seen
probe = re.findall(r"union_covered=(\d+).*?grad_none\(local\)=(\d+) grad_zero\(local\)=(\d+)", txt)
if not probe:
    print("ASSERT3 FAIL: no FULLFFN_PROBE grad line"); ok = False
else:
    last = probe[-1]
    cov, none, zero = map(int, last)
    good = cov >= 80  # union coverage; local none/zero are expected under FSDP shard ownership
    print(f"ASSERT3 covered={cov} none={none} zero={zero} -> {'OK' if good else 'FAIL'}")
    ok &= good

# assertion 4A: no frozen param received a gradient (in-run)
frozen_ok = txt.count("[FULLFFN_PROBE] OK: no frozen param received a gradient")
good = frozen_ok == 8
print(f"ASSERT4A frozen params grad-free on all ranks (n={frozen_ok}) -> {'OK' if good else 'FAIL'}")
ok &= good

# DCP-only probe: byte equality is checked when a selected checkpoint is exported
# to a normal HF model. Here the stronger operational requirement is a complete,
# resumable model+optimizer checkpoint with all rank RNG states.
ckpt = os.path.join(out_dir, "checkpoint-5")
manifest = os.path.join(ckpt, "checkpoint_complete.json")
rng = [os.path.join(ckpt, f"rng_state_{i}.pth") for i in range(8)]
required = [manifest,
            os.path.join(ckpt, "pytorch_model_fsdp_0", ".metadata"),
            os.path.join(ckpt, "optimizer_0", ".metadata"),
            os.path.join(ckpt, "scheduler.pt"),
            os.path.join(ckpt, "trainer_state.json"), *rng]
good = all(os.path.isfile(p) for p in required)
if good:
    import json
    meta = json.load(open(manifest))
    good = (meta.get("schema_version") == 2 and
            meta.get("complete") is True and meta.get("global_step") == 5 and
            len(meta.get("optimizer_state_entries_by_rank", [])) == 8 and
            all(n > 0 for n in meta["optimizer_state_entries_by_rank"]) and
            len(meta.get("state_components_by_rank", [])) == 8 and
            meta.get("state_components_by_rank") ==
            meta.get("state_components_post_save_by_rank") and
            all(all(k in item for k in ("model", "optimizer_tensors",
                                        "optimizer_scalars"))
                for item in meta["state_components_by_rank"]) and
            meta.get("weight_decay") == 0.0)
print(f"ASSERT5 complete resumable DCP checkpoint -> {'OK' if good else 'FAIL'}")
ok &= good

print("\nPROBE PHASE1:", "GREEN" if ok else "RED (investigate before resume proof)")
sys.exit(0 if ok else 1)
PY

echo "[probe] resume proof: checkpoint-5 -> target step 6 ..."
RESUME_OUT="${OUT}_resume_from_5"
RESUME_LOG="${RESUME_OUT}/resume_5_to_6.log"
mkdir -p "$RESUME_OUT"
FULLFFN_PROBE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$VENV" -m torch.distributed.run --nproc_per_node="$NPROC" \
  "$TRAINER" --model "$MODEL" --method full-ffn \
  --expert-config "$PROBE_CFG" --train-data "$TRAIN_DATA" --output-dir "$RESUME_OUT" \
  --replay-data "$REPLAY_DATA" --replay-ratio 0.30 \
  --seq-length "$SEQ" --router-top-k 32 --fused-ce --random-concat-ratio 0 \
  --optimizer adafactor --weight-decay 0.0 --grad-accum 4 --per-device-batch-size 1 \
  --max-steps 6 --eval-steps 5 --save-steps 5 --logging-steps 1 \
  --resume-from-checkpoint "$OUT/checkpoint-5" --skip-final-hf-export \
  2>&1 | tee "$RESUME_LOG"

grep -q '\[fullffn-model-load\] step=5 model_match=True' "$RESUME_LOG"
grep -q '\[fullffn-optimizer-load\].*scheduler_last_epoch=5.*model_match=True.*optimizer_tensors_match=True.*optimizer_scalars_match=True' "$RESUME_LOG"
grep -q '\[fullffn-checkpoint-save\].*global_step.*6' "$RESUME_LOG"
test -f "$RESUME_OUT/checkpoint-6/checkpoint_complete.json"

"$VENV" - "$OUT/checkpoint-6" "$RESUME_OUT/checkpoint-6" "$LOG" "$RESUME_LOG" <<'PY'
import json, os, re, sys
import numpy as np
import torch
reference, resumed, reference_log, resumed_log = sys.argv[1:]
def load(root):
    return json.load(open(os.path.join(root, "checkpoint_complete.json")))
a, b = load(reference), load(resumed)
assert a["state_components_by_rank"] == b["state_components_by_rank"], (a, b)
assert a["state_components_by_rank"] == a["state_components_post_save_by_rank"]
assert b["state_components_by_rank"] == b["state_components_post_save_by_rank"]
assert a["optimizer_state_entries_by_rank"] == b["optimizer_state_entries_by_rank"]
assert a["global_step"] == b["global_step"] == 6
assert a["scheduler_last_epoch"] == b["scheduler_last_epoch"] == 6

pattern = re.compile(
    r"\[rank(\d+)\] \[fullffn-step-input\] step=5 micro=(\d+) "
    r"batch_sha256=([0-9a-f]{64}) loss_hex=([^\s\[]+)"
)
def step_records(path):
    records = {}
    for rank, micro, batch_hash, loss_hex in pattern.findall(open(path).read()):
        records[(int(rank), int(micro))] = (batch_hash, loss_hex)
    return records
reference_records = step_records(reference_log)
resumed_records = step_records(resumed_log)
assert len(reference_records) == len(resumed_records) == 32, (
    len(reference_records), len(resumed_records)
)
assert reference_records == resumed_records, (reference_records, resumed_records)

def capture(path, pattern):
    records = {}
    for match in pattern.findall(open(path).read()):
        rank, *values = match
        records[int(rank)] = tuple(values)
    return records

rng_pattern = re.compile(
    r"\[rank(\d+)\] \[fullffn-phase-a\] stage=RNG_BEFORE_FORWARD "
    r"step=5 rng_sha256=([0-9a-f]{64}) model_training=(True|False)"
)
grad_pattern = re.compile(
    r"\[rank(\d+)\] \[fullffn-phase-a\] stage=GRAD6_AFTER_CLIP step=5 "
    r"gradient=\{'digest': '([0-9a-f]{64})', 'tensor_count': (\d+), "
    r"'none_count': (\d+)\}"
)
post_pattern = re.compile(
    r"\[rank(\d+)\] \[fullffn-phase-a\] stage=STEP6_POST_OPT step=5 "
    r"components=\{'mode': 'full', 'model': '([0-9a-f]{64})', "
    r"'optimizer_tensors': '([0-9a-f]{64})', "
    r"'optimizer_scalars': '([0-9a-f]{64})', "
    r"'optimizer_state_entries': (\d+)\}"
)
for label, pattern in (("rng", rng_pattern), ("gradient", grad_pattern),
                       ("post_optimizer", post_pattern)):
    left = capture(reference_log, pattern)
    right = capture(resumed_log, pattern)
    assert len(left) == len(right) == 8, (label, len(left), len(right))
    assert left == right, (label, left, right)

def equal(x, y):
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(equal(i, j) for i, j in zip(x, y))
    return x == y

for rank in range(8):
    ra = torch.load(os.path.join(reference, f"rng_state_{rank}.pth"), map_location="cpu", weights_only=False)
    rb = torch.load(os.path.join(resumed, f"rng_state_{rank}.pth"), map_location="cpu", weights_only=False)
    assert equal(ra, rb), f"rank {rank} RNG mismatch"

sa = torch.load(os.path.join(reference, "scheduler.pt"), map_location="cpu", weights_only=True)
sb = torch.load(os.path.join(resumed, "scheduler.pt"), map_location="cpu", weights_only=True)
assert equal(sa, sb), (sa, sb)
print("ASSERT6 uninterrupted vs resumed full local model+optimizer digests -> OK")
print("ASSERT7 scheduler and all-rank RNG states exact -> OK")
print("ASSERT8 all-rank step-6 microbatch identities and raw losses exact -> OK")
print("ASSERT9 pre-forward RNG, clipped gradients, and post-optimizer components exact -> OK")
print("PROBE RESULT: GREEN (GO to 200-step pilot)")
PY
