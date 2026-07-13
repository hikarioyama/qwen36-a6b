# Prior art for the α dial

A survey of related work for the **α dial** — the zero-training, weight-free,
deterministic inference-time operation this repository uses to repay the
"k=32 tax" (scale router gate scores at ranks 9–32 by α, then renormalize).

This is a literature *scan*, not an exhaustive search. Everything below is
phrased as "as far as our search found." The phenomenon the dial addresses is
already reported; we did not find an identical method, but we report the nearest
neighbors honestly rather than claim novelty of the phenomenon.

## Phenomenon (prior art)

Activating more experts per token at inference than the model was trained for
(`k' > k`) is a **known failure mode**, not our discovery:

- **Elastic MoE** (arXiv:2509.21892) reports the `k' > k` inference degradation
  directly and names it an **"inference-time scaling wall."** Its remedy is
  training-time.
- **Matryoshka MoE** (arXiv:2509.26520) targets the same elastic-`k` setting with
  a training-time method (nested experts trained to be robust across `k`).

Both solve the wall by *changing training*, whereas the α dial is a runtime
operation with no weight change.

## Closest operation

- **Certain Head, Uncertain Tail** (arXiv:2602.02443) is the nearest neighbor we
  found. It applies **temperature scaling + renormalization to the tail router
  scores** — mechanically adjacent to scaling ranks 9–32. The differences:
  - Its goal is **stochastic test-time sampling diversity**, not deterministic
    calibration.
  - It does **not** carry the `α = 0 ≡ exact top-k` equivalence property that
    makes the dial a clean interpolation anchored at the untouched base model.

  Honest note: this is the closest prior operation we located; the dial is not
  identical to it, but it is not far.

## Our contributions (as far as our search found)

1. **Mechanism decomposition.** The `k' > k` loss is *not* broken experts or
   worse routing. Expert **selection is nested and intact** (the top-8 of a k=32
   selection are the same 8 experts as k=8); the loss is **renormalization
   dilution alone**. This is verified by the `α = 0 ≡ k=8` exact-match check:
   98/100 identical predictions, at the bf16 noise level.
2. **A zero-training, weight-free, deterministic inference-time fix**: rank the
   selected gate scores, multiply ranks 9–32 by α, renormalize. No gradients, no
   weight change; ships as an inference-server setting.
3. **Dial-native training**: train the model with the **same α used at serving**,
   so calibration and capability are decoupled (the dial handles calibration; FFN
   training spends capacity on capability with the router frozen).

## Related

- **ERMoE** (arXiv:2511.10971) — analyzes expert-representation **dilution** and
  proposes an **architectural** fix.
- **SMoE** (arXiv:2508.18983) — **tail-expert substitution** as its mechanism.

Both touch the dilution / tail-expert theme from a training or architectural
angle rather than a deterministic inference-time dial.

## Verification note

All five arXiv IDs were **existence-checked** (title match) and the two
load-bearing abstracts (Elastic MoE, Certain Head/Uncertain Tail) were
**content-checked** on 2026-07-14 (search: Grok web research plus manual arXiv
verification). The search is **not exhaustive** — claims here are stated as "as
far as our search found," not as absolute novelty.
