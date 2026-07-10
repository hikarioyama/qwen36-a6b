# B2 Continual-Training Knowledge Ledger

Updated: 2026-07-10 JST

This ledger separates primary-source claims, local measurements, inference, and
experiment decisions. Grok was used only to discover candidate URLs. Codex opened
the primary paper pages and, for the highest-impact papers, read the full-text
HTML/PDF directly.

## Discovery quality

- Raw Grok result: `grok-urls-v2.raw.txt` (20 URLs plus one unwanted status line).
- Validated set: `grok-urls-v2.validated.txt` (17 relevant primary-paper URLs).
- Rejected false positives:
  - `https://arxiv.org/abs/2202.08908`: quantum optimal control.
  - `https://arxiv.org/abs/2206.14365`: quantum magnomechanics.
  - `https://aclanthology.org/2020.acl-main.674/`: advertisement understanding.
- Direct-reading depth in this revision:
  - full text/section-level: Sparse Upcycling, BTX, continual LR re-warming,
    continual forgetting, forward/reverse KL, and O-LoRA;
  - primary abstract plus bibliographic verification: remaining validated papers.

## Local measured facts

These are observations from this project, not paper claims.

- Model anchor: true-stock Qwen3.6-35B-A3B snapshot `995ad96...`.
- B2 intervention: selected-expert delta training, frozen router, router top-k 32,
  CE plus beta=0.5 forward KL over cached teacher top-64 support.
- Trainable delta parameters: 2,620,391,424 across 80 delta tensors.
- Packed data: 22,238 blocks at sequence length 7168, seed 5934875.
- B2 step 500 completed with eval loss 0.66595 and a monotonic recorded loss
  decline. Resume probe step 502 had eval loss 0.6664 and LR 1e-5.
- True resume to step 1000 is live on the training node; all eight ranks restored global step
  500, 80 optimizer states, scheduler epoch 500, and first completed step 501.
- Historical transfer anchor: base MMLU@k8 0.8467 and B2 MMLU@k32 0.8417 on 600
  items. The paired difference was unresolved, not proof of equivalence.
- A paired local GSM8K gate is live: base@k8 then B2@k32, n=600, no-think,
  max-new 2048, predeclared regression margin 0.02.

## Evidence and implications

### Sparse MoE initialization and routing

Primary sources:

- Switch Transformers: https://arxiv.org/abs/2101.03961
- GShard: https://arxiv.org/abs/2006.16668
- Sparse Upcycling: https://arxiv.org/abs/2212.05055
- Mixtral: https://arxiv.org/abs/2401.04088
- MoEfication: https://arxiv.org/abs/2110.01786

Paper claims:

- Sparse activation can increase parameter capacity without proportional active
  compute, but routing, communication, capacity, and training instability are
  central failure modes.
- Sparse Upcycling initializes every expert from the existing dense MLP and then
  continues training. Randomly initialized experts recover much more slowly.
- Sparse Upcycling found top-k routing workable, but routing/capacity treatment
  affects which tokens are retained when experts are full.

Applicability and inference:

- These papers support continuation from a strong checkpoint rather than
  rebuilding experts, but they do not validate expanding an already sparse
  Qwen-style router from k8 to k32.
- B2 keeps the router frozen and updates selected expert deltas. Therefore loss
  improvement alone cannot show that the newly active expert set is useful or
  balanced. Routing/co-activation measurements remain required.

Decision:

- Keep the current frozen-router B2 run unchanged.
- For B2-1000 evaluation, record per-layer expert utilization/co-activation when
  feasible; do not infer specialization from benchmark score alone.

### Domain experts and load-balancing trade-offs

Primary source:

- Branch-Train-MiX: https://arxiv.org/abs/2403.07816

Paper claims:

- BTX trains domain experts independently, combines their feed-forward parameters
  into an MoE, then learns routing in a mix stage.
- Its ablations show that removing load balancing hurt HumanEval but improved
  GSM8K; routing analysis found heavy reliance on the math expert and a nearly
  dead code expert without load balancing.
- With load balancing, the code expert became active and dominant for math/code,
  demonstrating that routing changes can trade coding against math rather than
  uniformly improve both.

Applicability and inference:

- BTX differs from B2 because B2 neither creates domain-labelled experts nor
  retrains the router. Still, the observed GSM8K/HumanEval opposition is directly
  relevant to this project's preservation question.
- The current paired order, GSM8K followed by HumanEval, is scientifically
  necessary. A GSM8K pass cannot be used as a proxy for coding preservation.

Decision:

- Preserve separate GSM8K and HumanEval gates and their predeclared margins.
- If the two move in opposite directions, inspect routing/co-activation before
  changing KL beta or training duration.

### Continual pre-training, LR, replay, and forgetting

Primary sources:

- Continual Pre-training of Language Models: https://arxiv.org/abs/2302.03241
- LR re-warming study: https://arxiv.org/abs/2308.04014
- Empirical catastrophic forgetting: https://arxiv.org/abs/2308.08747
- Forgetting in aligned models: https://arxiv.org/abs/2401.03129
- Replay/re-warm/re-decay: https://arxiv.org/abs/2403.08763
- Instruction mixing: https://arxiv.org/abs/2312.10793

Paper claims:

- Continual adaptation can improve new-domain performance while degrading old
  data or general-task performance; larger models are not automatically immune.
- The re-warming study found a clear adaptation/retention trade-off: larger peak
  LR improved downstream loss and worsened upstream loss. Constant LR was strong
  early, and early stopping was an economical retention strategy.
- The later scalable study reports that LR re-warming, re-decay, and replay can
  approach joint retraining in its tested settings.
- Continual-forgetting studies evaluate domain knowledge, reasoning, reading
  comprehension, format, and reliability separately; a single aggregate metric
  can miss important regressions.
- Instruction mixtures can help one capability while degrading another.

Applicability and inference:

- The LR studies use much smaller dense models and different data scales. They do
  not prescribe an LR for this 35B-A3B MoE run.
- B2's constant 1e-5 schedule makes training duration the current exposure axis.
  Step 750 is therefore a useful intermediate checkpoint, not merely recovery
  state.
- KL to the k8 teacher is a functional preservation constraint, but it is not the
  same as replaying the original data distribution. A future replay/mixing arm
  remains plausible only if paired gates show forgetting.

Decision:

- Do not change the live step-500-to-1000 schedule.
- Preserve checkpoint 750 and evaluate it if step 1000 regresses, enabling an
  early-stopping comparison without rerunning training.
- If preservation fails, discuss a separately preflighted small replay/mix arm;
  do not retroactively reinterpret beta=0.5 KL as equivalent to replay.

### Forward KL and truncated teacher support

Primary sources:

- Original distillation: https://arxiv.org/abs/1503.02531
- MiniLLM: https://arxiv.org/abs/2306.08543
- Rethinking KL for LLM distillation: https://arxiv.org/abs/2404.02657

Paper claims:

- Token-level forward KL is the standard white-box logit-alignment objective.
- The KL rethinking paper argues that familiar continuous-distribution slogans
  about forward-KL mean-seeking and reverse-KL mode-seeking do not directly hold
  for discrete LLM token distributions.
- In limited training, forward KL fits the teacher distribution's head earlier,
  while reverse KL emphasizes the tail earlier; both have the same limiting
  objective under sufficient optimization in the paper's analysis.
- MiniLLM reports gains from reverse KL in generative distillation, but its method
  is an on-policy system rather than a drop-in divergence swap: it adds student
  sampling, teacher-mixed sampling, single-step regularization, length
  normalization, and a pretraining-corpus language-modeling loss. Its ablations
  report degenerate short/repeated outputs when key stabilizers are removed.
- Its experiments were on substantially smaller students, and the authors mark
  scaling to larger models as unresolved.

Applicability and inference:

- B2 stores only the teacher's top-64 support. That deliberately exposes the head
  and discards most of the tail, so forward KL's early head emphasis is aligned
  with what the cache can represent.
- This supports beta=0.5 forward KL as a pragmatic preservation signal but does
  not prove that top-64 truncation preserves all broad capabilities.
- Reverse or adaptive KL cannot be reconstructed faithfully from the current
  cache without quantifying missing teacher mass.
- MiniLLM therefore does not justify changing only the sign/order of KL in B2:
  the current teacher-forced top-64 cache lacks its on-policy sampling and tail
  information, and would test a materially different, incomplete method.

Decision:

- Keep forward KL beta=0.5 for the live run.
- Before proposing reverse/adaptive KL, measure teacher top-64 retained probability
  mass by token/domain and determine whether the cached support is adequate.

### Orthogonal low-rank updates

Primary sources:

- LoRA: https://arxiv.org/abs/2106.09685
- O-LoRA: https://arxiv.org/abs/2310.14152

Paper claims:

- O-LoRA freezes prior task adapters and learns new low-rank updates in subspaces
  constrained to be orthogonal to earlier adapter subspaces.
- In its LLaMA-7B continual-learning experiment, the orthogonality constraint
  preserved substantially more zero-shot MMLU than unconstrained incremental
  LoRA variants.

Applicability and inference:

- B2's 2.62B selected-expert deltas are not low-rank LoRA matrices, so O-LoRA is
  not directly transplantable.
- The relevant mechanism is interference geometry: if B2 improves the target but
  fails preservation gates, measuring overlap between B2 update directions and
  protected/general gradients could guide a narrower next experiment.

Decision:

- Do not add orthogonality to the live run.
- Retain it as a next-branch candidate only after a measured preservation failure
  and a cheap gradient/subspace diagnostic.

## Current synthesis

The literature and local evidence agree on one operational point: specialization,
routing, learning rate/exposure, and preservation are coupled trade-offs. None of
the papers justifies promoting B2 from training loss alone. The current shortest
valid path remains:

1. finish paired GSM8K and HumanEval for B2-500;
2. preserve training-node checkpoints 750 and 1000;
3. evaluate B2-1000 against the frozen stock anchor, with checkpoint 750 available
   as an early-stopping candidate;
4. use routing/co-activation plus benchmark deltas to choose among more training,
   replay/mixing, or interference-constrained updates.

No literature finding in this revision warrants changing or stopping the live
B2 continuation before measured gate results arrive.
