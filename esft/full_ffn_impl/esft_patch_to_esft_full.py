# ---------------------------------------------------------------------------
# ADD THIS FUNCTION to esft_qwen/esft_patch.py
# (paste right AFTER `enable_router_training` and BEFORE `snapshot_router_weights`)
#
# Purely additive: it introduces no change to any existing symbol, so the delta
# and maskhook paths are byte-for-byte unaffected. It reuses EsftHandles and
# find_moe_blocks, both already imported/defined in esft_patch.py.
# ---------------------------------------------------------------------------


def to_esft_full(model: torch.nn.Module, *, train_shared_experts: bool = False) -> "EsftHandles":
    """FULL-FFN training: unfreeze EVERY routed expert's packed FFN tensors across
    ALL MoE layers, keeping router/gate/shared_expert/attn/embed/lm_head frozen.

    Contrast with :func:`to_esft_qwen` (maskhook) and the delta path
    ---------------------------------------------------------------
    Those two paths exist to train ONLY a *subset* of experts without paying for a
    full 32B gradient: maskhook zeroes non-selected gradient rows in a hook; delta
    trains small side Parameters. FULL-FFN trains all 256 experts of every layer,
    so there is nothing to mask (the whole dim-0 expert axis is supervised) and no
    delta to add -- we simply flip ``requires_grad`` on the two packed Parameters
    per layer and let FSDP FULL_SHARD split the resulting ~32.2B trainable params
    across ranks. Consequently:

      * NO grad hook is registered (``handles.hook_handles`` stays empty) -- there
        are no non-selected rows to zero.
      * NO mask is built (``handles.masks`` stays empty).
      * NO delta Parameter is created.
      * The optimiser is NOT built here and NOT via ``build_param_groups``; the HF
        Trainer builds it AFTER accelerate wraps the model in FSDP (a pre-wrap
        optimiser would capture unsharded params). Weight-decay policy is handled
        by ``--optim`` / ``--weight-decay`` on the trainable (expert) params only,
        since everything else is frozen.

    ``handles.expert_params`` still lists the two packed Parameters per layer so the
    GRAD_PROBE / FULLFFN_PROBE callbacks can measure their grad norm, and so the
    trainable-param count assertion has a reference set.

    Frozen-by-construction (never unfrozen here): ``gate.weight`` (router),
    ``shared_expert`` + ``shared_expert_gate``, attention (full + linear/GatedDelta),
    ``embed_tokens``, ``lm_head``, all RMSNorms. ``train_shared_experts=True`` opts
    the per-layer shared_expert MLP back in (off by default -- the campaign trains
    only the routed-expert FFN capacity).
    """
    # 1. Freeze everything. Non-expert modules stay frozen for full-FFN (unlike
    #    to_esft_qwen's train_non_expert_modules option), so pass False.
    model.requires_grad_(False)

    handles = EsftHandles(expert_config={})

    refs = find_moe_blocks(model)
    for ref in refs:
        experts = ref.experts
        for pname in ("gate_up_proj", "down_proj"):
            param = getattr(experts, pname)
            param.requires_grad_(True)
            handles.expert_params.append(param)
        if train_shared_experts and ref.shared_expert is not None:
            ref.shared_expert.requires_grad_(True)

    if not handles.expert_params:
        raise ValueError("to_esft_full found no routed-MoE experts to unfreeze")
    return handles
