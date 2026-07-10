"""ESFT relevance scoring and cumulative top-p expert selection.

Mirrors the DeepSeek ESFT reference (scripts/expert/generate_expert_config.py):

  gate_score[l, e]  += routing weight that token gave to expert e   (affinity)
  token_score[l, e] += 1 / top_k   for every token routed to expert e

Both are then divided by the total token count, so each row sums to ~1 over experts
(gate: per-token weights renormalise to 1; token: each token contributes top_k
selections * 1/top_k = 1). Selection then takes experts in descending score order
until the cumulative score reaches ``top_p``.
"""

from __future__ import annotations

import numpy as np


class ScoreAccumulator:
    """Streaming accumulation of ESFT gate/token scores over a corpus.

    Fed one MoE layer at a time via :meth:`update`; tracks the processed token
    count off the lowest-index MoE layer (every layer sees the same tokens).
    """

    def __init__(self, num_layers: int, num_experts: int, top_k: int):
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gate_raw = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.token_raw = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.n_tokens = 0
        self._ref_layer: int | None = None  # lowest layer index, used to count tokens

    def update(self, layer_pos: int, top_idx, top_val) -> None:
        """Accumulate one layer's routing for a batch of tokens.

        ``layer_pos`` is the 0-based position among MoE layers (not the raw model
        layer index). ``top_idx``/``top_val`` are ``(n_tokens, top_k)`` arrays.
        """
        idx = np.asarray(top_idx).reshape(-1, self.top_k)
        val = np.asarray(top_val, dtype=np.float64).reshape(-1, self.top_k)
        n = idx.shape[0]
        flat_idx = idx.reshape(-1)
        np.add.at(self.gate_raw[layer_pos], flat_idx, val.reshape(-1))
        np.add.at(self.token_raw[layer_pos], flat_idx, np.full(flat_idx.shape, 1.0 / self.top_k))
        if self._ref_layer is None or layer_pos < self._ref_layer:
            # First time we see a new lowest layer, (re)count from it.
            self._ref_layer = layer_pos
        if layer_pos == self._ref_layer:
            self.n_tokens += n

    def finalize(self) -> dict:
        """Return normalised score matrices and metadata."""
        total = float(self.n_tokens) if self.n_tokens else 1.0
        return {
            "gate_scores": self.gate_raw / total,
            "token_scores": self.token_raw / total,
            "gate_raw": self.gate_raw,
            "token_raw": self.token_raw,
            "n_tokens": self.n_tokens,
            "top_k": self.top_k,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
        }


def select_experts_top_p(scores_1d, top_p: float) -> list[int]:
    """Cumulative top-p selection, exactly matching the ESFT reference order.

    Experts are taken in descending score until the *running* cumulative score
    (checked before adding the next expert) reaches ``top_p``. Returned sorted
    ascending for stable, human-readable configs.
    """
    scores = np.asarray(scores_1d, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    current = 0.0
    for e in order:
        if current >= top_p:
            break
        selected.append(int(e))
        current += float(scores[e])
    return sorted(selected)


def build_expert_config(
    score_matrix,
    layer_indices,
    top_p: float,
    *,
    train_shared_experts: bool = False,
    train_non_expert_modules: bool = False,
) -> dict:
    """Turn a (num_moe_layers, num_experts) score matrix into an ESFT config dict.

    ``layer_indices[i]`` is the raw model layer index for row ``i`` of the matrix,
    so the emitted config keys are real layer indices (ESFT-compatible).
    """
    score_matrix = np.asarray(score_matrix)
    experts = {}
    for row, layer_idx in enumerate(layer_indices):
        experts[str(int(layer_idx))] = select_experts_top_p(score_matrix[row], top_p)
    return {
        "experts": experts,
        "shared_experts": bool(train_shared_experts),
        "non_expert_modules": bool(train_non_expert_modules),
    }


def jaccard(a, b) -> float:
    """Jaccard overlap between two iterables of expert ids."""
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def domain_overlap_matrix(configs: dict[str, dict]) -> dict:
    """Per-layer and mean Jaccard overlap between domains' selected expert sets.

    ``configs`` maps domain name -> ESFT config dict (as from build_expert_config).
    Returns {"domains": [...], "per_layer": {layer: matrix}, "mean": matrix}.
    """
    domains = list(configs)
    n = len(domains)
    # Union of all layer keys present across configs.
    layers = sorted(
        {int(l) for cfg in configs.values() for l in cfg["experts"]},
    )
    per_layer = {}
    accum = np.zeros((n, n), dtype=np.float64)
    for layer in layers:
        mat = np.ones((n, n), dtype=np.float64)
        for i, di in enumerate(domains):
            for j, dj in enumerate(domains):
                ei = configs[di]["experts"].get(str(layer), [])
                ej = configs[dj]["experts"].get(str(layer), [])
                mat[i, j] = jaccard(ei, ej)
        per_layer[layer] = mat
        accum += mat
    mean = accum / max(len(layers), 1)
    return {"domains": domains, "per_layer": per_layer, "mean": mean, "layers": layers}
