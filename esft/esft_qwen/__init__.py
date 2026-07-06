"""ESFT port for Qwen3.6-35B-A3B (Qwen3_5Moe architecture).

Shared library used by the Phase-0 scripts and the smoke test. Kept import-light
(no torch at module import beyond what submodules need) so that CPU-only usage and
`CUDA_VISIBLE_DEVICES=""` runs stay cheap.
"""

from .common import (
    TOP_K_DEFAULT,
    find_moe_blocks,
    compute_router_selection,
    infer_moe_dims,
)
from .scoring import (
    ScoreAccumulator,
    select_experts_top_p,
    build_expert_config,
    jaccard,
)
from .esft_patch import (
    to_esft_qwen,
    save_expert_patch,
    load_expert_patch,
    EsftHandles,
)

__all__ = [
    "TOP_K_DEFAULT",
    "find_moe_blocks",
    "compute_router_selection",
    "infer_moe_dims",
    "ScoreAccumulator",
    "select_experts_top_p",
    "build_expert_config",
    "jaccard",
    "to_esft_qwen",
    "save_expert_patch",
    "load_expert_patch",
    "EsftHandles",
]
