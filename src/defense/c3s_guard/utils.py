"""Shared types and helper interfaces for the C3S-Guard defense."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

ModelUpdate = Union[Mapping[str, torch.Tensor], torch.Tensor]


@dataclass
class CTSIntentResult:
    """Structured output of the CTS-Intent module."""

    round_idx: int
    group_scores: Dict[int, float] = field(default_factory=dict)
    group_features: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    flagged_groups: List[int] = field(default_factory=list)
    aux_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class S3LocResult:
    """Structured output of the S3-Loc module."""

    round_idx: int
    audit_rounds: int = 0
    client_scores: Dict[int, float] = field(default_factory=dict)
    flagged_clients: List[int] = field(default_factory=list)
    group_assignments: List[List[int]] = field(default_factory=list)
    aux_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DGCCleanResult:
    """Structured output of the DGC-Clean module."""

    round_idx: int
    cleaned_update: Optional[ModelUpdate] = None
    removed_energy: Optional[float] = None
    suspicious_subspace_rank: Optional[int] = None
    aux_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class C3SGuardDecision:
    """End-to-end decision container returned by C3S-Guard hooks."""

    round_idx: int
    audit_triggered: bool = False
    aggregation_action: str = "accept"
    detection: Optional[CTSIntentResult] = None
    localization: Optional[S3LocResult] = None
    cleaning: Optional[DGCCleanResult] = None
    aux_stats: Dict[str, Any] = field(default_factory=dict)


def clone_model_update(update: ModelUpdate) -> ModelUpdate:
    """Deep-copy a model update represented as a tensor or state-dict-like mapping."""

    pass


def flatten_model_update(
    update: Mapping[str, torch.Tensor],
    reference_keys: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, List[str], List[torch.Size]]:
    """Flatten a state-dict-like update into a single vector."""

    pass


def unflatten_model_update(
    vector: torch.Tensor,
    reference_update: Mapping[str, torch.Tensor],
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Restore a flat vector to a state-dict-like update."""

    pass


def average_model_updates(
    updates: Sequence[ModelUpdate],
    weights: Optional[Sequence[float]] = None,
) -> ModelUpdate:
    """Average a list of updates with optional scalar weights."""

    pass


def cosine_similarity_matrix(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute the pairwise cosine-similarity matrix for a batch of vectors."""

    pass


def sample_audit_groups(
    client_ids: Sequence[int],
    num_groups: int,
    group_size: int,
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:
    """Sample random audit groups used by S3-Loc under secure aggregation."""

    pass


def ensure_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Normalize a device specifier to a ``torch.device`` instance."""

    pass


def merge_stats(
    base: Optional[Mapping[str, Any]],
    extra: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge two flat statistics dictionaries into a new dictionary."""

    pass
