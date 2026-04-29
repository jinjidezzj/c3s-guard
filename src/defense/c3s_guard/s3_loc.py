"""S3-Loc: secure-aggregation-constrained multi-round localization.

This module implements the localization stage of C3S-Guard under secure
aggregation. The server only observes group-level anomaly labels ``b_g`` and
historical group composition records, then infers which clients are likely to be
malicious.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from defense.c3s_guard.utils import S3LocResult

try:
    from sklearn.linear_model import Lasso
except ImportError:  # pragma: no cover - handled at runtime.
    Lasso = None


class S3Loc:
    """Localize suspicious clients from repeated random group observations."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_clients: int,
        group_size: int,
        audit_rounds: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the S3-Loc localizer.

        Args:
            config: Dictionary of S3-Loc hyper-parameters.
            num_clients: Total number of clients in the federation.
            group_size: Number of clients in each secure-aggregation audit group.
            audit_rounds: Number of random grouping rounds per audit cycle.
            device: Device used for tensor bookkeeping.
        """

        self.config = self._build_config(config)
        self.num_clients = int(num_clients)
        self.group_size = int(group_size)
        self.audit_rounds = int(audit_rounds)
        self.device = torch.device("cpu") if device is None else torch.device(device)

        self._group_assignments: List[List[int]] = []
        self._group_labels: List[bool] = []
        self._group_scores: List[float] = []
        self._dropout_counts: List[int] = []
        self._round_indices: List[int] = []
        self._last_result: Optional[S3LocResult] = None

    def build_audit_plan(
        self,
        client_ids: Sequence[int],
        generator: Optional[torch.Generator] = None,
    ) -> Sequence[Sequence[int]]:
        """
        Build a random group schedule for one audit cycle.

        The returned value is a flat list of groups across all audit rounds. Each
        audit round shuffles the available clients and then partitions them into
        groups of size ``self.group_size``.

        Args:
            client_ids: Candidate client ids available for auditing.
            generator: Optional random generator for reproducible grouping.

        Returns:
            A flat sequence of groups, where each group is a sequence of client
            ids.
        """

        client_ids = list(client_ids)
        if len(client_ids) == 0:
            return []

        rng = generator
        if rng is None:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config["seed"]))

        all_groups: List[List[int]] = []
        for _ in range(self.audit_rounds):
            perm = torch.randperm(len(client_ids), generator=rng).tolist()
            shuffled = [client_ids[idx] for idx in perm]
            for start in range(0, len(shuffled), self.group_size):
                group = shuffled[start:start + self.group_size]
                if len(group) > 0:
                    all_groups.append(group)
        return all_groups

    def accumulate_observation(
        self,
        round_idx: int,
        group_assignments: Sequence[Sequence[int]],
        group_scores: Mapping[int, float],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Store one batch of audited group observations.

        Args:
            round_idx: Global audit round index.
            group_assignments: Sequence of audited groups. The order of this
                sequence defines the row order for the latent sensing matrix.
            group_scores: Mapping from group id to CTS score or binary decision.
            metadata: Optional dictionary. Supported keys:
                - ``group_ids``: explicit order matching ``group_assignments``.
                - ``b_g`` or ``group_labels``: binary anomaly labels per group.
                - ``dropout_counts``: actual surviving members per group.
        """

        metadata = dict(metadata or {})
        group_ids = metadata.get("group_ids")
        if group_ids is None:
            group_ids = list(group_scores.keys())

        labels = metadata.get("b_g", metadata.get("group_labels"))
        if labels is None:
            labels = [bool(group_scores[group_id] > 0.0) for group_id in group_ids]
        else:
            labels = [bool(label) for label in labels]

        dropout_counts = metadata.get("dropout_counts")
        if dropout_counts is None:
            dropout_counts = [len(group) for group in group_assignments]

        if len(group_assignments) != len(labels):
            raise ValueError("group_assignments and labels must have the same length.")
        if len(group_assignments) != len(dropout_counts):
            raise ValueError("group_assignments and dropout_counts must have the same length.")

        for index, group in enumerate(group_assignments):
            group_id = group_ids[index] if index < len(group_ids) else index
            self._group_assignments.append(list(group))
            self._group_scores.append(float(group_scores[group_id]))
            self._group_labels.append(bool(labels[index]))
            self._dropout_counts.append(int(dropout_counts[index]))
            self._round_indices.append(int(round_idx))

    def estimate_client_scores(
        self,
        candidate_client_ids: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        """Estimate client-level suspicion scores from accumulated observations."""

        if len(self._group_assignments) == 0:
            return {}

        A_matrix = self._build_matrix_from_groups(self._group_assignments, self.num_clients)
        method = str(self.config["method"]).lower()
        if method == "weighted_counting":
            weighted_y, _ = self._prepare_weighted_signal(
                group_signal=self._group_scores,
                designed_group_sizes=A_matrix.sum(axis=1),
                dropout_counts=self._dropout_counts,
            )
            participation = np.maximum(A_matrix.sum(axis=0), 1.0)
            raw_scores = (A_matrix.T @ weighted_y) / np.sqrt(participation)
        elif method == "contrastive_weighted_counting":
            weighted_y, contrastive_stats = self._prepare_contrastive_signal(
                b_g_list=self._group_labels,
                group_signal=self._group_scores,
                designed_group_sizes=A_matrix.sum(axis=1),
                dropout_counts=self._dropout_counts,
                negative_weight=float(self.config.get("contrastive_negative_weight", 0.25)),
                negative_quantile=float(self.config.get("contrastive_negative_quantile", 0.9)),
                positive_scale=float(self.config.get("contrastive_positive_scale", 1.0)),
            )
            participation = np.maximum(A_matrix.sum(axis=0), 1.0)
            raw_scores = (A_matrix.T @ weighted_y) / np.sqrt(participation)
            if (
                float(np.max(raw_scores)) <= 1e-8
                and float(contrastive_stats.get("fallback_positive_mass", 0.0)) > 0.0
            ):
                positive_only = np.asarray(
                    contrastive_stats.get("positive_only_signal", weighted_y),
                    dtype=np.float32,
                )
                raw_scores = (A_matrix.T @ positive_only) / np.sqrt(participation)
        else:
            y_corrected = self._prepare_observation_vector(
                b_g_list=self._group_labels,
                designed_group_sizes=A_matrix.sum(axis=1),
                dropout_counts=self._dropout_counts,
            )
            raw_scores = A_matrix.T @ y_corrected

        if bool(self.config.get("pos_neg_contrast_enabled", True)):
            raw_scores, _ = self._apply_pos_neg_contrast(
                base_scores=raw_scores,
                A_matrix=A_matrix,
                b_g_list=self._group_labels,
                contrast_lambda=float(self.config.get("pos_neg_contrast_lambda", 0.9)),
                blend=float(self.config.get("pos_neg_contrast_blend", 0.20)),
                min_positive_groups=int(self.config.get("pos_neg_contrast_min_positive_groups", 3)),
                min_negative_groups=int(self.config.get("pos_neg_contrast_min_negative_groups", 3)),
            )

        if candidate_client_ids is None:
            candidate_client_ids = range(self.num_clients)
        return {int(client_id): float(raw_scores[int(client_id)]) for client_id in candidate_client_ids}

    def localize(
        self,
        round_idx: int,
        candidate_client_ids: Optional[Sequence[int]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> S3LocResult:
        """Run the full S3-Loc localization step for the current audit buffer."""

        metadata = dict(metadata or {})
        if len(self._group_assignments) == 0:
            result = S3LocResult(round_idx=round_idx, audit_rounds=0)
            self._last_result = result
            return result

        s = int(metadata.get("s", self.config["suspicious_topk"]))
        method = str(metadata.get("method", self.config["method"]))
        suspicious_set, p_hat, degenerate, client_scores, aux_stats = localize_s3_clients(
            A_matrix=self._build_matrix_from_groups(self._group_assignments, self.num_clients),
            b_g_list=self._group_labels,
            s=s,
            method=method,
            group_signal=self._group_scores,
            dropout_counts=self._dropout_counts,
            lasso_alpha=float(metadata.get("lasso_alpha", self.config["lasso_alpha"])),
            degenerate_threshold=float(
                metadata.get("degenerate_threshold", self.config["degenerate_threshold"])
            ),
            contrastive_negative_weight=float(
                metadata.get("contrastive_negative_weight", self.config["contrastive_negative_weight"])
            ),
            contrastive_negative_quantile=float(
                metadata.get("contrastive_negative_quantile", self.config["contrastive_negative_quantile"])
            ),
            contrastive_positive_scale=float(
                metadata.get("contrastive_positive_scale", self.config["contrastive_positive_scale"])
            ),
            pos_neg_contrast_enabled=bool(
                metadata.get("pos_neg_contrast_enabled", self.config.get("pos_neg_contrast_enabled", True))
            ),
            pos_neg_contrast_lambda=float(
                metadata.get("pos_neg_contrast_lambda", self.config.get("pos_neg_contrast_lambda", 0.9))
            ),
            pos_neg_contrast_blend=float(
                metadata.get("pos_neg_contrast_blend", self.config.get("pos_neg_contrast_blend", 0.20))
            ),
            pos_neg_contrast_min_positive_groups=int(
                metadata.get(
                    "pos_neg_contrast_min_positive_groups",
                    self.config.get("pos_neg_contrast_min_positive_groups", 3),
                )
            ),
            pos_neg_contrast_min_negative_groups=int(
                metadata.get(
                    "pos_neg_contrast_min_negative_groups",
                    self.config.get("pos_neg_contrast_min_negative_groups", 3),
                )
            ),
            candidate_client_ids=candidate_client_ids,
            return_details=True,
        )

        result = S3LocResult(
            round_idx=round_idx,
            audit_rounds=len(set(self._round_indices)),
            client_scores=client_scores,
            flagged_clients=suspicious_set,
            group_assignments=[list(group) for group in self._group_assignments],
            aux_stats={
                "p_hat": p_hat,
                "degenerate_mode": degenerate,
                "method": aux_stats["method"],
                "raw_positive_ratio": aux_stats["raw_positive_ratio"],
                "corrected_y": aux_stats["corrected_y"],
                "weighted_y": aux_stats.get("weighted_y", []),
                "signal_center": aux_stats.get("signal_center", 0.0),
                "signal_scale": aux_stats.get("signal_scale", 1.0),
                "pos_neg_contrast_enabled": aux_stats.get("pos_neg_contrast_enabled", False),
                "pos_neg_contrast_reason": aux_stats.get("pos_neg_contrast_reason", "none"),
                "pos_neg_contrast_lambda": aux_stats.get("pos_neg_contrast_lambda", 0.9),
                "pos_neg_contrast_blend": aux_stats.get("pos_neg_contrast_blend", 0.20),
                "selected_pos_rate": aux_stats.get("selected_pos_rate", {}),
                "selected_neg_rate": aux_stats.get("selected_neg_rate", {}),
                "selected_contrast_scores": aux_stats.get("selected_contrast_scores", {}),
                "selected_scores": aux_stats["selected_scores"],
            },
        )
        self._last_result = result
        return result

    def reset_cycle(self) -> None:
        """Clear the current audit buffer while keeping persistent priors."""

        self._group_assignments = []
        self._group_labels = []
        self._group_scores = []
        self._dropout_counts = []
        self._round_indices = []

    def state_dict(self) -> Dict[str, Any]:
        """Serialize S3-Loc state."""

        return {
            "config": dict(self.config),
            "num_clients": self.num_clients,
            "group_size": self.group_size,
            "audit_rounds": self.audit_rounds,
            "group_assignments": [list(group) for group in self._group_assignments],
            "group_labels": list(self._group_labels),
            "group_scores": list(self._group_scores),
            "dropout_counts": list(self._dropout_counts),
            "round_indices": list(self._round_indices),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore S3-Loc state."""

        self.config = self._build_config(state_dict.get("config", self.config))
        self.num_clients = int(state_dict.get("num_clients", self.num_clients))
        self.group_size = int(state_dict.get("group_size", self.group_size))
        self.audit_rounds = int(state_dict.get("audit_rounds", self.audit_rounds))
        self._group_assignments = [list(group) for group in state_dict.get("group_assignments", [])]
        self._group_labels = [bool(value) for value in state_dict.get("group_labels", [])]
        self._group_scores = [float(value) for value in state_dict.get("group_scores", [])]
        self._dropout_counts = [int(value) for value in state_dict.get("dropout_counts", [])]
        self._round_indices = [int(value) for value in state_dict.get("round_indices", [])]

    def _build_config(self, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = {
            "method": "weighted_counting",
            "suspicious_topk": 1,
            "lasso_alpha": 0.05,
            "degenerate_threshold": 0.8,
            "contrastive_negative_weight": 0.25,
            "contrastive_negative_quantile": 0.9,
            "contrastive_positive_scale": 1.0,
            "pos_neg_contrast_enabled": True,
            "pos_neg_contrast_lambda": 0.9,
            "pos_neg_contrast_blend": 0.20,
            "pos_neg_contrast_min_positive_groups": 3,
            "pos_neg_contrast_min_negative_groups": 3,
            "seed": 1234,
        }
        if config is not None:
            merged.update(dict(config))
        return merged

    @staticmethod
    def _build_matrix_from_groups(group_assignments: Sequence[Sequence[int]], num_clients: int) -> np.ndarray:
        """Convert historical group membership lists into the binary sensing matrix A."""

        A_matrix = np.zeros((len(group_assignments), num_clients), dtype=np.float32)
        for row_index, group in enumerate(group_assignments):
            if len(group) == 0:
                continue
            client_indices = np.asarray(group, dtype=np.int64)
            A_matrix[row_index, client_indices] = 1.0
        return A_matrix

    @staticmethod
    def _prepare_observation_vector(
        b_g_list: Sequence[bool],
        designed_group_sizes: Sequence[float],
        dropout_counts: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Apply dropout-aware normalization to the binary anomaly labels.

        The raw CTS-Intent output used by S3-Loc is the binary abnormal-group
        indicator ``b_g``. When a group loses clients due to dropout, the raw
        group signal is scaled by ``|G| / |G'|`` as requested by the paper-level
        design, so positive abnormal groups keep comparable strength.
        """

        y = np.asarray(b_g_list, dtype=np.float32)
        designed_sizes = np.asarray(designed_group_sizes, dtype=np.float32)
        if dropout_counts is None:
            return y

        actual_sizes = np.asarray(dropout_counts, dtype=np.float32)
        if actual_sizes.shape[0] != y.shape[0]:
            raise ValueError("dropout_counts must have the same length as b_g_list.")
        safe_actual_sizes = np.maximum(actual_sizes, 1.0)
        scale = designed_sizes / safe_actual_sizes
        return y * scale

    @staticmethod
    def _prepare_weighted_signal(
        group_signal: Sequence[float],
        designed_group_sizes: Sequence[float],
        dropout_counts: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Convert continuous CTS scores into a non-negative localization weight."""

        signal = np.asarray(group_signal, dtype=np.float32)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if signal.ndim != 1:
            raise ValueError("group_signal must be a 1D sequence.")

        finite_signal = signal[np.isfinite(signal)]
        if finite_signal.size == 0:
            finite_signal = np.asarray([0.0], dtype=np.float32)

        center = float(np.median(finite_signal))
        abs_dev = np.abs(finite_signal - center)
        mad = float(np.median(abs_dev))
        sigma = max(1.4826 * mad, float(np.std(finite_signal)), 1e-6)
        weighted = np.maximum((signal - center) / sigma, 0.0)

        if dropout_counts is not None:
            designed_sizes = np.asarray(designed_group_sizes, dtype=np.float32)
            actual_sizes = np.asarray(dropout_counts, dtype=np.float32)
            if actual_sizes.shape[0] != weighted.shape[0]:
                raise ValueError("dropout_counts must have the same length as group_signal.")
            weighted = weighted * (designed_sizes / np.maximum(actual_sizes, 1.0))

        return weighted.astype(np.float32, copy=False), {
            "signal_center": center,
            "signal_scale": sigma,
        }

    @staticmethod
    def _prepare_contrastive_signal(
        b_g_list: Sequence[bool],
        group_signal: Sequence[float],
        designed_group_sizes: Sequence[float],
        dropout_counts: Optional[Sequence[int]] = None,
        negative_weight: float = 0.25,
        negative_quantile: float = 0.9,
        positive_scale: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Build a signed localization signal from anomalous and high-score normal groups."""

        weighted_signal, weight_stats = S3Loc._prepare_weighted_signal(
            group_signal=group_signal,
            designed_group_sizes=designed_group_sizes,
            dropout_counts=dropout_counts,
        )
        y_binary = np.asarray(b_g_list, dtype=np.float32)
        if y_binary.shape[0] != weighted_signal.shape[0]:
            raise ValueError("b_g_list must have the same length as group_signal.")

        positive_mask = y_binary > 0.5
        negative_mask = ~positive_mask

        negative_candidates = weighted_signal[negative_mask]
        negative_positive = negative_candidates[negative_candidates > 0.0]
        if negative_positive.size > 0:
            cutoff = float(np.quantile(negative_positive, np.clip(negative_quantile, 0.0, 1.0)))
            negative_component = np.maximum(weighted_signal - cutoff, 0.0) * negative_mask.astype(np.float32)
        else:
            cutoff = 0.0
            negative_component = np.zeros_like(weighted_signal)

        positive_component = weighted_signal * positive_mask.astype(np.float32)
        positive_active = positive_component[positive_component > 0.0]
        negative_active = negative_component[negative_component > 0.0]

        # Keep the positive evidence in its original CTS-residual scale and only
        # apply a light penalty to a very small set of high-score normal groups.
        signed_signal = (
            float(positive_scale) * positive_component
            - float(negative_weight) * negative_component
        ).astype(np.float32, copy=False)

        stats = dict(weight_stats)
        stats.update(
            {
                "negative_cutoff": cutoff,
                "positive_mass": float(np.sum(positive_component)),
                "negative_mass": float(np.sum(negative_component)),
                "positive_group_count": int(positive_active.size),
                "negative_group_count": int(negative_active.size),
                "positive_mean": float(np.mean(positive_active)) if positive_active.size > 0 else 0.0,
                "negative_mean": float(np.mean(negative_active)) if negative_active.size > 0 else 0.0,
                "positive_only_signal": positive_component.astype(np.float32, copy=False).tolist(),
                "fallback_positive_mass": float(np.sum(positive_component)),
            }
        )
        return signed_signal, stats

    @staticmethod
    def _apply_pos_neg_contrast(
        base_scores: np.ndarray,
        A_matrix: np.ndarray,
        b_g_list: Sequence[bool],
        contrast_lambda: float,
        blend: float,
        min_positive_groups: int,
        min_negative_groups: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Blend base scores with positive-vs-negative group contrast.

        This suppresses benign hub clients that appear frequently in both
        anomalous and normal groups.
        """

        base_scores = np.asarray(base_scores, dtype=np.float32)
        y_binary = np.asarray(b_g_list, dtype=np.float32) > 0.5
        num_groups = int(A_matrix.shape[0])
        pos_count = int(np.sum(y_binary))
        neg_count = int(num_groups - pos_count)
        min_positive_groups = max(int(min_positive_groups), 1)
        min_negative_groups = max(int(min_negative_groups), 1)

        stats: Dict[str, Any] = {
            "enabled": False,
            "reason": "insufficient_groups",
            "lambda": float(contrast_lambda),
            "blend": float(blend),
            "positive_group_count": pos_count,
            "negative_group_count": neg_count,
            "base_scale": 0.0,
            "contrast_scale": 0.0,
        }
        if pos_count < min_positive_groups or neg_count < min_negative_groups:
            return base_scores, stats

        pos_rate = np.asarray(A_matrix[y_binary].mean(axis=0), dtype=np.float32)
        neg_rate = np.asarray(A_matrix[~y_binary].mean(axis=0), dtype=np.float32)
        contrast = pos_rate - float(contrast_lambda) * neg_rate

        base_scale = float(np.std(base_scores))
        contrast_scale = float(np.std(contrast))
        if base_scale <= 1e-8 or contrast_scale <= 1e-8:
            stats["reason"] = "zero_variance"
            stats["base_scale"] = base_scale
            stats["contrast_scale"] = contrast_scale
            return base_scores, stats

        base_center = float(np.median(base_scores))
        contrast_center = float(np.median(contrast))
        base_z = (base_scores - base_center) / max(base_scale, 1e-8)
        contrast_z = (contrast - contrast_center) / max(contrast_scale, 1e-8)
        blend = float(min(max(blend, 0.0), 1.0))
        mixed = (1.0 - blend) * base_z + blend * contrast_z

        stats.update(
            {
                "enabled": True,
                "reason": "applied",
                "base_center": base_center,
                "base_scale": base_scale,
                "contrast_center": contrast_center,
                "contrast_scale": contrast_scale,
                "pos_rate_mean": float(np.mean(pos_rate)),
                "neg_rate_mean": float(np.mean(neg_rate)),
                "contrast_mean": float(np.mean(contrast)),
                "contrast_std": float(np.std(contrast)),
                "contrast_max": float(np.max(contrast)),
                "contrast_min": float(np.min(contrast)),
                "pos_rate": pos_rate.astype(np.float32, copy=False),
                "neg_rate": neg_rate.astype(np.float32, copy=False),
                "contrast_scores": contrast.astype(np.float32, copy=False),
            }
        )
        return mixed.astype(np.float32, copy=False), stats


def localize_s3_clients(
    A_matrix: np.ndarray,
    b_g_list: Sequence[bool],
    s: int,
    method: str = "counting",
    group_signal: Optional[Sequence[float]] = None,
    dropout_counts: Optional[Sequence[int]] = None,
    lasso_alpha: float = 0.05,
    degenerate_threshold: float = 0.8,
    contrastive_negative_weight: float = 0.25,
    contrastive_negative_quantile: float = 0.9,
    contrastive_positive_scale: float = 1.0,
    pos_neg_contrast_enabled: bool = True,
    pos_neg_contrast_lambda: float = 0.9,
    pos_neg_contrast_blend: float = 0.20,
    pos_neg_contrast_min_positive_groups: int = 3,
    pos_neg_contrast_min_negative_groups: int = 3,
    candidate_client_ids: Optional[Sequence[int]] = None,
    return_details: bool = False,
) -> Tuple[List[int], float, bool] | Tuple[List[int], float, bool, Dict[int, float], Dict[str, Any]]:
    """Localize suspicious clients from secure-aggregation group observations.

    Args:
        A_matrix: Binary sensing matrix of shape ``[num_groups, num_clients]``.
            ``A[g, i] = 1`` means client ``i`` participated in group ``g``.
        b_g_list: Binary abnormal-group decisions returned by CTS-Intent.
        s: Expected number of malicious clients. The top-``s`` entries are
            returned when localization is reliable.
        method: ``"counting"`` for binary peeling/counting, ``"weighted_counting"``
            for CTS-weighted counting, ``"contrastive_weighted_counting"`` for a
            lightly penalized positive-vs-normal variant, or ``"lasso"`` for sparse
            recovery.
        group_signal: Optional continuous CTS score for each group. Used by the
            weighted localization variants.
        dropout_counts: Actual surviving members for each group. If provided,
            positive abnormal groups are scaled by ``|G| / |G'|``.
        lasso_alpha: Regularization strength used by sklearn ``Lasso``.
        degenerate_threshold: If the abnormal group ratio exceeds this value,
            localization is considered unreliable.
        contrastive_negative_weight: Penalty weight assigned to high-score
            normal groups in contrastive localization.
        contrastive_negative_quantile: Quantile used to suppress low-score
            normal groups before negative weighting.
        contrastive_positive_scale: Scaling factor for abnormal-group evidence
            in contrastive localization.
        pos_neg_contrast_enabled: Whether to blend client scores with
            positive-vs-negative group participation contrast.
        pos_neg_contrast_lambda: Penalty coefficient for normal-group
            participation in the contrast term.
        pos_neg_contrast_blend: Blending ratio between base score and
            positive-vs-negative contrast score.
        pos_neg_contrast_min_positive_groups: Minimum anomalous-group count
            required before applying the contrast blend.
        pos_neg_contrast_min_negative_groups: Minimum normal-group count
            required before applying the contrast blend.
        candidate_client_ids: Optional subset of client ids to score and rank.
        return_details: Internal flag used by the class wrapper to also return
            raw score dictionaries and diagnostics.

    Returns:
        By default returns ``(suspicious_set, p_hat, degenerate)``.
        If ``return_details=True``, also returns ``client_scores`` and
        ``aux_stats``.
    """

    A_matrix = np.asarray(A_matrix, dtype=np.float32)
    if A_matrix.ndim != 2:
        raise ValueError("A_matrix must be a 2D numpy array.")

    num_groups, num_clients = A_matrix.shape
    y_binary = np.asarray(b_g_list, dtype=np.float32)
    if y_binary.shape[0] != num_groups:
        raise ValueError("b_g_list length must match the number of rows in A_matrix.")

    if candidate_client_ids is None:
        candidate_client_ids = list(range(num_clients))
    else:
        candidate_client_ids = [int(client_id) for client_id in candidate_client_ids]

    designed_group_sizes = A_matrix.sum(axis=1)
    y_corrected = S3Loc._prepare_observation_vector(
        b_g_list=b_g_list,
        designed_group_sizes=designed_group_sizes,
        dropout_counts=dropout_counts,
    )

    # The degenerate-mode decision follows the specification exactly: it is based
    # on the raw binary anomaly ratio, not the dropout-corrected surrogate.
    p_hat = float(y_binary.mean()) if num_groups > 0 else 0.0
    degenerate = bool(p_hat > float(degenerate_threshold))

    if degenerate:
        suspicious_set: List[int] = []
        client_scores = {client_id: 0.0 for client_id in candidate_client_ids}
        aux_stats = {
            "method": "degenerate",
            "raw_positive_ratio": p_hat,
            "corrected_y": y_corrected.tolist(),
            "selected_scores": {},
            "weighted_y": [],
            "signal_center": 0.0,
            "signal_scale": 1.0,
        }
        if return_details:
            return suspicious_set, p_hat, degenerate, client_scores, aux_stats
        return suspicious_set, p_hat, degenerate

    s = max(0, min(int(s), len(candidate_client_ids)))
    method = method.lower()

    if method == "counting":
        # Counting/peeling score: each abnormal group votes for every client that
        # appeared in that group. Dropout correction turns the vote weight into a
        # soft count when the actual group size shrinks.
        raw_scores = A_matrix.T @ y_corrected
        weighted_y = y_corrected
        weight_stats = {"signal_center": 0.0, "signal_scale": 1.0}
    elif method == "weighted_counting":
        if group_signal is None:
            raise ValueError("group_signal is required for method='weighted_counting'.")
        weighted_y, weight_stats = S3Loc._prepare_weighted_signal(
            group_signal=group_signal,
            designed_group_sizes=designed_group_sizes,
            dropout_counts=dropout_counts,
        )
        participation = np.maximum(A_matrix.sum(axis=0), 1.0)
        raw_scores = (A_matrix.T @ weighted_y) / np.sqrt(participation)
    elif method == "contrastive_weighted_counting":
        if group_signal is None:
            raise ValueError("group_signal is required for method='contrastive_weighted_counting'.")
        weighted_y, weight_stats = S3Loc._prepare_contrastive_signal(
            b_g_list=b_g_list,
            group_signal=group_signal,
            designed_group_sizes=designed_group_sizes,
            dropout_counts=dropout_counts,
            negative_weight=float(contrastive_negative_weight),
            negative_quantile=float(contrastive_negative_quantile),
            positive_scale=float(contrastive_positive_scale),
        )
        participation = np.maximum(A_matrix.sum(axis=0), 1.0)
        raw_scores = (A_matrix.T @ weighted_y) / np.sqrt(participation)
        if float(np.max(raw_scores)) <= 1e-8 and float(weight_stats.get("fallback_positive_mass", 0.0)) > 0.0:
            positive_only_signal = np.asarray(weight_stats.get("positive_only_signal", []), dtype=np.float32)
            if positive_only_signal.shape == weighted_y.shape:
                raw_scores = (A_matrix.T @ positive_only_signal) / np.sqrt(participation)
    elif method == "lasso":
        if Lasso is None:
            raise ImportError("scikit-learn is required for method='lasso'.")
        # The sparse vector x is constrained to be non-negative because the
        # latent 'backdoor strength' cannot be negative.
        model = Lasso(
            alpha=float(lasso_alpha),
            fit_intercept=False,
            positive=True,
            max_iter=10000,
            tol=1e-4,
            random_state=0,
        )
        model.fit(A_matrix, y_corrected)
        raw_scores = model.coef_.astype(np.float32, copy=False)
        weighted_y = y_corrected
        weight_stats = {"signal_center": 0.0, "signal_scale": 1.0}
    else:
        raise ValueError(
            "method must be 'counting', 'weighted_counting', "
            "'contrastive_weighted_counting', or 'lasso'."
        )

    pos_neg_stats: Dict[str, Any] = {
        "enabled": False,
        "reason": "disabled",
        "lambda": float(pos_neg_contrast_lambda),
        "blend": float(pos_neg_contrast_blend),
        "positive_group_count": int(np.sum(y_binary > 0.5)),
        "negative_group_count": int(num_groups - np.sum(y_binary > 0.5)),
    }
    if bool(pos_neg_contrast_enabled):
        raw_scores, pos_neg_stats = S3Loc._apply_pos_neg_contrast(
            base_scores=raw_scores,
            A_matrix=A_matrix,
            b_g_list=b_g_list,
            contrast_lambda=float(pos_neg_contrast_lambda),
            blend=float(pos_neg_contrast_blend),
            min_positive_groups=int(pos_neg_contrast_min_positive_groups),
            min_negative_groups=int(pos_neg_contrast_min_negative_groups),
        )

    min_score = float(np.min(raw_scores)) if raw_scores.size > 0 else 0.0
    if min_score < 0.0:
        raw_scores = raw_scores - min_score

    client_scores = {client_id: float(raw_scores[client_id]) for client_id in candidate_client_ids}
    if s == 0:
        suspicious_set = []
    else:
        ordered_candidates = sorted(
            candidate_client_ids,
            key=lambda client_id: (-client_scores[client_id], client_id),
        )
        suspicious_set = ordered_candidates[:s]

    selected_pos_rate: Dict[int, float] = {}
    selected_neg_rate: Dict[int, float] = {}
    selected_contrast_score: Dict[int, float] = {}
    if bool(pos_neg_stats.get("enabled", False)):
        pos_rate = np.asarray(pos_neg_stats.get("pos_rate", []), dtype=np.float32)
        neg_rate = np.asarray(pos_neg_stats.get("neg_rate", []), dtype=np.float32)
        contrast_scores = np.asarray(pos_neg_stats.get("contrast_scores", []), dtype=np.float32)
        if (
            pos_rate.shape[0] == num_clients
            and neg_rate.shape[0] == num_clients
            and contrast_scores.shape[0] == num_clients
        ):
            selected_pos_rate = {
                int(client_id): float(pos_rate[int(client_id)])
                for client_id in suspicious_set
            }
            selected_neg_rate = {
                int(client_id): float(neg_rate[int(client_id)])
                for client_id in suspicious_set
            }
            selected_contrast_score = {
                int(client_id): float(contrast_scores[int(client_id)])
                for client_id in suspicious_set
            }

    aux_stats = {
        "method": method,
        "raw_positive_ratio": p_hat,
        "corrected_y": y_corrected.tolist(),
        "weighted_y": weighted_y.tolist(),
        "signal_center": float(weight_stats["signal_center"]),
        "signal_scale": float(weight_stats["signal_scale"]),
        "negative_cutoff": float(weight_stats.get("negative_cutoff", 0.0)),
        "positive_mass": float(weight_stats.get("positive_mass", 0.0)),
        "negative_mass": float(weight_stats.get("negative_mass", 0.0)),
        "positive_group_count": int(weight_stats.get("positive_group_count", 0)),
        "negative_group_count": int(weight_stats.get("negative_group_count", 0)),
        "positive_mean": float(weight_stats.get("positive_mean", 0.0)),
        "negative_mean": float(weight_stats.get("negative_mean", 0.0)),
        "pos_neg_contrast_enabled": bool(pos_neg_stats.get("enabled", False)),
        "pos_neg_contrast_reason": str(pos_neg_stats.get("reason", "none")),
        "pos_neg_contrast_lambda": float(pos_neg_stats.get("lambda", pos_neg_contrast_lambda)),
        "pos_neg_contrast_blend": float(pos_neg_stats.get("blend", pos_neg_contrast_blend)),
        "pos_neg_positive_group_count": int(pos_neg_stats.get("positive_group_count", 0)),
        "pos_neg_negative_group_count": int(pos_neg_stats.get("negative_group_count", 0)),
        "pos_neg_base_scale": float(pos_neg_stats.get("base_scale", 0.0)),
        "pos_neg_contrast_scale": float(pos_neg_stats.get("contrast_scale", 0.0)),
        "selected_pos_rate": selected_pos_rate,
        "selected_neg_rate": selected_neg_rate,
        "selected_contrast_scores": selected_contrast_score,
        "selected_scores": {client_id: client_scores[client_id] for client_id in suspicious_set},
    }
    if return_details:
        return suspicious_set, p_hat, degenerate, client_scores, aux_stats
    return suspicious_set, p_hat, degenerate


# Backward-friendly alias matching the requested function-style API.
run_s3_loc = localize_s3_clients
