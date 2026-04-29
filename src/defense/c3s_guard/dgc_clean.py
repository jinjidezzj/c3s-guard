"""DGC-Clean: detection-driven suspicious subspace cleaning.

This module implements the third stage of C3S-Guard.

It supports three projection modes:
1) parameter_grad: estimate a suspicious gradient subspace in parameter space
   and remove its component from the aggregated update.
2) feature_fc: estimate a suspicious subspace in penultimate feature space from
   trigger-vs-clean activation differences, then apply closed-form projection on
   classifier weights W <- W (I - rho * V V^T).
3) group_diff: estimate suspicious parameter-space directions from anomaly-vs-normal
   audit-group update differences, then project every round update continuously.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import Module

from defense.c3s_guard.utils import DGCCleanResult, ModelUpdate


def build_backdoor_subspace_from_group_diff(
    anomalous_group_updates: list,
    normal_group_updates: list,
    m: int = 10,
    min_anomalous: int = 2,
    min_energy_ratio: float = 0.3,
    device: str = "cuda",
) -> tuple[Optional[torch.Tensor], dict]:
    """Build B_tilde from anomaly-vs-normal group-update differences via SVD."""

    anomalous_count = int(len(anomalous_group_updates))
    normal_count = int(len(normal_group_updates))
    if anomalous_count < int(min_anomalous):
        return None, {"reason": "too_few_anomalous", "count": anomalous_count}
    if normal_count == 0:
        return None, {"reason": "no_normal_groups"}

    target_device = torch.device(device)
    anomalous_tensors = [
        torch.as_tensor(vector, device=target_device).flatten().float()
        for vector in anomalous_group_updates
    ]
    normal_tensors = [
        torch.as_tensor(vector, device=target_device).flatten().float()
        for vector in normal_group_updates
    ]

    if len(anomalous_tensors) == 0:
        return None, {"reason": "too_few_anomalous", "count": 0}
    if len(normal_tensors) == 0:
        return None, {"reason": "no_normal_groups"}

    dim = int(anomalous_tensors[0].numel())
    if any(int(vector.numel()) != dim for vector in anomalous_tensors + normal_tensors):
        return None, {"reason": "dimension_mismatch", "dim": dim}

    normal_stack = torch.stack(normal_tensors, dim=0)  # [k_normal, d]
    mu_normal = normal_stack.mean(dim=0)  # [d]

    diff_vectors = []
    for delta_bad in anomalous_tensors:
        diff_vectors.append(delta_bad - mu_normal)
    D = torch.stack(diff_vectors, dim=1)  # [d, k_bad]

    col_norms = D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    D_normalized = D / col_norms

    U, S, _ = torch.linalg.svd(D_normalized, full_matrices=False)
    actual_rank = int(min(max(int(m), 1), int(S.shape[0])))
    B_tilde = U[:, :actual_rank].contiguous()

    total_energy = (S**2).sum().clamp(min=1e-10)
    kept_energy = (S[:actual_rank] ** 2).sum()
    energy_ratio = float((kept_energy / total_energy).item())

    diagnostics = {
        "method": "group_diff_svd",
        "num_anomalous": anomalous_count,
        "num_normal": normal_count,
        "actual_rank": actual_rank,
        "energy_ratio": energy_ratio,
        "singular_values": [float(value) for value in S[:actual_rank].detach().cpu().tolist()],
        "status": "success",
    }
    if energy_ratio < float(min_energy_ratio):
        diagnostics["status"] = "low_energy"
        diagnostics["warning"] = (
            f"Energy ratio {energy_ratio:.3f} below threshold {float(min_energy_ratio):.3f}"
        )

    return B_tilde.to(target_device), diagnostics


class DGCClean:
    """Project aggregated updates away from a suspicious gradient subspace."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Module,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the DGC-Clean module.

        Args:
            config: Dictionary of DGC-Clean hyper-parameters.
            model: BackdoorBench-compatible classification model.
            device: Device used for subspace estimation and cleaning.
        """

        self.config = self._build_config(config)
        self.device = self._resolve_device(device, model)
        self.model = model.to(self.device)
        self.model.eval()

        # EMA teacher persists across calls so the caller can reuse the same
        # DGCClean instance over multiple audit cycles.
        self.ema_model = deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.config["seed"]))

        self._param_shapes = [parameter.shape for parameter in self.model.parameters()]
        self._param_numels = [parameter.numel() for parameter in self.model.parameters()]
        self._param_names = [name for name, _ in self.model.named_parameters()]
        self._param_slices = self._build_param_slices()
        self._param_slice_by_name = {str(item["name"]): item for item in self._param_slices}
        self._buffer_names = [name for name, _ in self.model.named_buffers()]
        self._last_subspace: Optional[Dict[str, Any]] = None

    def fit_suspicious_subspace(
        self,
        suspicious_updates: Sequence[ModelUpdate],
        reference_update: Optional[ModelUpdate] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate the suspicious gradient subspace used for cleaning.

        Preferred mode: estimate the subspace from ``m`` samples of the backdoor
        bad-behavior loss gradients, using ``y_star`` and ``x_syn`` from
        ``metadata``. Fallback mode: if those are unavailable, build the subspace
        directly from the supplied suspicious update vectors.

        Args:
            suspicious_updates: Updates associated with suspicious groups or
                clients. Used as a fallback source of suspicious directions.
            reference_update: Optional clean reference update. Currently unused,
                but kept for interface compatibility.
            metadata: Optional extra information. Supported keys include
                ``y_star``, ``x_syn``, ``model``, ``num_gradient_samples`` and
                ``gradient_probe_batch_size``.

        Returns:
            A dictionary describing the fitted subspace representation. The key
            ``basis`` stores an orthonormal basis ``Q`` with shape ``[d, r]``.
        """

        metadata = dict(metadata or {})
        y_star = metadata.get("y_star")
        x_syn = metadata.get("x_syn")
        num_gradient_samples = int(metadata.get("num_gradient_samples", self.config["num_gradient_samples"]))
        probe_batch_size = int(
            metadata.get("gradient_probe_batch_size", self.config["gradient_probe_batch_size"])
        )

        basis_source: List[torch.Tensor] = []
        sample_mode = "updates"
        projection_scope = str(metadata.get("projection_scope", self.config.get("projection_scope", "full"))).lower()

        if y_star is not None and x_syn is not None:
            x_syn_tensor = self._prepare_probe_tensor(x_syn)
            probe_batches = self._sample_probe_batches(
                x_syn=x_syn_tensor,
                num_batches=num_gradient_samples,
                batch_size=probe_batch_size,
            )
            sample_mode = "bd_gradients"
            for probe_batch in probe_batches:
                grad_vector = self._compute_bd_gradient(
                    model=self.model,
                    x_batch=probe_batch,
                    y_star=int(y_star),
                )
                if grad_vector.norm(p=2) > float(self.config["gradient_eps"]):
                    basis_source.append(grad_vector)
        else:
            for update in suspicious_updates:
                update_vector = self._flatten_update(update)
                if update_vector.norm(p=2) > float(self.config["gradient_eps"]):
                    basis_source.append(update_vector)

        if len(basis_source) > 0 and projection_scope != "full":
            mask = self._build_projection_mask(
                scope=projection_scope,
                dim=int(basis_source[0].numel()),
                dtype=basis_source[0].dtype,
            )
            basis_source = [vector * mask for vector in basis_source]
            basis_source = [
                vector
                for vector in basis_source
                if vector.norm(p=2) > float(self.config["gradient_eps"])
            ]

        if len(basis_source) == 0:
            flat_dim = int(sum(self._param_numels))
            basis = torch.empty(flat_dim, 0, device=self.device, dtype=self._parameter_dtype())
            subspace = {
                "basis": basis,
                "rank": 0,
                "num_vectors": 0,
                "sample_mode": sample_mode,
                "projection_scope": projection_scope,
            }
            self._last_subspace = subspace
            return subspace

        # Stack as [m, d]. The transpose view has shape [d, m], so reduced QR
        # returns an orthonormal basis with cost O(d * m), which is tractable for
        # small ``m`` and avoids any dense d-by-d matrix construction.
        B_transposed = torch.stack(basis_source, dim=0)  # [m, d]
        q_matrix, _ = torch.linalg.qr(B_transposed.t(), mode="reduced")

        # Remove numerically tiny directions to keep the projection stable.
        keep_mask = []
        for column_index in range(q_matrix.shape[1]):
            column_norm = q_matrix[:, column_index].norm(p=2)
            keep_mask.append(bool(column_norm > float(self.config["gradient_eps"])))
        if any(keep_mask):
            basis = q_matrix[:, keep_mask].contiguous()
        else:
            basis = torch.empty(q_matrix.shape[0], 0, device=q_matrix.device, dtype=q_matrix.dtype)

        subspace = {
            "basis": basis.detach(),
            "rank": int(basis.shape[1]),
            "num_vectors": len(basis_source),
            "sample_mode": sample_mode,
            "projection_scope": projection_scope,
        }
        self._last_subspace = subspace
        return subspace

    def clean_aggregated_update(
        self,
        aggregated_update: ModelUpdate,
        suspicious_subspace: Optional[Mapping[str, Any]] = None,
        reference_update: Optional[ModelUpdate] = None,
        round_idx: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DGCCleanResult:
        """Remove suspicious subspace components from an aggregated update.

        This method performs the full DGC-Clean pipeline:
        1. Build a low-rank suspicious subspace.
        2. Project the aggregated update away from that subspace.
        3. Apply the projected update to the current model.
        4. Run ``E_d`` epochs of EMA-teacher distillation plus the adaptive
           backdoor loss.
        5. Return the final cleaned update relative to the original model.
        """

        metadata = dict(metadata or {})
        y_star = metadata.get("y_star")
        p_hat = float(metadata.get("p_hat", 0.0))
        x_syn = metadata.get("x_syn")
        effective_rho = float(metadata.get("effective_rho", p_hat))
        effective_rho = float(max(0.0, min(1.0, effective_rho)))
        cleaning_tier = str(metadata.get("cleaning_tier", "none"))
        distill_steps_override = metadata.get("distill_steps_override")
        projection_strength = metadata.get("projection_strength")
        enable_projection = bool(self.config.get("enable_projection", True))
        enable_distillation = bool(self.config.get("enable_distillation", True))
        projection_scope = str(metadata.get("projection_scope", self.config.get("projection_scope", "full"))).lower()
        subspace_method = str(
            metadata.get("subspace_method", self.config.get("subspace_method", "parameter_grad"))
        ).lower()

        if subspace_method in {"feature_fc", "feature", "activation_feature", "feature_activation"}:
            return self._clean_aggregated_update_feature_fc(
                aggregated_update=aggregated_update,
                round_idx=round_idx,
                metadata=metadata,
                enable_projection=enable_projection,
                enable_distillation=enable_distillation,
                p_hat=p_hat,
                effective_rho=effective_rho,
                cleaning_tier=cleaning_tier,
                projection_strength=projection_strength,
            )
        if subspace_method in {"group_diff", "group_diff_svd"}:
            # Expect a pre-built suspicious_subspace supplied by audit-group difference SVD.
            if suspicious_subspace is None:
                suspicious_subspace = {
                    "basis": torch.empty(0, 0, device=self.device, dtype=self._parameter_dtype()),
                    "rank": 0,
                    "num_vectors": 0,
                    "sample_mode": "group_diff_svd_unavailable",
                }
            metadata = dict(metadata)
            metadata["subspace_method"] = "group_diff"

        if y_star is None:
            raise ValueError("metadata must provide y_star for DGC-Clean.")
        if x_syn is None:
            raise ValueError("metadata must provide x_syn for DGC-Clean.")

        delta_flat = self._flatten_update(aggregated_update)
        if (not enable_projection):
            suspicious_subspace = {
                "basis": torch.empty(delta_flat.numel(), 0, device=self.device, dtype=delta_flat.dtype),
                "rank": 0,
                "num_vectors": 0,
                "sample_mode": "projection_disabled",
            }
        elif suspicious_subspace is None:
            suspicious_subspace = self.fit_suspicious_subspace(
                suspicious_updates=[],
                reference_update=reference_update,
                metadata=metadata,
            )

        basis = suspicious_subspace.get("basis")
        if basis is None:
            basis = torch.empty(delta_flat.numel(), 0, device=self.device, dtype=delta_flat.dtype)
        else:
            basis = basis.to(self.device, dtype=delta_flat.dtype)

        scope_mask = self._build_projection_mask(
            scope=projection_scope,
            dim=int(delta_flat.numel()),
            dtype=delta_flat.dtype,
        )
        if scope_mask.numel() != delta_flat.numel():
            scope_mask = torch.ones_like(delta_flat)
            projection_scope = "full"
        scoped_dim = int((scope_mask > 0).sum().item())
        if projection_scope != "full":
            delta_for_projection = delta_flat * scope_mask
            if basis.numel() > 0:
                basis = basis * scope_mask.unsqueeze(1)
        else:
            delta_for_projection = delta_flat

        if (not enable_projection) or basis.numel() == 0:
            projection = torch.zeros_like(delta_flat)
            projected_delta_flat = delta_flat.clone()
        else:
            coefficients = basis.t().matmul(delta_for_projection)
            projection = basis.matmul(coefficients)
            if projection_scope != "full":
                projection = projection * scope_mask
            projected_delta_flat = delta_flat - projection

        projection_norm_before = float(projection.norm(p=2).item())
        projection_norm_sq = float(projection.pow(2).sum().item())
        delta_norm_sq = float(delta_flat.pow(2).sum().item())
        delta_scope_norm_sq = float((delta_flat * scope_mask).pow(2).sum().item())
        removed_energy = projection_norm_sq / max(delta_norm_sq, float(self.config["gradient_eps"]))
        removed_energy_scope = projection_norm_sq / max(delta_scope_norm_sq, float(self.config["gradient_eps"]))

        lambda_t = self._compute_lambda_t(
            p_hat=p_hat,
            cleaning_confidence=float(metadata.get("cleaning_confidence", 1.0)),
            localization_reliable=bool(metadata.get("localization_reliable", True)),
            effective_rho=effective_rho,
        )
        projection_scale = self._compute_projection_scale(lambda_t)
        if projection_strength is None:
            rho_reference = max(float(self.config.get("rho_reference", 0.20)), 1e-6)
            projection_strength = float(max(0.0, min(1.0, effective_rho / rho_reference)))
        else:
            projection_strength = float(max(0.0, min(1.0, float(projection_strength))))
        if enable_projection and basis.numel() != 0:
            projection = projection * projection_scale * projection_strength
            if projection_scope != "full":
                projection = projection * scope_mask
            projected_delta_flat = delta_flat - projection
            projection_norm_sq = float(projection.pow(2).sum().item())
            removed_energy = projection_norm_sq / max(delta_norm_sq, float(self.config["gradient_eps"]))
            removed_energy_scope = projection_norm_sq / max(delta_scope_norm_sq, float(self.config["gradient_eps"]))
        projection_norm_after = float(projection.norm(p=2).item())

        projected_delta = self._unflatten_like_model(projected_delta_flat)
        student_model = self.apply_cleaned_update(self.model, projected_delta, inplace=False)

        if enable_distillation:
            distill_stats = self._run_online_distillation(
                student_model=student_model,
                y_star=int(y_star),
                p_hat=p_hat,
                x_syn=self._prepare_probe_tensor(x_syn),
                lambda_t=lambda_t,
                distill_steps_override=distill_steps_override,
                base_update_norm=float(delta_flat.norm(p=2).item()),
            )
        else:
            self._update_ema_teacher(student_model)
            distill_stats = {
                "avg_total_loss": 0.0,
                "avg_kd_loss": 0.0,
                "avg_bd_loss": 0.0,
                "lambda_t": float(lambda_t),
                "p_hat": float(p_hat),
                "enabled": False,
                "distill_steps": 0,
            }

        base_params_flat = self._flatten_model_parameters(self.model)
        student_params_flat = self._flatten_model_parameters(student_model)
        final_delta_flat = student_params_flat - base_params_flat
        final_delta = self._unflatten_like_model(final_delta_flat)

        result = DGCCleanResult(
            round_idx=-1 if round_idx is None else round_idx,
            cleaned_update=final_delta,
            removed_energy=removed_energy,
            suspicious_subspace_rank=int(basis.shape[1]),
            aux_stats={
                "lambda_t": lambda_t,
                "projection_norm": float(projection.norm(p=2).item()),
                "projection_norm_before": float(projection_norm_before),
                "projection_norm_after": float(projection_norm_after),
                "projection_scale": float(projection_scale),
                "projection_strength": float(projection_strength),
                "projection_scope": str(projection_scope),
                "projection_scope_dim": int(scoped_dim),
                "delta_norm": float(delta_flat.norm(p=2).item()),
                "delta_scope_norm": float((delta_flat * scope_mask).norm(p=2).item()),
                "clean_model_delta_norm": float(final_delta_flat.norm(p=2).item()),
                "removed_energy_scope": float(removed_energy_scope),
                "subspace_mode": suspicious_subspace.get("sample_mode", "unknown"),
                "enable_projection": bool(enable_projection),
                "enable_distillation": bool(enable_distillation),
                "effective_rho": float(effective_rho),
                "cleaning_tier": str(cleaning_tier),
                "distill_stats": distill_stats,
            },
        )
        return result

    def _clean_aggregated_update_feature_fc(
        self,
        aggregated_update: ModelUpdate,
        round_idx: Optional[int],
        metadata: Mapping[str, Any],
        enable_projection: bool,
        enable_distillation: bool,
        p_hat: float,
        effective_rho: float,
        cleaning_tier: str,
        projection_strength: Optional[float],
    ) -> DGCCleanResult:
        x_syn = metadata.get("x_syn")
        if x_syn is None:
            raise ValueError("metadata must provide x_syn for feature-space DGC-Clean.")
        x_syn_tensor = self._prepare_probe_tensor(x_syn)

        y_star = metadata.get("y_star")
        lambda_t = self._compute_lambda_t(
            p_hat=p_hat,
            cleaning_confidence=float(metadata.get("cleaning_confidence", 1.0)),
            localization_reliable=bool(metadata.get("localization_reliable", True)),
            effective_rho=effective_rho,
        )
        projection_scale = self._compute_projection_scale(lambda_t)
        if projection_strength is None:
            rho_reference = max(float(self.config.get("rho_reference", 0.20)), 1e-6)
            projection_strength = float(max(0.0, min(1.0, effective_rho / rho_reference)))
        else:
            projection_strength = float(max(0.0, min(1.0, float(projection_strength))))

        # Start from the standard aggregated-update model, then apply an extra
        # classifier-head correction in feature space.
        base_params_flat = self._flatten_model_parameters(self.model)
        student_model = self.apply_cleaned_update(self.model, aggregated_update, inplace=False)
        delta_after_agg_flat = self._flatten_model_parameters(student_model) - base_params_flat

        feature_subspace = {
            "basis": torch.empty(0, 0, device=self.device, dtype=self._parameter_dtype()),
            "rank": 0,
            "num_vectors": 0,
            "sample_mode": "activation_feature_fc",
            "feature_dim": 0,
            "trigger_names": [],
        }
        projection_success = False
        projection_reason = "disabled"
        projection_dim = 0
        classifier_param_names: Set[str] = set()

        projection_rho = 0.0
        if enable_projection:
            projection_rho = float(max(0.0, min(1.0, projection_scale * projection_strength)))

        if enable_projection and projection_rho > 0.0:
            feature_subspace = self._fit_activation_feature_subspace(
                model=student_model,
                x_syn=x_syn_tensor,
            )
            basis = feature_subspace["basis"]
            if basis.numel() > 0 and int(feature_subspace.get("rank", 0)) > 0:
                projection_stats = self._apply_classifier_feature_projection(
                    model=student_model,
                    basis=basis,
                    rho=projection_rho,
                )
                projection_success = bool(projection_stats.get("applied", False))
                projection_reason = str(projection_stats.get("reason", "unknown"))
                projection_dim = int(projection_stats.get("feature_dim", 0))
                classifier_param_names = set(projection_stats.get("trainable_param_names", []))
            else:
                projection_reason = "empty_feature_basis"
        elif enable_projection:
            projection_reason = "zero_projection_strength"

        delta_after_projection_flat = self._flatten_model_parameters(student_model) - base_params_flat
        projection_vector = delta_after_agg_flat - delta_after_projection_flat
        projection_norm_after = float(projection_vector.norm(p=2).item())
        projection_norm_before = projection_norm_after / max(float(projection_rho), 1e-12) if projection_rho > 0 else 0.0

        delta_norm_sq = float(delta_after_agg_flat.pow(2).sum().item())
        projection_norm_sq = float(projection_vector.pow(2).sum().item())
        classifier_scope_mask = self._build_projection_mask(
            scope="classifier",
            dim=int(delta_after_agg_flat.numel()),
            dtype=delta_after_agg_flat.dtype,
        )
        delta_scope_norm_sq = float((delta_after_agg_flat * classifier_scope_mask).pow(2).sum().item())
        removed_energy = projection_norm_sq / max(delta_norm_sq, float(self.config["gradient_eps"]))
        removed_energy_scope = projection_norm_sq / max(delta_scope_norm_sq, float(self.config["gradient_eps"]))

        feature_disable_distill = bool(self.config.get("feature_space_disable_distillation", True))
        enable_distillation_effective = bool(enable_distillation) and (not feature_disable_distill)
        distill_classifier_only = bool(self.config.get("feature_space_distill_classifier_only", True))
        trainable_parameter_names = classifier_param_names if distill_classifier_only else None

        if enable_distillation_effective:
            if y_star is None:
                raise ValueError("metadata must provide y_star when distillation is enabled.")
            distill_stats = self._run_online_distillation(
                student_model=student_model,
                y_star=int(y_star),
                p_hat=p_hat,
                x_syn=x_syn_tensor,
                lambda_t=lambda_t,
                distill_steps_override=metadata.get("distill_steps_override"),
                trainable_parameter_names=trainable_parameter_names,
                base_update_norm=float(delta_after_agg_flat.norm(p=2).item()),
            )
        else:
            self._update_ema_teacher(student_model)
            distill_stats = {
                "avg_total_loss": 0.0,
                "avg_kd_loss": 0.0,
                "avg_bd_loss": 0.0,
                "lambda_t": float(lambda_t),
                "p_hat": float(p_hat),
                "enabled": False,
                "distill_steps": 0,
            }

        final_delta_flat = self._flatten_model_parameters(student_model) - base_params_flat
        final_delta = self._unflatten_like_model(final_delta_flat)

        basis = feature_subspace["basis"]
        result = DGCCleanResult(
            round_idx=-1 if round_idx is None else round_idx,
            cleaned_update=final_delta,
            removed_energy=removed_energy,
            suspicious_subspace_rank=int(feature_subspace.get("rank", int(basis.shape[1]) if basis.numel() > 0 else 0)),
            aux_stats={
                "lambda_t": float(lambda_t),
                "projection_norm": float(projection_norm_after),
                "projection_norm_before": float(projection_norm_before),
                "projection_norm_after": float(projection_norm_after),
                "projection_scale": float(projection_scale),
                "projection_strength": float(projection_strength),
                "projection_scope": "classifier",
                "projection_scope_dim": int((classifier_scope_mask > 0).sum().item()),
                "delta_norm": float(delta_after_agg_flat.norm(p=2).item()),
                "delta_scope_norm": float((delta_after_agg_flat * classifier_scope_mask).norm(p=2).item()),
                "clean_model_delta_norm": float(final_delta_flat.norm(p=2).item()),
                "removed_energy_scope": float(removed_energy_scope),
                "subspace_mode": "activation_feature_fc",
                "enable_projection": bool(enable_projection),
                "enable_distillation": bool(enable_distillation_effective),
                "effective_rho": float(effective_rho),
                "cleaning_tier": str(cleaning_tier),
                "projection_applied": bool(projection_success),
                "projection_reason": str(projection_reason),
                "feature_projection_rho": float(projection_rho),
                "feature_subspace_rank": int(feature_subspace.get("rank", 0)),
                "feature_subspace_dim": int(feature_subspace.get("feature_dim", 0)),
                "feature_subspace_vectors": int(feature_subspace.get("num_vectors", 0)),
                "feature_trigger_names": list(feature_subspace.get("trigger_names", [])),
                "feature_projection_dim": int(projection_dim),
                "feature_distill_classifier_only": bool(distill_classifier_only),
                "distill_stats": distill_stats,
            },
        )
        self._last_subspace = feature_subspace
        return result

    def apply_cleaned_update(
        self,
        model: Module,
        cleaned_update: ModelUpdate,
        inplace: bool = False,
    ) -> Module:
        """Apply a cleaned update to a model instance."""

        target_model = model if inplace else deepcopy(model)
        target_model.to(self.device)
        update_list = self._coerce_update_to_list(cleaned_update)

        with torch.no_grad():
            for parameter, delta in zip(target_model.parameters(), update_list):
                parameter.add_(delta.to(parameter.device, dtype=parameter.dtype))
        return target_model

    def run(
        self,
        model: Module,
        delta_t: Sequence[torch.Tensor],
        y_star: int,
        p_hat: float,
        x_syn: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Module]:
        """Convenience wrapper matching the requested DGC-Clean API."""

        self.model = model.to(self.device)
        result = self.clean_aggregated_update(
            aggregated_update=list(delta_t),
            metadata={
                "y_star": int(y_star),
                "p_hat": float(p_hat),
                "x_syn": x_syn,
            },
        )
        cleaned_update = result.cleaned_update
        if not isinstance(cleaned_update, list):
            cleaned_update = self._coerce_update_to_list(cleaned_update)
        return cleaned_update, deepcopy(self.ema_model)

    def reset(self) -> None:
        """Clear transient cleaning state."""

        self._last_subspace = None
        self.ema_model = deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize DGC-Clean state."""

        return {
            "config": dict(self.config),
            "ema_model": deepcopy(self.ema_model).cpu().state_dict(),
            "last_subspace_rank": 0 if self._last_subspace is None else self._last_subspace.get("rank", 0),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore DGC-Clean state."""

        self.config = self._build_config(state_dict.get("config", self.config))
        ema_state = state_dict.get("ema_model")
        if ema_state is not None:
            self.ema_model.load_state_dict(ema_state)
            self.ema_model.to(self.device)
            self.ema_model.eval()
            for parameter in self.ema_model.parameters():
                parameter.requires_grad_(False)

    def _build_config(self, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = {
            "num_gradient_samples": 8,
            "gradient_probe_batch_size": 8,
            "gradient_eps": 1e-12,
            "enable_projection": True,
            "enable_distillation": True,
            "batch_size": 32,
            "distill_epochs": 2,
            "distill_lr": 3e-4,
            "distill_lr_multiplier": 1.0,
            "distill_weight_decay": 0.0,
            "distill_max_steps": 3,
            "distill_max_delta_ratio": 0.10,
            "kd_temperature": 1.0,
            "ema_gamma": 0.999,
            "lambda_min": 0.08,
            "lambda_max": 0.60,
            "lambda_p0": 0.05,
            "lambda_p1": 0.20,
            "projection_scale_max_boost": 0.10,
            "projection_scope": "full",
            "subspace_method": "parameter_grad",
            "group_diff_rank": 10,
            "group_diff_min_anomalous": 2,
            "group_diff_min_energy_ratio": 0.3,
            "projection_classifier_keywords": ["fc", "classifier", "head", "linear"],
            "feature_subspace_rank": 10,
            "feature_subspace_energy_threshold": 0.0,
            "feature_centering": True,
            "feature_trigger_types": ["patch", "blend", "sig", "color"],
            "feature_space_disable_distillation": True,
            "feature_space_distill_classifier_only": True,
            "lambda_cap_when_unreliable": 0.30,
            "rho_reference": 0.20,
            "patch_size": 3,
            "blend_alpha": 0.2,
            "sig_delta": 0.2,
            "sig_frequency": 6.0,
            "sig_phase": 0.0,
            "color_offsets": [0.12, -0.08, 0.08],
            "normalization_mean": [0.4914, 0.4822, 0.4465],
            "normalization_std": [0.2470, 0.2430, 0.2610],
            "seed": 1234,
        }
        if config is not None:
            merged.update(dict(config))
        return merged

    def _resolve_device(self, device: Optional[torch.device], model: Module) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_param_slices(self) -> List[Dict[str, Any]]:
        slices: List[Dict[str, Any]] = []
        offset = 0
        for name, parameter in self.model.named_parameters():
            numel = int(parameter.numel())
            slices.append({
                "name": str(name),
                "start": int(offset),
                "end": int(offset + numel),
                "numel": int(numel),
            })
            offset += numel
        return slices

    def _resolve_projection_indices(self, scope: str) -> torch.Tensor:
        scope_text = str(scope).lower().strip()
        total_dim = int(sum(self._param_numels))
        if scope_text in {"", "full"}:
            return torch.arange(total_dim, device=self.device, dtype=torch.long)

        if scope_text != "classifier":
            raise ValueError(f"Unsupported projection_scope: {scope}")

        keywords = [str(token).lower() for token in self.config.get("projection_classifier_keywords", [])]
        selected: List[torch.Tensor] = []
        for item in self._param_slices:
            name = str(item["name"]).lower()
            if any(keyword in name for keyword in keywords):
                selected.append(torch.arange(int(item["start"]), int(item["end"]), device=self.device, dtype=torch.long))

        if len(selected) == 0:
            # Fallback: use the last parameter tensor(s), typically classifier weight/bias.
            if len(self._param_slices) >= 2:
                tail = self._param_slices[-2:]
            elif len(self._param_slices) == 1:
                tail = self._param_slices[-1:]
            else:
                tail = []
            for item in tail:
                selected.append(torch.arange(int(item["start"]), int(item["end"]), device=self.device, dtype=torch.long))

        if len(selected) == 0:
            return torch.arange(total_dim, device=self.device, dtype=torch.long)
        return torch.cat(selected, dim=0)

    def _build_projection_mask(self, scope: str, dim: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros(dim, device=self.device, dtype=dtype)
        indices = self._resolve_projection_indices(scope)
        if indices.numel() == 0:
            return mask
        indices = indices.clamp(min=0, max=max(dim - 1, 0))
        mask.index_fill_(0, indices, 1.0)
        return mask

    def _resolve_classifier_head(self, model: Module) -> Optional[Tuple[str, torch.nn.Linear]]:
        keywords = [str(token).lower() for token in self.config.get("projection_classifier_keywords", [])]
        linear_modules: List[Tuple[str, torch.nn.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules.append((str(name), module))
        if len(linear_modules) == 0:
            return None

        ranked: List[Tuple[int, int]] = []
        for idx, (name, _) in enumerate(linear_modules):
            lowered = name.lower()
            keyword_hit = 1 if any(keyword in lowered for keyword in keywords) else 0
            ranked.append((keyword_hit, idx))
        ranked.sort()
        _, best_idx = ranked[-1]
        return linear_modules[int(best_idx)]

    def _resolve_classifier_param_names(self, model: Module) -> List[str]:
        head = self._resolve_classifier_head(model)
        if head is None:
            return []
        module_name, module = head
        prefix = f"{module_name}." if module_name else ""
        names = [f"{prefix}weight"]
        if getattr(module, "bias", None) is not None:
            names.append(f"{prefix}bias")
        return names

    def _extract_penultimate_features(self, model: Module, normalized_inputs: torch.Tensor) -> torch.Tensor:
        head = self._resolve_classifier_head(model)
        if head is None:
            raise RuntimeError("Failed to resolve classifier head for feature-space projection.")
        _, classifier = head
        captured: Dict[str, torch.Tensor] = {}

        def _pre_hook(module: Module, args: Tuple[Any, ...]) -> None:
            if len(args) == 0 or (not isinstance(args[0], torch.Tensor)):
                raise RuntimeError("Classifier pre-hook did not receive tensor features.")
            captured["features"] = args[0].detach()

        handle = classifier.register_forward_pre_hook(_pre_hook)
        was_training = bool(model.training)
        model.eval()
        with torch.no_grad():
            _ = model(normalized_inputs)
        handle.remove()
        if was_training:
            model.train()

        features = captured.get("features")
        if features is None:
            raise RuntimeError("Failed to capture classifier input features.")
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        return features.detach().float()

    def _fit_activation_feature_subspace(self, model: Module, x_syn: torch.Tensor) -> Dict[str, Any]:
        num_batches = max(1, int(self.config.get("num_gradient_samples", 8)))
        batch_size = max(1, int(self.config.get("gradient_probe_batch_size", 8)))
        trigger_names_cfg = self.config.get("feature_trigger_types", ["patch", "blend", "sig", "color"])
        allowed_trigger_names = {str(name).strip().lower() for name in trigger_names_cfg if str(name).strip()}
        if len(allowed_trigger_names) == 0:
            allowed_trigger_names = {"patch", "blend", "sig", "color"}

        sampled_batches = self._sample_probe_batches(
            x_syn=x_syn,
            num_batches=num_batches,
            batch_size=batch_size,
        )
        diff_rows: List[torch.Tensor] = []
        used_trigger_names: Set[str] = set()
        for raw_batch in sampled_batches:
            raw_batch_01 = self._to_raw_inputs(raw_batch)
            clean_batch = self._normalize_inputs(raw_batch_01)
            clean_features = self._extract_penultimate_features(model, clean_batch)
            trigger_batches = self._build_trigger_batches(raw_batch_01)
            for trigger_name, trigger_batch in trigger_batches.items():
                name = str(trigger_name).lower()
                if name not in allowed_trigger_names:
                    continue
                trigger_features = self._extract_penultimate_features(model, trigger_batch)
                diff_rows.append((trigger_features - clean_features).detach())
                used_trigger_names.add(name)

        if len(diff_rows) == 0:
            return {
                "basis": torch.empty(0, 0, device=self.device, dtype=self._parameter_dtype()),
                "rank": 0,
                "num_vectors": 0,
                "sample_mode": "activation_feature_fc",
                "feature_dim": 0,
                "trigger_names": sorted(used_trigger_names),
            }

        diff_matrix = torch.cat(diff_rows, dim=0).to(self.device)  # [N, D]
        if bool(self.config.get("feature_centering", True)):
            diff_matrix = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)
        diff_matrix_t = diff_matrix.t().contiguous()  # [D, N]
        if diff_matrix_t.shape[0] == 0 or diff_matrix_t.shape[1] == 0:
            return {
                "basis": torch.empty(0, 0, device=self.device, dtype=self._parameter_dtype()),
                "rank": 0,
                "num_vectors": 0,
                "sample_mode": "activation_feature_fc",
                "feature_dim": int(diff_matrix_t.shape[0]),
                "trigger_names": sorted(used_trigger_names),
            }

        u_matrix, singular_values, _ = torch.linalg.svd(diff_matrix_t, full_matrices=False)
        max_rank = min(int(u_matrix.shape[1]), max(1, int(self.config.get("feature_subspace_rank", 10))))
        energy_threshold = float(self.config.get("feature_subspace_energy_threshold", 0.0))
        if energy_threshold > 0.0 and singular_values.numel() > 0:
            energy = singular_values.pow(2)
            cumulative = torch.cumsum(energy, dim=0) / max(float(energy.sum().item()), 1e-12)
            threshold_rank = int((cumulative >= min(max(energy_threshold, 0.0), 1.0)).nonzero(as_tuple=False)[0].item()) + 1
            max_rank = min(max_rank, max(1, threshold_rank))
        basis = u_matrix[:, :max_rank].contiguous()
        if basis.numel() > 0:
            keep_cols: List[int] = []
            for col in range(int(basis.shape[1])):
                if float(basis[:, col].norm(p=2).item()) > float(self.config["gradient_eps"]):
                    keep_cols.append(col)
            basis = basis[:, keep_cols] if len(keep_cols) > 0 else basis[:, :0]

        return {
            "basis": basis.detach(),
            "rank": int(basis.shape[1]),
            "num_vectors": int(diff_matrix.shape[0]),
            "sample_mode": "activation_feature_fc",
            "feature_dim": int(diff_matrix_t.shape[0]),
            "trigger_names": sorted(used_trigger_names),
        }

    def _apply_classifier_feature_projection(
        self,
        model: Module,
        basis: torch.Tensor,
        rho: float,
    ) -> Dict[str, Any]:
        head = self._resolve_classifier_head(model)
        if head is None:
            return {
                "applied": False,
                "reason": "classifier_not_found",
                "feature_dim": 0,
                "trainable_param_names": [],
            }
        module_name, classifier = head
        if not isinstance(classifier, torch.nn.Linear):
            return {
                "applied": False,
                "reason": "classifier_not_linear",
                "feature_dim": 0,
                "trainable_param_names": [],
            }

        weight = classifier.weight
        in_features = int(weight.shape[1])
        if int(basis.shape[0]) != in_features:
            return {
                "applied": False,
                "reason": "feature_dim_mismatch",
                "feature_dim": int(basis.shape[0]),
                "expected_dim": int(in_features),
                "trainable_param_names": self._resolve_classifier_param_names(model),
            }

        rho_clamped = float(max(0.0, min(1.0, float(rho))))
        if rho_clamped <= 0.0 or int(basis.shape[1]) == 0:
            return {
                "applied": False,
                "reason": "empty_basis_or_zero_rho",
                "feature_dim": int(basis.shape[0]),
                "trainable_param_names": self._resolve_classifier_param_names(model),
            }

        vv_t = basis.matmul(basis.t()).to(device=weight.device, dtype=weight.dtype)
        identity = torch.eye(vv_t.shape[0], device=weight.device, dtype=weight.dtype)
        projection_matrix = identity - rho_clamped * vv_t
        with torch.no_grad():
            weight_before = weight.detach().clone()
            weight_after = weight_before.matmul(projection_matrix)
            classifier.weight.copy_(weight_after)
            delta_weight = weight_after - weight_before

        raw_projection = weight_before.matmul(vv_t)
        return {
            "applied": True,
            "reason": "ok",
            "feature_dim": int(basis.shape[0]),
            "projection_norm_before": float(raw_projection.norm(p=2).item()),
            "projection_norm_after": float(delta_weight.norm(p=2).item()),
            "trainable_param_names": self._resolve_classifier_param_names(model),
            "classifier_module": str(module_name),
        }

    def _parameter_dtype(self) -> torch.dtype:
        for parameter in self.model.parameters():
            return parameter.dtype
        return torch.float32

    def _prepare_probe_tensor(self, x_syn: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x_syn, torch.Tensor):
            tensor = x_syn.detach().clone().float()
        else:
            tensor = torch.stack([item.detach().clone().float() for item in x_syn], dim=0)
        if tensor.dim() != 4:
            raise ValueError("X_syn must have shape [N, C, H, W].")
        return tensor.to(self.device)

    def _sample_probe_batches(self, x_syn: torch.Tensor, num_batches: int, batch_size: int) -> List[torch.Tensor]:
        num_samples = int(x_syn.shape[0])
        if num_samples == 0:
            raise ValueError("X_syn must contain at least one sample.")

        total_needed = max(1, num_batches) * max(1, batch_size)
        perm = torch.randperm(num_samples, generator=self._rng)
        if total_needed > num_samples:
            extra = total_needed - num_samples
            extra_perm = torch.randint(0, num_samples, size=(extra,), generator=self._rng)
            perm = torch.cat([perm, extra_perm], dim=0)
        else:
            perm = perm[:total_needed]

        batches = []
        for index in range(max(1, num_batches)):
            start = index * max(1, batch_size)
            end = start + max(1, batch_size)
            batch_indices = perm[start:end].tolist()
            batches.append(x_syn[batch_indices])
        return batches

    def _compute_bd_gradient(self, model: Module, x_batch: torch.Tensor, y_star: int) -> torch.Tensor:
        working_model = deepcopy(model).to(self.device)
        working_model.train()
        working_model.zero_grad(set_to_none=True)

        bd_loss = self._compute_bd_loss(working_model, x_batch, y_star)
        gradients = torch.autograd.grad(
            bd_loss,
            tuple(parameter for parameter in working_model.parameters() if parameter.requires_grad),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        return self._flatten_tensor_sequence([gradient.detach() for gradient in gradients])

    def _compute_bd_loss(self, model: Module, raw_inputs: torch.Tensor, y_star: int) -> torch.Tensor:
        trigger_batches = self._build_trigger_batches(raw_inputs)
        triggered_inputs = torch.cat(list(trigger_batches.values()), dim=0)
        logits = self._extract_logits(model(triggered_inputs))
        margins = self._margin_from_logits(logits, y_star)
        # softplus(m) = log(1 + exp(m)) exactly matches the requested loss.
        return F.softplus(margins).mean()

    def _run_online_distillation(
        self,
        student_model: Module,
        y_star: int,
        p_hat: float,
        x_syn: torch.Tensor,
        lambda_t: float,
        distill_steps_override: Optional[int] = None,
        trainable_parameter_names: Optional[Set[str]] = None,
        base_update_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Run short EMA-teacher distillation on the probe set.

        The student starts from the projected update ``w_t + delta_clean``. We
        then optimize the student for a few epochs with the sum of:
        - KD loss from the EMA teacher on clean probe images.
        - Adaptive backdoor-cleaning loss on triggered probe images.
        """

        distill_epochs = int(self.config["distill_epochs"])
        if distill_steps_override is not None:
            distill_epochs = max(0, int(distill_steps_override))
        distill_epochs = min(distill_epochs, max(0, int(self.config.get("distill_max_steps", 3))))
        if distill_epochs <= 0:
            self._update_ema_teacher(student_model)
            return {
                "avg_total_loss": 0.0,
                "avg_kd_loss": 0.0,
                "avg_bd_loss": 0.0,
                "teacher_student_weight_diff_initial": 0.0,
                "teacher_student_kl_initial": 0.0,
                "early_stop_triggered": False,
                "delta_norm_at_stop": 0.0,
                "distill_steps": 0,
            }

        # Teacher snapshot is created from current global model each audit cycle.
        teacher_model = deepcopy(self.model).to(self.device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        student_model.to(self.device)
        student_model.train()
        trainable_parameters: List[torch.Tensor] = []
        if trainable_parameter_names is None:
            trainable_parameters = [parameter for parameter in student_model.parameters() if parameter.requires_grad]
        else:
            allowed = {str(name) for name in trainable_parameter_names}
            trainable_parameters = [
                parameter
                for name, parameter in student_model.named_parameters()
                if parameter.requires_grad and str(name) in allowed
            ]
        if len(trainable_parameters) == 0:
            self._update_ema_teacher(student_model)
            return {
                "avg_total_loss": 0.0,
                "avg_kd_loss": 0.0,
                "avg_bd_loss": 0.0,
                "lambda_t": float(lambda_t),
                "p_hat": float(p_hat),
                "teacher_student_weight_diff_initial": 0.0,
                "teacher_student_kl_initial": 0.0,
                "early_stop_triggered": False,
                "delta_norm_at_stop": 0.0,
                "distill_steps": 0,
            }

        distill_lr = float(self.config["distill_lr"]) * float(self.config.get("distill_lr_multiplier", 1.0))
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=float(distill_lr),
            weight_decay=float(self.config["distill_weight_decay"]),
        )

        batch_size = max(1, int(self.config["batch_size"]))
        temperature = float(self.config["kd_temperature"])
        total_loss_sum = 0.0
        kd_loss_sum = 0.0
        bd_loss_sum = 0.0
        total_steps = 0
        early_stop_triggered = False
        delta_norm_at_stop = 0.0
        teacher_student_weight_diff_initial = 0.0
        teacher_student_kl_initial = 0.0
        student_before_flat = self._flatten_model_parameters(student_model).detach()

        with torch.no_grad():
            probe_preview = self._normalize_inputs(self._to_raw_inputs(x_syn[: min(int(x_syn.shape[0]), 32)]))
            teacher_logits_preview = self._extract_logits(teacher_model(probe_preview))
            student_logits_preview = self._extract_logits(student_model(probe_preview))
            teacher_student_kl_initial = float(
                F.kl_div(
                    F.log_softmax(student_logits_preview / temperature, dim=-1),
                    F.softmax(teacher_logits_preview / temperature, dim=-1),
                    reduction="batchmean",
                ).item()
            )
            teacher_student_weight_diff_initial = float(
                (
                    self._flatten_model_parameters(teacher_model)
                    - self._flatten_model_parameters(student_model)
                ).norm(p=2).item()
            )

        reference_update_norm = float(base_update_norm) if base_update_norm is not None else float(
            student_before_flat.norm(p=2).item()
        )
        max_delta_ratio = float(max(self.config.get("distill_max_delta_ratio", 0.10), 0.0))
        max_allowed_delta_norm = max(1e-8, max_delta_ratio * max(reference_update_norm, 1e-8))

        for _ in range(distill_epochs):
            perm = torch.randperm(x_syn.shape[0], generator=self._rng)
            for start in range(0, x_syn.shape[0], batch_size):
                batch_indices = perm[start:start + batch_size].tolist()
                raw_batch = self._to_raw_inputs(x_syn[batch_indices])
                clean_batch = self._normalize_inputs(raw_batch)

                with torch.no_grad():
                    teacher_logits = self._extract_logits(teacher_model(clean_batch))

                student_logits = self._extract_logits(student_model(clean_batch))
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                bd_loss = self._compute_bd_loss(student_model, raw_batch, y_star)
                total_loss = kd_loss + lambda_t * bd_loss

                student_model.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                total_steps += 1
                total_loss_sum += float(total_loss.item())
                kd_loss_sum += float(kd_loss.item())
                bd_loss_sum += float(bd_loss.item())

                with torch.no_grad():
                    delta_norm = float(
                        (
                            self._flatten_model_parameters(student_model) - student_before_flat
                        ).norm(p=2).item()
                    )
                if delta_norm > max_allowed_delta_norm:
                    early_stop_triggered = True
                    delta_norm_at_stop = float(delta_norm)
                    break
            if early_stop_triggered:
                break

        student_model.eval()
        self._update_ema_teacher(student_model)

        normalizer = max(total_steps, 1)
        return {
            "avg_total_loss": total_loss_sum / normalizer,
            "avg_kd_loss": kd_loss_sum / normalizer,
            "avg_bd_loss": bd_loss_sum / normalizer,
            "lambda_t": float(lambda_t),
            "p_hat": float(p_hat),
            "teacher_student_weight_diff_initial": float(teacher_student_weight_diff_initial),
            "teacher_student_kl_initial": float(teacher_student_kl_initial),
            "early_stop_triggered": bool(early_stop_triggered),
            "delta_norm_at_stop": float(delta_norm_at_stop),
            "distill_lr": float(distill_lr),
            "max_allowed_delta_norm": float(max_allowed_delta_norm),
            "distill_steps": int(total_steps),
        }

    def _update_ema_teacher(self, student_model: Module) -> None:
        gamma = float(self.config["ema_gamma"])
        with torch.no_grad():
            for ema_parameter, student_parameter in zip(self.ema_model.parameters(), student_model.parameters()):
                ema_parameter.mul_(gamma).add_(student_parameter.detach(), alpha=1.0 - gamma)
            for ema_buffer, student_buffer in zip(self.ema_model.buffers(), student_model.buffers()):
                ema_buffer.copy_(student_buffer.detach())
        self.ema_model.eval()

    def _compute_lambda_t(
        self,
        p_hat: float,
        cleaning_confidence: float = 1.0,
        localization_reliable: bool = True,
        effective_rho: float = 0.0,
    ) -> float:
        lambda_min = float(self.config["lambda_min"])
        lambda_max = float(self.config["lambda_max"])
        p0 = float(self.config["lambda_p0"])
        p1 = float(self.config["lambda_p1"])
        if p1 <= p0:
            lambda_t = lambda_max
        else:
            ratio = (float(p_hat) - p0) / (p1 - p0)
            ratio = max(0.0, min(1.0, ratio))
            lambda_t = lambda_min + (lambda_max - lambda_min) * ratio
        if not localization_reliable:
            confidence_scale = 0.5 + 0.5 * max(0.0, min(1.0, float(cleaning_confidence)))
            lambda_t = min(lambda_t * confidence_scale, float(self.config.get("lambda_cap_when_unreliable", 0.30)))
        rho_reference = max(float(self.config.get("rho_reference", 0.20)), 1e-6)
        rho_scale = max(0.0, min(2.0, float(effective_rho) / rho_reference))
        lambda_t = lambda_t * rho_scale
        lambda_t = max(lambda_t, 0.0)
        return lambda_t

    def _compute_projection_scale(self, lambda_t: float) -> float:
        lambda_min = float(self.config["lambda_min"])
        lambda_max = float(self.config["lambda_max"])
        if lambda_max <= lambda_min:
            return 1.0
        ratio = (float(lambda_t) - lambda_min) / (lambda_max - lambda_min)
        ratio = max(0.0, min(1.0, ratio))
        return 1.0 + float(self.config.get("projection_scale_max_boost", 0.25)) * ratio

    def _flatten_model_parameters(self, model: Module) -> torch.Tensor:
        return self._flatten_tensor_sequence([parameter.detach() for parameter in model.parameters()])

    def _flatten_update(self, update: ModelUpdate) -> torch.Tensor:
        if isinstance(update, Mapping):
            flat_parts = []
            named_parameters = OrderedDict(self.model.named_parameters())
            for name in self._param_names:
                tensor = update.get(name)
                if tensor is None:
                    tensor = torch.zeros_like(named_parameters[name])
                flat_parts.append(tensor.detach().to(self.device).reshape(-1))
            return torch.cat(flat_parts, dim=0)
        if isinstance(update, torch.Tensor):
            return update.detach().to(self.device).reshape(-1)
        if isinstance(update, (list, tuple)):
            return self._flatten_tensor_sequence(update)
        raise TypeError("Unsupported update type for flattening.")

    def _flatten_tensor_sequence(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        flat_parts = [tensor.detach().to(self.device).reshape(-1) for tensor in tensors]
        if len(flat_parts) == 0:
            return torch.empty(0, device=self.device, dtype=self._parameter_dtype())
        return torch.cat(flat_parts, dim=0)

    def _unflatten_like_model(self, flat_vector: torch.Tensor) -> List[torch.Tensor]:
        flat_vector = flat_vector.to(self.device)
        expected_numel = int(sum(self._param_numels))
        if flat_vector.numel() != expected_numel:
            raise ValueError(
                f"Flat vector has {flat_vector.numel()} elements, expected {expected_numel}."
            )

        outputs: List[torch.Tensor] = []
        offset = 0
        for shape, numel, parameter in zip(self._param_shapes, self._param_numels, self.model.parameters()):
            piece = flat_vector[offset:offset + numel].view(shape).to(dtype=parameter.dtype)
            outputs.append(piece.clone())
            offset += numel
        return outputs

    def _coerce_update_to_list(self, update: ModelUpdate) -> List[torch.Tensor]:
        if isinstance(update, list):
            return [tensor.detach().clone().to(self.device) for tensor in update]
        if isinstance(update, tuple):
            return [tensor.detach().clone().to(self.device) for tensor in update]
        if isinstance(update, Mapping):
            named_parameters = OrderedDict(self.model.named_parameters())
            return [
                update.get(name, torch.zeros_like(named_parameters[name])).detach().clone().to(self.device)
                for name in self._param_names
            ]
        if isinstance(update, torch.Tensor):
            return self._unflatten_like_model(update)
        raise TypeError("Unsupported update type.")

    def _normalize_inputs(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config["normalization_mean"], dtype=raw_inputs.dtype, device=raw_inputs.device)
        std = torch.tensor(self.config["normalization_std"], dtype=raw_inputs.dtype, device=raw_inputs.device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return (raw_inputs - mean) / std

    def _to_raw_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        tensor = inputs.detach().clone().float().to(self.device)
        if tensor.numel() == 0:
            return tensor
        min_value = float(tensor.min().item())
        max_value = float(tensor.max().item())
        looks_normalized = (min_value < -0.2) or (max_value > 1.2)
        if looks_normalized:
            mean = torch.tensor(self.config["normalization_mean"], dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
            std = torch.tensor(self.config["normalization_std"], dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
            tensor = tensor * std + mean
        return torch.clamp(tensor, 0.0, 1.0)

    def _build_trigger_batches(self, raw_inputs: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        raw = self._to_raw_inputs(raw_inputs)
        patch_size = min(int(self.config["patch_size"]), raw.shape[-2], raw.shape[-1])

        patch = raw.clone()
        patch[:, :, -patch_size:, -patch_size:] = 1.0

        blend_noise = torch.rand(raw.shape, generator=self._rng, device="cpu", dtype=raw.dtype).to(raw.device)
        blend_alpha = float(self.config["blend_alpha"])
        blend = torch.clamp((1.0 - blend_alpha) * raw + blend_alpha * blend_noise, 0.0, 1.0)

        sig = torch.clamp(raw + self._build_sig_pattern(raw), 0.0, 1.0)

        color_offsets = torch.tensor(self.config["color_offsets"], device=raw.device, dtype=raw.dtype)
        if color_offsets.numel() == 1:
            color_offsets = color_offsets.repeat(raw.shape[1])
        color_offsets = color_offsets.view(1, raw.shape[1], 1, 1)
        color = torch.clamp(raw + color_offsets, 0.0, 1.0)

        return OrderedDict(
            patch=self._normalize_inputs(patch),
            blend=self._normalize_inputs(blend),
            sig=self._normalize_inputs(sig),
            color=self._normalize_inputs(color),
        )

    def _build_sig_pattern(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = raw_inputs.shape
        xs = torch.linspace(0.0, 1.0, steps=width, device=raw_inputs.device, dtype=raw_inputs.dtype)
        sine = torch.sin(2.0 * torch.pi * float(self.config["sig_frequency"]) * xs + float(self.config["sig_phase"]))
        pattern = sine.view(1, 1, 1, width).repeat(1, channels, height, 1)
        return float(self.config["sig_delta"]) * pattern

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError("Model forward must return logits as a tensor or tuple/list containing one tensor.")

    def _margin_from_logits(self, logits: torch.Tensor, y_star: int) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError("Expected logits with shape [B, C].")
        target_logits = logits[:, int(y_star)]
        sum_logits = logits.sum(dim=-1)
        other_mean = (sum_logits - target_logits) / max(logits.shape[-1] - 1, 1)
        return target_logits - other_mean


def run_dgc_clean(
    model: Module,
    delta_t: Sequence[torch.Tensor],
    y_star: int,
    p_hat: float,
    X_syn: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[torch.Tensor], Module]:
    """Run DGC-Clean with the requested raw interface.

    Args:
        model: Current global model.
        delta_t: Aggregated update represented as a list of parameter-shaped
            tensors matching ``model.parameters()``.
        y_star: Estimated target class from CTS-Intent.
        p_hat: Pollution-strength estimate from S3-Loc.
        X_syn: Probe set shared with CTS-Intent, shape ``[N_syn, C, H, W]``.
        config: Hyper-parameter dictionary.

    Returns:
        ``(delta_clean, ema_model)`` where ``delta_clean`` is the final cleaned
        update after projection and online distillation.
    """

    cleaner = DGCClean(config=config or {}, model=model)
    return cleaner.run(model=model, delta_t=delta_t, y_star=y_star, p_hat=p_hat, x_syn=X_syn)
