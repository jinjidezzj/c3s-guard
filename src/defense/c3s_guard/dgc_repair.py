"""DGC-Repair: Gate -> Risk-aware sampling -> Min-Max style server repair."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from defense.c3s_guard.utils import ModelUpdate


class DGCRepair:
    """Server-side behavior repair module for SA-constrained FL."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]],
        model: Module,
        cts_intent: Any,
        device: Union[str, torch.device],
    ) -> None:
        merged = self.default_config()
        merged.update(dict(config or {}))
        self.config = merged
        self.model = model
        self.cts_intent = cts_intent
        self.device = torch.device(device)
        self.num_clients = int(self.config.get("num_clients", 100))
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.config.get("seed", 1234)))
        self._sample_rng = np.random.default_rng(int(self.config.get("seed", 1234)) + 17)
        self.reset()

    @staticmethod
    def default_config() -> Dict[str, Any]:
        return {
            "seed": 1234,
            "num_clients": 100,
            "enabled": True,
            "ablation": "full",
            "tau0": 0.5,
            "tau1": 2.0,
            "alpha_min": 0.2,
            "rho": 0.9,
            "gate_floor_ratio": 0.3,
            "check_interval": 5,
            "alert_horizon": 3,
            "p_hat_alert": 0.15,
            "tau_alert": 0.6,
            "tau_reject": 1.5,
            "tau_alert_z": 1.5,
            "tau_reject_z": 3.0,
            "cts_light_ema_beta": 0.9,
            "cts_light_ema_std_eps": 1e-3,
            "reject_confirm_rounds": 2,
            "reject_requires_safe": True,
            "max_consecutive_reject_without_safe": 1,
            "reject_fallback_scale_without_safe": 0.25,
            "enable_no_safe_minimal_fallback": True,
            "no_safe_minimal_repair_steps": 1,
            "no_safe_minimal_lambda_scale": 0.50,
            "no_safe_high_risk_branch": "gated",  # noop | gated
            "no_safe_alert_scale": 0.70,
            "no_safe_noop_max_rounds": 2,
            "tau_safe": 1.5,
            "lambda_min": 0.05,
            "lambda_medium": 0.5,
            "lambda_max": 2.0,
            "p_max": 0.3,
            "mu_reg": 1e-4,
            "steps_normal": 1,
            "steps_alert": 3,
            "steps_abnormal": 5,
            "repair_lr_ratio_normal": 0.01,
            "repair_lr_ratio_alert": 0.03,
            "repair_lr_ratio_abnormal": 0.05,
            "repair_lr_min": 5e-5,
            "repair_lr_max": 2e-3,
            "grad_clip": 1.0,
            "acc_init_min": 0.40,
            "acc_init_fallback_min": 0.30,
            "weak_safe_init_probe_min": 0.10,
            "full_safe_init_probe_min": 0.40,
            "safe_init_max_wait_round": 15,
            "allow_safe_init_after_wait": True,
            "safe_init_fallback_ratio": 0.50,
            "allow_safe_init_without_probe": True,
            "safe_init_abnormal_ratio_max": 0.10,
            "safe_init_cts_max": 1.5,
            "safe_init_p_hat_max": 0.12,
            "full_safe_upgrade_max_reject_count": 1,
            "full_safe_upgrade_probe_ratio": 0.80,
            "full_safe_upgrade_fallback_probe_min": 0.15,
            "full_safe_upgrade_min_wait_round": 15,
            "full_safe_upgrade_relax_start_round": 60,
            "full_safe_upgrade_relax_probe_min": 0.12,
            "full_safe_upgrade_use_relaxed_context": True,
            "acc_full_repair": 0.50,
            "acc_full_repair_floor": 0.08,
            "lambda_scale_min_when_safe_low": 0.15,
            "lambda_scale_min_when_safe_low_abnormal": 0.35,
            "steps_cap_when_safe_low": 3,
            "steps_cap_when_safe_low_abnormal": 5,
            "repair_lr_scale_when_safe_low": 0.60,
            "repair_lr_scale_when_safe_low_abnormal": 0.80,
            "steps_cap_when_lambda_zero": 1,
            "repair_lr_scale_when_lambda_zero": 0.35,
            "safe_update_threshold": 0.10,
            "safe_update_p_hat_max": 0.08,
            "block_safe_update_in_alert_mode": True,
            "safe_update_abnormal_ratio_max": 0.10,
            "safe_update_cts_max": 1.5,
            "safe_update_max_probe_drop": 0.03,
            "safe_update_abnormal_ratio_max_weak": 0.25,
            "safe_update_cts_max_weak": 2.5,
            "safe_update_p_hat_max_weak": 0.25,
            "safe_signal_source": "dcbd",  # raw | dcbd | fct
            "safe_update_ratio_source": "confirmed",  # raw | confirmed | auto
            "safe_ema_alpha": 0.99,
            "safe_refresh_enable": False,
            "safe_refresh_stale_rounds": 20,
            "safe_refresh_margin": 0.10,
            "safe_refresh_ema": 0.5,
            "safe_refresh_confirmed_abnormal_ratio_max": 0.20,
            "safe_refresh_cts_max": 2.0,
            "safe_refresh_max_confirmed": 3,
            "safe_init_fallback_allow_relaxed_context": True,
            "safe_init_fallback_relaxed_abnormal_ratio_max": 0.20,
            "safe_init_fallback_relaxed_cts_max": 3.0,
            "safe_init_fallback_relaxed_p_hat_max": 0.20,
            "risk_aware_sampling": False,
            "risk_aware_source": "confirmed",  # raw | confirmed
            "risk_allow_negative_signal": False,
            "risk_confirmed_min_count": 2,
            "risk_confirmed_min_suspicious_score": 0.05,
            "risk_confirmed_fallback_source": "none",  # none | raw
            "confirmed_gate_enable": False,
            "confirmed_gate_beta": 8.0,
            "confirmed_gate_threshold": 0.20,
            "risk_beta": 0.9,
            "risk_gamma": 1.0,
            "risk_xi": 0.5,
            "risk_decay": 0.98,
            "risk_p_min": 0.5,
            "repair_num_triggers": 4,
            "probe_size_light": 128,
            "num_triggers_light": 2,
            "trigger_types": ["patch", "blend", "color", "occlusion", "sig"],
            "patch_size": 3,
            "blend_alpha_min": 0.10,
            "blend_alpha_max": 0.25,
            "color_shift": 16.0 / 255.0,
            "occlusion_ratio": 0.22,
            "sig_amp_min": 0.03,
            "sig_amp_max": 0.05,
            "normalization_mean": [0.4914, 0.4822, 0.4465],
            "normalization_std": [0.2470, 0.2435, 0.2616],
            "max_probe_eval_batches": 4,
            "safe_probe_eval_random_offset": True,
            "safe_probe_eval_use_bn_calibration": True,
            "safe_probe_bn_calibration_batches": 2,
            "temperature": 1.0,
            "bootstrap_steps_cap": 2,
            "bootstrap_steps_cap_abnormal": 3,
            "bootstrap_repair_lr_scale": 0.7,
            "bootstrap_allow_anti_bd": True,
            "bootstrap_lambda_ratio": 0.30,
            "bootstrap_lambda_max": 0.40,
            "bootstrap_min_probe_for_anti_bd": 0.10,
        }

    def reset(self) -> None:
        self.risk_scores = torch.zeros(self.num_clients, dtype=torch.float32)
        self.current_round = -1
        self.last_cts_global_light = 0.0
        self.last_cts_global_light_centered = 0.0
        self.last_cts_global_light_z = 0.0
        self.cts_light_ema_mean = 0.0
        self.cts_light_ema_var = 1.0
        self.cts_light_ema_initialized = False
        self.last_cts_checked_round = -1
        self.alert_until_round = -1
        self.recent_alert_mode = False
        self.consecutive_reject_count = 0
        self.reject_counter = 0
        self.last_gate_update: Optional[ModelUpdate] = None
        self.last_gate_stats: Dict[str, Any] = {}
        self.w_safe: Optional[Module] = None
        self.safe_model_stage = "none"  # none | weak | full
        self.safe_initialized = False
        self.safe_updated = False
        self.last_safe_probe_acc = 0.0
        self.last_safe_update_round = -1
        self.last_repair_metrics: Dict[str, Any] = {
            "cts_global_light": 0.0,
            "cts_global_light_centered": 0.0,
            "cts_global_light_z": 0.0,
            "cts_checked_this_round": False,
            "recent_alert_mode": False,
            "gate_mean": 1.0,
            "gate_min": 1.0,
            "gate_reject_count": 0,
            "gate_floor_activated": False,
            "effective_ratio": 1.0,
            "risk_score_mean": 0.0,
            "risk_score_max": 0.0,
            "risk_aware_source": str(self.config.get("risk_aware_source", "raw")),
            "risk_confirmed_count": 0,
            "risk_signal_mean": 0.0,
            "risk_signal_min": 0.0,
            "risk_signal_max": 0.0,
            "risk_update_applied": False,
            "risk_update_skip_reason": "none",
            "repair_steps": 0,
            "lambda_bd": 0.0,
            "repair_lr": 0.0,
            "l_kd": 0.0,
            "l_anti_bd": 0.0,
            "l_reg": 0.0,
            "safe_updated": False,
            "safe_initialized": False,
            "safe_initialized_this_round": False,
            "safe_model_exists": False,
            "safe_model_ready": False,
            "safe_model_stage": "none",
            "safe_bootstrap_mode": False,
            "bootstrap_repair_enabled": False,
            "safe_probe_acc": 0.0,
            "safe_init_allowed": False,
            "safe_init_reason": "none",
            "safe_init_used_fallback": False,
            "safe_init_source_model": "none",
            "safe_init_threshold_used": "none",
            "safe_init_blocking_condition": "none",
            "safe_init_context_ok": False,
            "safe_upgrade_allowed": False,
            "safe_upgrade_reason": "none",
            "safe_update_allowed": False,
            "safe_update_reason": "none",
            "safe_model_update_reason": "none",
            "safe_update_context_ok": False,
            "safe_signal_source": str(self.config.get("safe_signal_source", "dcbd")),
            "safe_signal_source_effective": "dcbd",
            "safe_update_ratio_source": str(self.config.get("safe_update_ratio_source", "confirmed")),
            "safe_update_ratio_source_effective": "raw",
            "safe_effective_abnormal_ratio": 0.0,
            "safe_refresh_applied": False,
            "safe_refresh_reason": "disabled",
            "safe_refresh_stale_rounds": int(self.config.get("safe_refresh_stale_rounds", 20)),
            "safe_refresh_margin": float(self.config.get("safe_refresh_margin", 0.10)),
            "safe_refresh_ema": float(self.config.get("safe_refresh_ema", 0.5)),
            "safe_refresh_last_update_round": int(self.last_safe_update_round),
            "alert_mode": False,
            "alert_mode_reason": "none",
            "reject_reason": "none",
            "reject_allowed": True,
            "reject_blocked_by_no_safe_model": False,
            "consecutive_reject_count": 0,
            "reject_counter": 0,
            "reject_fallback_branch_used": False,
            "reject_block_reason": "none",
            "final_update_branch_hint": "raw",
            "lambda_branch": "normal",
            "lambda_base_before_safe_scale": 0.0,
            "lambda_safe_scale_ratio": 1.0,
            "lambda_safe_scale_floor": 1.0,
            "lambda_cap_reason": "none",
            "repair_steps_planned": 0,
            "repair_steps_after_safe_cap": 0,
            "repair_steps_after_lambda_cap": 0,
            "repair_steps_final": 0,
            "repair_state_before_safe_check": "off",
            "repair_state_after_safe_check": "off",
            "repair_disabled_reason": "none",
            "lambda_disabled_reason": "none",
            "assert_abnormal_steps_planned": True,
            "assert_lambda_cap_reason_present": True,
            "assert_safe_init_reason_present": True,
            "assert_safe_update_reason_present": True,
            "global_gate_rejected": False,
            "num_abnormal_groups": 0,
            "abnormal_group_ratio": 0.0,
            "confirmed_abnormal_ratio": 0.0,
            "confirmed_gate_enabled": bool(self.config.get("confirmed_gate_enable", False)),
            "confirmed_gate_mean": 1.0,
            "confirmed_gate_min": 1.0,
            "confirmed_gate_threshold": float(self.config.get("confirmed_gate_threshold", 0.20)),
            "confirmed_gate_beta": float(self.config.get("confirmed_gate_beta", 8.0)),
            "confirmed_gate_reason": "disabled",
            "p_hat": 0.0,
            "suspicious_clients": [],
            "normal_clients": [],
        }

    def on_round_start(self, round_idx: int) -> None:
        self.current_round = int(round_idx)
        self.safe_initialized = False
        self.safe_updated = False
        self.last_gate_update = None
        self.last_gate_stats = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    @property
    def ablation(self) -> str:
        return str(self.config.get("ablation", "full")).strip().lower()

    def gate_enabled(self) -> bool:
        return self.ablation != "wo_gate"

    def risk_enabled(self) -> bool:
        return self.ablation != "wo_risk"

    def repair_enabled(self) -> bool:
        return self.ablation != "wo_repair"

    def augmented_triggers_enabled(self) -> bool:
        if self.ablation == "repair_fixed_triggers":
            return False
        if self.ablation == "repair_augmented_triggers":
            return True
        return True

    def _clone_update(self, update: ModelUpdate) -> ModelUpdate:
        if isinstance(update, Mapping):
            return OrderedDict((str(k), v.detach().clone()) for k, v in update.items())
        if isinstance(update, torch.Tensor):
            return update.detach().clone()
        raise TypeError("Unsupported update type for DGC-Repair.")

    def _zero_update_like(self, update: ModelUpdate) -> ModelUpdate:
        if isinstance(update, Mapping):
            return OrderedDict((str(k), torch.zeros_like(v)) for k, v in update.items())
        if isinstance(update, torch.Tensor):
            return torch.zeros_like(update)
        raise TypeError("Unsupported update type for DGC-Repair.")

    def _scale_update(self, update: ModelUpdate, scale: float) -> ModelUpdate:
        s = float(scale)
        if isinstance(update, Mapping):
            return OrderedDict((str(k), v.detach().clone() * s) for k, v in update.items())
        if isinstance(update, torch.Tensor):
            return update.detach().clone() * s
        raise TypeError("Unsupported update type for DGC-Repair.")

    def _add_scaled_inplace(self, target: ModelUpdate, source: ModelUpdate, alpha: float) -> None:
        a = float(alpha)
        if isinstance(target, Mapping) and isinstance(source, Mapping):
            for key in target.keys():
                target[key].add_(source[key].to(dtype=target[key].dtype, device=target[key].device), alpha=a)
            return
        if isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor):
            target.add_(source.to(dtype=target.dtype, device=target.device), alpha=a)
            return
        raise TypeError("Mismatched update types in _add_scaled_inplace.")

    def compute_audit_gate(
        self,
        *,
        group_ids: Sequence[int],
        group_updates: Mapping[int, ModelUpdate],
        group_members: Mapping[int, Sequence[int]],
        cts_diff_map: Mapping[int, float],
        b_g_map: Mapping[int, bool],
        confirmed_suspects: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Compute group-wise soft gate and gated aggregate for audit rounds."""

        if len(group_ids) == 0:
            self.last_gate_update = None
            return {
                "gate_values": {},
                "gate_mean": 1.0,
                "gate_min": 1.0,
                "effective_ratio": 1.0,
                "gate_floor_activated": False,
                "num_abnormal_groups": 0,
                "abnormal_group_ratio": 0.0,
                "confirmed_abnormal_ratio": 0.0,
                "confirmed_gate_enabled": bool(self.config.get("confirmed_gate_enable", False)),
                "confirmed_gate_mean": 1.0,
                "confirmed_gate_min": 1.0,
                "confirmed_gate_reject_count": 0,
                "confirmed_gate_threshold": float(self.config.get("confirmed_gate_threshold", 0.20)),
                "confirmed_gate_beta": float(self.config.get("confirmed_gate_beta", 8.0)),
                "confirmed_gate_reason": "no_groups",
            }

        cts_values = np.asarray([float(cts_diff_map.get(int(gid), 0.0)) for gid in group_ids], dtype=np.float32)
        mu = float(np.median(cts_values))
        sigma = float(1.4826 * np.median(np.abs(cts_values - mu)) + 1e-8)
        tau0 = float(self.config.get("tau0", 0.5))
        tau1 = float(max(self.config.get("tau1", 2.0), tau0 + 1e-6))
        rho = float(max(self.config.get("rho", 0.9), 0.0))
        alpha_min = float(np.clip(self.config.get("alpha_min", 0.2), 0.0, 1.0))

        gate_values: Dict[int, float] = {}
        group_sizes: Dict[int, int] = {}
        for gid in group_ids:
            group_id = int(gid)
            group_sizes[group_id] = max(int(len(group_members.get(group_id, []))), 1)
            z = (float(cts_diff_map.get(group_id, 0.0)) - mu) / sigma
            if (not self.gate_enabled()) or z <= tau0:
                gate = 1.0
            elif z >= tau1:
                gate = 0.0
            else:
                gate = max(1.0 - rho * (z - tau0) / max(tau1 - tau0, 1e-8), alpha_min)
            gate_values[group_id] = float(np.clip(gate, 0.0, 1.0))

        confirmed_set = {
            int(client_id)
            for client_id in (confirmed_suspects or [])
            if 0 <= int(client_id) < int(self.num_clients)
        }
        confirmed_gate_enabled = bool(self.config.get("confirmed_gate_enable", False))
        confirmed_gate_beta = float(max(self.config.get("confirmed_gate_beta", 8.0), 0.0))
        confirmed_gate_threshold = float(np.clip(self.config.get("confirmed_gate_threshold", 0.20), 0.0, 1.0))
        confirmed_gate_values: Dict[int, float] = {}
        confirmed_gate_reason = "disabled"
        if confirmed_gate_enabled:
            if len(confirmed_set) <= 0:
                confirmed_gate_reason = "no_confirmed_suspects"
            else:
                confirmed_gate_reason = "applied"
            for gid in group_ids:
                group_id = int(gid)
                members = [int(cid) for cid in group_members.get(group_id, [])]
                group_size = max(len(members), 1)
                confirmed_count = int(sum(1 for cid in members if cid in confirmed_set))
                confirmed_ratio = float(confirmed_count / float(group_size))
                logit = float(np.clip(confirmed_gate_beta * (confirmed_ratio - confirmed_gate_threshold), -40.0, 40.0))
                suppress = float(1.0 / (1.0 + np.exp(-logit)))
                confirmed_gate = float(np.clip(1.0 - suppress, 0.0, 1.0))
                if len(confirmed_set) <= 0:
                    confirmed_gate = 1.0
                confirmed_gate_values[group_id] = float(confirmed_gate)
                gate_values[group_id] = float(np.clip(gate_values[group_id] * confirmed_gate, 0.0, 1.0))
        else:
            for gid in group_ids:
                confirmed_gate_values[int(gid)] = 1.0

        total_clients = float(sum(group_sizes.values()))
        effective_clients = float(sum(gate_values[gid] * group_sizes[gid] for gid in group_ids))
        effective_ratio = float(effective_clients / max(total_clients, 1e-8))
        floor_activated = False
        floor_ratio = float(np.clip(self.config.get("gate_floor_ratio", 0.3), 0.0, 1.0))
        if effective_ratio < floor_ratio:
            floor_activated = True
            for gid in group_ids:
                gate_values[int(gid)] = max(float(gate_values[int(gid)]), floor_ratio)
            effective_clients = float(sum(gate_values[int(gid)] * group_sizes[int(gid)] for gid in group_ids))
            effective_ratio = float(effective_clients / max(total_clients, 1e-8))

        ref_gid = int(group_ids[0])
        ref_update = group_updates[ref_gid]
        gated_sum = self._zero_update_like(ref_update)
        denom = 0.0
        for gid in group_ids:
            group_id = int(gid)
            if group_id not in group_updates:
                continue
            gate = float(gate_values[group_id])
            n_r = float(group_sizes[group_id])
            self._add_scaled_inplace(gated_sum, group_updates[group_id], gate)
            denom += gate * n_r
        if denom <= 1e-8:
            gated_update = self._clone_update(ref_update)
        else:
            gated_update = self._scale_update(gated_sum, 1.0 / denom)
        self.last_gate_update = gated_update
        self.last_gate_stats = {
            "gate_values": {int(k): float(v) for k, v in gate_values.items()},
            "gate_mean": float(np.mean(list(gate_values.values()))) if gate_values else 1.0,
            "gate_min": float(np.min(list(gate_values.values()))) if gate_values else 1.0,
            "gate_reject_count": int(sum(1 for v in gate_values.values() if float(v) <= 1e-6)),
            "gate_floor_activated": bool(floor_activated),
            "effective_ratio": float(effective_ratio),
            "num_abnormal_groups": int(sum(1 for gid in group_ids if bool(b_g_map.get(int(gid), False)))),
            "abnormal_group_ratio": float(
                sum(1 for gid in group_ids if bool(b_g_map.get(int(gid), False))) / max(len(group_ids), 1)
            ),
            "confirmed_abnormal_ratio": float(
                sum(
                    1
                    for gid in group_ids
                    if bool(b_g_map.get(int(gid), False))
                    and len(
                        set(int(cid) for cid in group_members.get(int(gid), []))
                        .intersection(confirmed_set)
                    )
                    > 0
                )
                / max(len(group_ids), 1)
            ),
            "confirmed_gate_enabled": bool(confirmed_gate_enabled),
            "confirmed_gate_mean": float(np.mean(list(confirmed_gate_values.values()))) if confirmed_gate_values else 1.0,
            "confirmed_gate_min": float(np.min(list(confirmed_gate_values.values()))) if confirmed_gate_values else 1.0,
            "confirmed_gate_values": {int(k): float(v) for k, v in confirmed_gate_values.items()},
            "confirmed_gate_reject_count": int(
                sum(1 for _, value in confirmed_gate_values.items() if float(value) <= 0.5)
            ),
            "confirmed_gate_threshold": float(confirmed_gate_threshold),
            "confirmed_gate_beta": float(confirmed_gate_beta),
            "confirmed_gate_reason": str(confirmed_gate_reason),
            "median_cts_diff": float(mu),
            "sigma_cts_diff": float(sigma),
        }
        return dict(self.last_gate_stats)

    def update_risk_scores(
        self,
        *,
        group_ids: Sequence[int],
        group_members: Mapping[int, Sequence[int]],
        b_g_map: Mapping[int, bool],
        selected_clients: Optional[Sequence[int]] = None,
        confirmed_suspects: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        if (not self.risk_enabled()) or (not bool(self.config.get("risk_aware_sampling", True))):
            return {
                "risk_score_mean": float(self.risk_scores.mean().item()),
                "risk_score_max": float(self.risk_scores.max().item()),
                "risk_aware_source": str(self.config.get("risk_aware_source", "raw")),
                "risk_confirmed_count": int(len(confirmed_suspects or [])),
                "risk_signal_mean": 0.0,
                "risk_signal_min": 0.0,
                "risk_signal_max": 0.0,
                "risk_update_applied": False,
                "risk_update_skip_reason": "disabled",
            }

        beta = float(np.clip(self.config.get("risk_beta", 0.9), 0.0, 1.0))
        xi = float(max(self.config.get("risk_xi", 0.5), 0.0))
        decay = float(np.clip(self.config.get("risk_decay", 0.98), 0.0, 1.0))
        risk_source = str(self.config.get("risk_aware_source", "raw")).strip().lower()
        if risk_source not in {"raw", "confirmed"}:
            risk_source = "raw"
        confirmed_set = {
            int(client_id)
            for client_id in (confirmed_suspects or [])
            if 0 <= int(client_id) < int(self.num_clients)
        }

        total_count = np.zeros(self.num_clients, dtype=np.float32)
        suspicious_count = np.zeros(self.num_clients, dtype=np.float32)
        normal_count = np.zeros(self.num_clients, dtype=np.float32)

        for gid in group_ids:
            clients = [int(cid) for cid in group_members.get(int(gid), []) if 0 <= int(cid) < self.num_clients]
            if not clients:
                continue
            is_bad = bool(b_g_map.get(int(gid), False))
            for cid in clients:
                total_count[cid] += 1.0
                if is_bad:
                    suspicious_count[cid] += 1.0
                else:
                    normal_count[cid] += 1.0

        suspicious_score = np.divide(suspicious_count, np.maximum(total_count, 1.0))
        normal_score = np.divide(normal_count, np.maximum(total_count, 1.0))
        raw_signal = suspicious_score - xi * normal_score
        signal = np.array(raw_signal, dtype=np.float32, copy=True)
        update_applied = True
        skip_reason = "none"

        if risk_source == "confirmed":
            if len(confirmed_set) <= 0:
                signal = np.zeros_like(signal, dtype=np.float32)
                update_applied = False
                skip_reason = "no_confirmed_suspects"
            else:
                confirmed_mask = np.zeros(self.num_clients, dtype=np.float32)
                for client_id in confirmed_set:
                    confirmed_mask[int(client_id)] = 1.0
                confirmed_signal = np.array(suspicious_score, dtype=np.float32, copy=True) * confirmed_mask
                min_confirmed_count = int(max(self.config.get("risk_confirmed_min_count", 2), 0))
                min_confirmed_suspicious = float(
                    max(self.config.get("risk_confirmed_min_suspicious_score", 0.05), 0.0)
                )
                effective_confirmed_signal = float(np.max(confirmed_signal)) if confirmed_signal.size > 0 else 0.0
                if len(confirmed_set) < min_confirmed_count or effective_confirmed_signal < min_confirmed_suspicious:
                    fallback_source = str(self.config.get("risk_confirmed_fallback_source", "none")).strip().lower()
                    if fallback_source == "raw":
                        signal = np.array(raw_signal, dtype=np.float32, copy=True)
                        skip_reason = "confirmed_low_conf_fallback_raw"
                    else:
                        signal = np.zeros_like(signal, dtype=np.float32)
                        update_applied = False
                        skip_reason = "confirmed_low_conf_skip"
                else:
                    signal = confirmed_signal

        if not bool(self.config.get("risk_allow_negative_signal", False)):
            signal = np.clip(signal, a_min=0.0, a_max=None)
            if risk_source == "raw":
                skip_reason = "raw_clipped_nonnegative" if skip_reason == "none" else skip_reason
            elif risk_source == "confirmed" and skip_reason == "none":
                skip_reason = "confirmed_nonnegative"

        risk_np = self.risk_scores.detach().cpu().numpy()
        risk_np = beta * risk_np + (1.0 - beta) * signal

        if selected_clients is not None:
            selected_set = {int(cid) for cid in selected_clients if 0 <= int(cid) < self.num_clients}
            if len(selected_set) > 0:
                mask = np.ones(self.num_clients, dtype=np.float32)
                for cid in selected_set:
                    mask[cid] = 0.0
                risk_np = risk_np * (1.0 - mask * (1.0 - decay))
        else:
            risk_np *= decay

        self.risk_scores = torch.from_numpy(risk_np.astype(np.float32))
        signal_mean = float(np.mean(signal)) if signal.size > 0 else 0.0
        signal_min = float(np.min(signal)) if signal.size > 0 else 0.0
        signal_max = float(np.max(signal)) if signal.size > 0 else 0.0
        return {
            "risk_score_mean": float(self.risk_scores.mean().item()),
            "risk_score_max": float(self.risk_scores.max().item()),
            "risk_aware_source": str(risk_source),
            "risk_confirmed_count": int(len(confirmed_set)),
            "risk_signal_mean": float(signal_mean),
            "risk_signal_min": float(signal_min),
            "risk_signal_max": float(signal_max),
            "risk_update_applied": bool(update_applied),
            "risk_update_skip_reason": str(skip_reason),
        }

    def get_sampling_probabilities(self) -> np.ndarray:
        """Return risk-aware client sampling probabilities."""

        if (not self.risk_enabled()) or (not bool(self.config.get("risk_aware_sampling", True))):
            return np.full((self.num_clients,), 1.0 / float(max(self.num_clients, 1)), dtype=np.float64)

        gamma = float(max(self.config.get("risk_gamma", 2.0), 0.0))
        logits = -gamma * self.risk_scores.detach().cpu().numpy().astype(np.float64)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.maximum(np.sum(probs), 1e-12)

        p_min_cfg = float(self.config.get("risk_p_min", -1.0))
        if p_min_cfg <= 0.0:
            p_min = min(0.001, 1.0 / max(10.0 * float(self.num_clients), 1.0))
        else:
            # Compatibility: if user passes a ratio in (0,1], interpret it as
            # fraction of uniform mass instead of absolute probability.
            uniform_p = 1.0 / max(float(self.num_clients), 1.0)
            if 0.0 < p_min_cfg <= uniform_p:
                p_min = p_min_cfg
            elif 0.0 < p_min_cfg <= 1.0:
                p_min = p_min_cfg / max(float(self.num_clients), 1.0)
            else:
                p_min = p_min_cfg
        p_min = float(max(0.0, min(p_min, 1.0 / max(float(self.num_clients), 1.0))))
        probs = p_min + (1.0 - float(self.num_clients) * p_min) * probs
        probs = np.maximum(probs, 0.0)
        probs = probs / np.maximum(np.sum(probs), 1e-12)
        return probs

    def sample_clients(
        self,
        clients_per_round: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        k = int(clients_per_round)
        if k >= self.num_clients:
            return list(range(self.num_clients))
        if k <= 0:
            return []
        probs = self.get_sampling_probabilities()
        sampler = rng if rng is not None else self._sample_rng
        chosen = sampler.choice(self.num_clients, size=k, replace=False, p=probs)
        return [int(x) for x in chosen.tolist()]

    def _to_raw_inputs(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.detach().clone().float().to(self.device)
        if x.numel() == 0:
            return x
        min_value = float(x.min().item())
        max_value = float(x.max().item())
        looks_normalized = (min_value < -0.2) or (max_value > 1.2)
        if looks_normalized:
            mean = torch.tensor(self.config["normalization_mean"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            std = torch.tensor(self.config["normalization_std"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            x = x * std + mean
        return torch.clamp(x, 0.0, 1.0)

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config["normalization_mean"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.config["normalization_std"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return (x - mean) / std

    def _apply_trigger(self, x_raw: torch.Tensor, trigger_type: str, augmented: bool) -> torch.Tensor:
        trig = str(trigger_type).strip().lower()
        _, c, h, w = x_raw.shape
        out = x_raw.clone()
        if trig == "patch":
            size = int(self.config.get("patch_size", 3))
            if augmented:
                size = int(np.clip(size + int(torch.randint(-1, 2, (1,), generator=self._rng).item()), 2, min(h, w)))
                max_y = max(h - size, 0)
                max_x = max(w - size, 0)
                y0 = int(torch.randint(0, max_y + 1, (1,), generator=self._rng).item()) if max_y > 0 else 0
                x0 = int(torch.randint(0, max_x + 1, (1,), generator=self._rng).item()) if max_x > 0 else 0
            else:
                size = min(size, h, w)
                y0 = h - size
                x0 = w - size
            out[:, :, y0:y0 + size, x0:x0 + size] = 1.0
            return out
        if trig in {"blend", "noise"}:
            alpha_min = float(self.config.get("blend_alpha_min", 0.10))
            alpha_max = float(self.config.get("blend_alpha_max", 0.25))
            if augmented:
                alpha = float(alpha_min + (alpha_max - alpha_min) * torch.rand(1, generator=self._rng).item())
            else:
                alpha = float((alpha_min + alpha_max) * 0.5)
            noise = torch.rand(out.shape, generator=self._rng, dtype=out.dtype, device="cpu").to(out.device)
            out = torch.clamp((1.0 - alpha) * out + alpha * noise, 0.0, 1.0)
            return out
        if trig == "color":
            shift = float(self.config.get("color_shift", 16.0 / 255.0))
            if augmented:
                offsets = (torch.rand((1, c, 1, 1), generator=self._rng, device="cpu").to(out.device) * 2.0 - 1.0) * shift
            else:
                offsets = torch.full((1, c, 1, 1), shift * 0.5, device=out.device, dtype=out.dtype)
            out = torch.clamp(out + offsets, 0.0, 1.0)
            return out
        if trig == "occlusion":
            ratio = float(self.config.get("occlusion_ratio", 0.22))
            size = max(1, int(min(h, w) * ratio))
            if augmented:
                max_y = max(h - size, 0)
                max_x = max(w - size, 0)
                y0 = int(torch.randint(0, max_y + 1, (1,), generator=self._rng).item()) if max_y > 0 else 0
                x0 = int(torch.randint(0, max_x + 1, (1,), generator=self._rng).item()) if max_x > 0 else 0
            else:
                y0 = max((h - size) // 2, 0)
                x0 = max((w - size) // 2, 0)
            out[:, :, y0:y0 + size, x0:x0 + size] = 0.0
            return out
        amp_min = float(self.config.get("sig_amp_min", 0.03))
        amp_max = float(self.config.get("sig_amp_max", 0.05))
        amp = float(amp_min + (amp_max - amp_min) * torch.rand(1, generator=self._rng).item()) if augmented else float((amp_min + amp_max) * 0.5)
        freq_x = float(3.0 + torch.rand(1, generator=self._rng).item() * 3.0) if augmented else 4.0
        freq_y = float(1.0 + torch.rand(1, generator=self._rng).item() * 2.0) if augmented else 1.5
        phase = float(torch.rand(1, generator=self._rng).item() * 2.0 * np.pi) if augmented else 0.0
        ys = torch.linspace(0.0, 1.0, h, device=out.device, dtype=out.dtype).view(h, 1)
        xs = torch.linspace(0.0, 1.0, w, device=out.device, dtype=out.dtype).view(1, w)
        pattern_2d = torch.sin(2.0 * np.pi * (freq_x * xs + freq_y * ys) + phase).view(1, 1, h, w)
        out = torch.clamp(out + amp * pattern_2d, 0.0, 1.0)
        return out

    def build_augmented_triggers(self, x_syn: torch.Tensor, num_triggers: int, augmented: bool) -> List[torch.Tensor]:
        raw = self._to_raw_inputs(x_syn)
        trigger_types = [str(t).strip().lower() for t in self.config.get("trigger_types", ["patch", "blend", "color", "occlusion", "sig"])]
        trigger_types = [t for t in trigger_types if t]
        if len(trigger_types) == 0:
            trigger_types = ["patch", "blend", "color", "occlusion", "sig"]
        target_n = max(1, int(num_triggers))
        selected: List[str] = []
        if target_n <= len(trigger_types):
            perm = torch.randperm(len(trigger_types), generator=self._rng).tolist()
            for idx in perm[:target_n]:
                selected.append(trigger_types[idx])
        else:
            for _ in range(target_n):
                idx = int(torch.randint(0, len(trigger_types), (1,), generator=self._rng).item())
                selected.append(trigger_types[idx])
        out: List[torch.Tensor] = []
        for trig in selected:
            raw_trig = self._apply_trigger(raw, trig, augmented=augmented)
            out.append(self._normalize_inputs(raw_trig))
        return out

    def _apply_update_to_model(self, model: Module, update: ModelUpdate, scale: float) -> None:
        s = float(scale)
        with torch.no_grad():
            if isinstance(update, Mapping):
                named_parameters = dict(model.named_parameters())
                named_buffers = dict(model.named_buffers())
                for name, delta in update.items():
                    if name in named_parameters:
                        ref = named_parameters[name]
                        ref.add_(delta.to(device=ref.device, dtype=ref.dtype), alpha=s)
                    elif name in named_buffers:
                        refb = named_buffers[name]
                        refb.add_(delta.to(device=refb.device, dtype=refb.dtype), alpha=s)
            elif isinstance(update, torch.Tensor):
                raise TypeError("Tensor update format is not supported for model apply in DGC-Repair.")
            else:
                raise TypeError("Unsupported update type for _apply_update_to_model.")

    def _extract_update_from_models(
        self,
        *,
        model_before: Module,
        model_after: Module,
        reference_update: ModelUpdate,
        server_lr: float,
    ) -> ModelUpdate:
        inv = 1.0 / max(float(server_lr), 1e-8)
        if isinstance(reference_update, Mapping):
            before_state = model_before.state_dict()
            after_state = model_after.state_dict()
            update = OrderedDict()
            for key, ref_delta in reference_update.items():
                if key not in before_state or key not in after_state:
                    update[str(key)] = torch.zeros_like(ref_delta)
                    continue
                diff = (after_state[key].detach() - before_state[key].detach()) * inv
                update[str(key)] = diff.to(dtype=ref_delta.dtype, device=ref_delta.device)
            return update
        raise TypeError("Only mapping updates are supported for extraction in DGC-Repair.")

    def _quick_probe_accuracy(self, model: Module, clean_dataloader: Optional[DataLoader]) -> float:
        if clean_dataloader is None:
            return 0.0
        eval_model = model
        if bool(self.config.get("safe_probe_eval_use_bn_calibration", True)):
            # Probe-accuracy is used to drive safe-teacher stage transitions. A tiny BN
            # recalibration on clean batches makes this signal much less noisy and avoids
            # under-estimating a usable teacher as "weak forever".
            eval_model = deepcopy(model).to(self.device)
            bn_layers = [module for module in eval_model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
            calib_batches = max(int(self.config.get("safe_probe_bn_calibration_batches", 2)), 0)
            if len(bn_layers) > 0 and calib_batches > 0:
                eval_model.train()
                with torch.no_grad():
                    used_batches = 0
                    for batch in clean_dataloader:
                        if used_batches >= calib_batches:
                            break
                        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                            images = batch[0].to(self.device)
                        else:
                            continue
                        _ = eval_model(images)
                        used_batches += 1
        eval_model.eval()
        max_batches = max(int(self.config.get("max_probe_eval_batches", 4)), 1)
        random_offset = bool(self.config.get("safe_probe_eval_random_offset", True))
        start_batch = 0
        try:
            total_batches = int(len(clean_dataloader))  # type: ignore[arg-type]
        except Exception:
            total_batches = 0
        if random_offset and total_batches > max_batches:
            start_batch = int(
                torch.randint(
                    low=0,
                    high=max(total_batches - max_batches + 1, 1),
                    size=(1,),
                    generator=self._rng,
                ).item()
            )
        total = 0
        correct = 0
        processed = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(clean_dataloader):
                if batch_idx < start_batch:
                    continue
                if processed >= max_batches:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device).long()
                else:
                    continue
                outputs = eval_model(images)
                logits = self._extract_logits(outputs)
                pred = logits.argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
                processed += 1
            if processed < max_batches and start_batch > 0:
                for batch_idx, batch in enumerate(clean_dataloader):
                    if batch_idx >= start_batch:
                        break
                    if processed >= max_batches:
                        break
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        images = batch[0].to(self.device)
                        labels = batch[1].to(self.device).long()
                    else:
                        continue
                    outputs = eval_model(images)
                    logits = self._extract_logits(outputs)
                    pred = logits.argmax(dim=1)
                    correct += int((pred == labels).sum().item())
                    total += int(labels.numel())
                    processed += 1
        if total <= 0:
            return 0.0
        return float(correct / total)

    def _maybe_init_or_update_w_safe(
        self,
        *,
        round_idx: int,
        post_gate_model: Module,
        clean_dataloader: Optional[DataLoader],
        is_audit_round: bool,
        abnormal_group_ratio: float,
        confirmed_abnormal_ratio: float,
        confirmed_suspect_count: int,
        cts_global_light: float,
        p_hat: float,
        global_gate_rejected: bool,
    ) -> Tuple[bool, bool, float, Dict[str, Any]]:
        safe_initialized = False
        safe_updated = False
        probe_acc = 0.0
        stage_before = str(self.safe_model_stage)
        diag: Dict[str, Any] = {
            "safe_init_reason": "not_audit_round",
            "safe_update_reason": "not_audit_round",
            "safe_init_allowed": False,
            "safe_update_allowed": False,
            "safe_probe_acc": 0.0,
            "safe_init_used_fallback": False,
            "safe_init_source_model": "none",
            "safe_init_threshold_used": "none",
            "safe_init_blocking_condition": "none",
            "safe_init_context_ok": False,
            "safe_upgrade_allowed": False,
            "safe_upgrade_reason": "none",
            "safe_update_context_ok": False,
            "safe_signal_source": str(self.config.get("safe_signal_source", "dcbd")),
            "safe_signal_source_effective": "dcbd",
            "safe_update_ratio_source": str(self.config.get("safe_update_ratio_source", "confirmed")),
            "safe_update_ratio_source_effective": "raw",
            "safe_effective_abnormal_ratio": float(abnormal_group_ratio),
            "safe_raw_abnormal_ratio": float(abnormal_group_ratio),
            "safe_confirmed_abnormal_ratio": float(confirmed_abnormal_ratio),
            "safe_refresh_applied": False,
            "safe_refresh_reason": "disabled",
            "safe_refresh_stale_rounds": int(self.config.get("safe_refresh_stale_rounds", 20)),
            "safe_refresh_margin": float(self.config.get("safe_refresh_margin", 0.10)),
            "safe_refresh_ema": float(self.config.get("safe_refresh_ema", 0.5)),
            "safe_refresh_last_update_round": int(self.last_safe_update_round),
            "safe_model_stage_before": stage_before,
            "safe_model_stage_after": stage_before,
        }
        if not is_audit_round:
            diag["safe_model_update_reason"] = str(diag.get("safe_update_reason", "not_audit_round"))
            return safe_initialized, safe_updated, probe_acc, diag

        requested_ratio_source = str(self.config.get("safe_update_ratio_source", "confirmed")).strip().lower()
        if requested_ratio_source not in {"raw", "confirmed", "auto"}:
            requested_ratio_source = "confirmed"
        requested_signal_source = str(self.config.get("safe_signal_source", "dcbd")).strip().lower()
        if requested_signal_source not in {"raw", "dcbd", "fct"}:
            if requested_ratio_source in {"confirmed", "auto"}:
                requested_signal_source = "fct"
            else:
                requested_signal_source = "dcbd"
        confirmed_ratio_available = bool(float(confirmed_abnormal_ratio) >= 0.0)
        use_confirmed_ratio = bool(
            requested_signal_source == "fct"
            and confirmed_ratio_available
        )
        effective_abnormal_ratio = float(
            confirmed_abnormal_ratio if use_confirmed_ratio else abnormal_group_ratio
        )
        diag["safe_signal_source"] = str(requested_signal_source)
        diag["safe_signal_source_effective"] = (
            "fct" if use_confirmed_ratio else requested_signal_source
        )
        diag["safe_update_ratio_source"] = str(requested_ratio_source)
        diag["safe_update_ratio_source_effective"] = (
            "confirmed" if use_confirmed_ratio else "raw"
        )
        diag["safe_effective_abnormal_ratio"] = float(effective_abnormal_ratio)
        diag["safe_raw_abnormal_ratio"] = float(abnormal_group_ratio)
        diag["safe_confirmed_abnormal_ratio"] = float(confirmed_abnormal_ratio)

        safe_init_abnormal_ratio_max = float(self.config.get("safe_init_abnormal_ratio_max", 0.10))
        safe_init_cts_max = float(self.config.get("safe_init_cts_max", self.config.get("tau_safe", 1.5)))
        safe_init_p_hat_max = float(self.config.get("safe_init_p_hat_max", 0.12))
        weak_safe_init_probe_min = float(self.config.get("weak_safe_init_probe_min", 0.10))
        full_safe_init_probe_min = float(self.config.get("full_safe_init_probe_min", self.config.get("acc_init_min", 0.30)))
        full_safe_upgrade_max_reject_count = int(max(self.config.get("full_safe_upgrade_max_reject_count", 1), 0))
        full_safe_upgrade_probe_ratio = float(max(self.config.get("full_safe_upgrade_probe_ratio", 0.80), 0.0))
        full_safe_upgrade_fallback_probe_min = float(max(self.config.get("full_safe_upgrade_fallback_probe_min", 0.15), 0.0))
        full_safe_upgrade_min_wait_round = int(max(self.config.get("full_safe_upgrade_min_wait_round", 0), 0))
        safe_update_threshold = float(self.config.get("safe_update_abnormal_ratio_max", self.config.get("safe_update_threshold", 0.10)))
        tau_safe = float(self.config.get("safe_update_cts_max", self.config.get("tau_safe", 1.5)))
        safe_update_p_hat_max = float(self.config.get("safe_update_p_hat_max", 0.08))
        block_safe_update_in_alert_mode = bool(self.config.get("block_safe_update_in_alert_mode", True))
        safe_update_max_probe_drop = float(max(self.config.get("safe_update_max_probe_drop", 0.03), 0.0))
        if clean_dataloader is not None:
            probe_acc = self._quick_probe_accuracy(post_gate_model, clean_dataloader)
        diag["safe_probe_acc"] = float(probe_acc)
        acc_init_min = float(self.config.get("acc_init_min", 0.30))
        acc_init_fallback_min = float(self.config.get("acc_init_fallback_min", 0.10))
        safe_init_max_wait_round = int(self.config.get("safe_init_max_wait_round", 40))
        allow_safe_init_after_wait = bool(self.config.get("allow_safe_init_after_wait", True))
        allow_safe_init_without_probe = bool(self.config.get("allow_safe_init_without_probe", True))
        safe_init_fallback_ratio = float(max(self.config.get("safe_init_fallback_ratio", 0.50), 0.0))
        fallback_min_effective = max(acc_init_fallback_min, acc_init_min * safe_init_fallback_ratio)
        fallback_allow_relaxed_context = bool(
            self.config.get("safe_init_fallback_allow_relaxed_context", True)
        )
        fallback_relaxed_abnormal_max = float(
            max(self.config.get("safe_init_fallback_relaxed_abnormal_ratio_max", 0.20), 0.0)
        )
        fallback_relaxed_cts_max = float(self.config.get("safe_init_fallback_relaxed_cts_max", 3.0))
        fallback_relaxed_p_hat_max = float(
            max(self.config.get("safe_init_fallback_relaxed_p_hat_max", 0.20), 0.0)
        )
        safe_context_ok = bool(
            float(effective_abnormal_ratio) <= safe_init_abnormal_ratio_max
            and float(cts_global_light) <= safe_init_cts_max
            and float(p_hat) <= safe_init_p_hat_max
            and (not bool(global_gate_rejected))
        )
        safe_context_ok_relaxed = bool(
            float(effective_abnormal_ratio) <= fallback_relaxed_abnormal_max
            and float(cts_global_light) <= fallback_relaxed_cts_max
            and float(p_hat) <= fallback_relaxed_p_hat_max
            and (not bool(global_gate_rejected))
        )
        safe_context_ok_for_fallback = bool(
            safe_context_ok
            or (fallback_allow_relaxed_context and safe_context_ok_relaxed)
        )
        diag.update({
            "acc_init_min": float(acc_init_min),
            "acc_init_fallback_min": float(acc_init_fallback_min),
            "safe_init_fallback_ratio": float(safe_init_fallback_ratio),
            "safe_init_fallback_min_effective": float(fallback_min_effective),
            "weak_safe_init_probe_min": float(weak_safe_init_probe_min),
            "full_safe_init_probe_min": float(full_safe_init_probe_min),
            "safe_init_max_wait_round": int(safe_init_max_wait_round),
            "full_safe_upgrade_probe_ratio": float(full_safe_upgrade_probe_ratio),
            "full_safe_upgrade_fallback_probe_min": float(full_safe_upgrade_fallback_probe_min),
            "full_safe_upgrade_min_wait_round": int(full_safe_upgrade_min_wait_round),
            "safe_init_context_ok": bool(safe_context_ok),
            "safe_init_context_ok_relaxed": bool(safe_context_ok_relaxed),
            "safe_init_context_ok_for_fallback": bool(safe_context_ok_for_fallback),
            "safe_init_fallback_allow_relaxed_context": bool(fallback_allow_relaxed_context),
            "safe_init_fallback_relaxed_abnormal_ratio_max": float(fallback_relaxed_abnormal_max),
            "safe_init_fallback_relaxed_cts_max": float(fallback_relaxed_cts_max),
            "safe_init_fallback_relaxed_p_hat_max": float(fallback_relaxed_p_hat_max),
            "safe_init_abnormal_ratio_max": float(safe_init_abnormal_ratio_max),
            "safe_init_cts_max": float(safe_init_cts_max),
            "safe_init_p_hat_max": float(safe_init_p_hat_max),
            "full_safe_upgrade_max_reject_count": int(full_safe_upgrade_max_reject_count),
            "safe_update_abnormal_ratio_max": float(safe_update_threshold),
            "safe_update_cts_max": float(tau_safe),
            "safe_update_p_hat_max": float(safe_update_p_hat_max),
            "safe_update_max_probe_drop": float(safe_update_max_probe_drop),
            "block_safe_update_in_alert_mode": bool(block_safe_update_in_alert_mode),
        })
        if self.w_safe is None:
            init_allowed = False
            init_stage = "none"
            if clean_dataloader is None and allow_safe_init_without_probe:
                if safe_context_ok:
                    init_allowed = True
                    diag["safe_init_reason"] = "no_probe_allowed"
                    diag["safe_init_threshold_used"] = "no_probe"
                    init_stage = "weak"
                else:
                    diag["safe_init_reason"] = "no_probe_but_context_unsafe"
                    diag["safe_init_blocking_condition"] = "unsafe_context"
            elif probe_acc >= full_safe_init_probe_min:
                diag["safe_init_threshold_used"] = "full_safe_init_probe_min"
                if safe_context_ok:
                    init_allowed = True
                    init_stage = "full"
                else:
                    diag["safe_init_reason"] = "probe_acc_ok_but_context_unsafe"
                    diag["safe_init_blocking_condition"] = "unsafe_context"
            elif probe_acc >= weak_safe_init_probe_min:
                diag["safe_init_threshold_used"] = "weak_safe_init_probe_min"
                if safe_context_ok:
                    init_allowed = True
                    init_stage = "weak"
                else:
                    diag["safe_init_reason"] = "probe_acc_ok_but_context_unsafe"
                    diag["safe_init_blocking_condition"] = "unsafe_context"
            if init_allowed and diag["safe_init_reason"] == "not_audit_round":
                diag["safe_init_reason"] = (
                    "probe_acc_ge_full_safe_init_probe_min"
                    if init_stage == "full"
                    else "probe_acc_ge_weak_safe_init_probe_min"
                )
            elif (
                allow_safe_init_after_wait
                and int(round_idx) >= safe_init_max_wait_round
                and probe_acc >= fallback_min_effective
                and safe_context_ok_for_fallback
            ):
                init_allowed = True
                diag["safe_init_reason"] = "fallback_after_wait"
                diag["safe_init_used_fallback"] = True
                diag["safe_init_threshold_used"] = "fallback_min_effective"
                init_stage = "weak"
            else:
                if not allow_safe_init_after_wait:
                    diag["safe_init_reason"] = "fallback_disabled"
                    diag["safe_init_blocking_condition"] = "fallback_disabled"
                elif int(round_idx) < safe_init_max_wait_round:
                    diag["safe_init_reason"] = "fallback_waiting"
                    diag["safe_init_blocking_condition"] = "wait_round_not_reached"
                elif probe_acc < fallback_min_effective:
                    diag["safe_init_reason"] = "fallback_probe_too_low"
                    diag["safe_init_blocking_condition"] = "probe_below_fallback_threshold"
                elif not safe_context_ok:
                    diag["safe_init_reason"] = "fallback_unsafe_context"
                    diag["safe_init_blocking_condition"] = "unsafe_context"
                else:
                    diag["safe_init_reason"] = "init_not_allowed"
                    diag["safe_init_blocking_condition"] = "unknown"
            diag["safe_init_allowed"] = bool(init_allowed)
            if init_allowed:
                self.w_safe = deepcopy(post_gate_model).to(self.device)
                self.w_safe.eval()
                for p in self.w_safe.parameters():
                    p.requires_grad_(False)
                safe_initialized = True
                self.safe_model_stage = str(init_stage)
                self.last_safe_probe_acc = float(probe_acc)
                self.last_safe_update_round = int(round_idx)
                diag["safe_init_source_model"] = "post_gate_model"
                diag["safe_model_update_reason"] = "initialized"
                diag["safe_model_stage_after"] = str(self.safe_model_stage)
                return safe_initialized, safe_updated, float(probe_acc), diag
            diag["safe_model_stage_after"] = str(self.safe_model_stage)
            diag["safe_model_update_reason"] = str(diag.get("safe_update_reason", "none"))
            return safe_initialized, safe_updated, float(probe_acc), diag

        # stage-aware safe update policy: weak stage is intentionally more permissive
        is_full_stage = str(self.safe_model_stage) == "full"
        update_abnormal_max = (
            safe_update_threshold
            if is_full_stage
            else float(max(self.config.get("safe_update_abnormal_ratio_max_weak", 0.25), safe_update_threshold))
        )
        update_cts_max = (
            tau_safe
            if is_full_stage
            else float(max(self.config.get("safe_update_cts_max_weak", 2.5), tau_safe))
        )
        update_p_hat_max = (
            safe_update_p_hat_max
            if is_full_stage
            else float(max(self.config.get("safe_update_p_hat_max_weak", 0.25), safe_update_p_hat_max))
        )
        enforce_alert_block = bool(block_safe_update_in_alert_mode and is_full_stage)
        safe_update_context_ok = bool(
            float(effective_abnormal_ratio) <= update_abnormal_max
            and float(cts_global_light) <= update_cts_max
            and float(p_hat) <= update_p_hat_max
            and (not enforce_alert_block or not bool(self.recent_alert_mode))
            and (not bool(global_gate_rejected))
        )
        probe_not_degraded = bool(
            self.last_safe_probe_acc <= 0.0
            or float(probe_acc) + safe_update_max_probe_drop >= float(self.last_safe_probe_acc)
        )
        update_allowed = bool(safe_update_context_ok and probe_not_degraded)
        diag["safe_update_context_ok"] = bool(safe_update_context_ok)
        diag["safe_update_allowed"] = bool(update_allowed)
        diag["safe_update_abnormal_ratio_max_effective"] = float(update_abnormal_max)
        diag["safe_update_cts_max_effective"] = float(update_cts_max)
        diag["safe_update_p_hat_max_effective"] = float(update_p_hat_max)
        if update_allowed:
            ema_alpha = float(np.clip(self.config.get("safe_ema_alpha", 0.99), 0.0, 1.0))
            with torch.no_grad():
                safe_params = dict(self.w_safe.named_parameters())
                curr_params = dict(post_gate_model.named_parameters())
                for name, p_safe in safe_params.items():
                    if name in curr_params:
                        p_safe.mul_(ema_alpha).add_(
                            curr_params[name].detach().to(device=p_safe.device, dtype=p_safe.dtype),
                            alpha=(1.0 - ema_alpha),
                        )
                safe_buffers = dict(self.w_safe.named_buffers())
                curr_buffers = dict(post_gate_model.named_buffers())
                for name, b_safe in safe_buffers.items():
                    if name in curr_buffers:
                        b_safe.copy_(curr_buffers[name].detach().to(device=b_safe.device, dtype=b_safe.dtype))
            safe_updated = True
            self.last_safe_probe_acc = float(probe_acc)
            self.last_safe_update_round = int(round_idx)
            diag["safe_update_reason"] = "ema_update_applied"
        else:
            if float(effective_abnormal_ratio) > update_abnormal_max:
                diag["safe_update_reason"] = "abnormal_ratio_too_high"
            elif float(cts_global_light) > update_cts_max:
                diag["safe_update_reason"] = "cts_global_too_high"
            elif float(p_hat) > update_p_hat_max:
                diag["safe_update_reason"] = "p_hat_too_high"
            elif bool(global_gate_rejected):
                diag["safe_update_reason"] = "global_gate_rejected"
            elif enforce_alert_block and bool(self.recent_alert_mode):
                diag["safe_update_reason"] = "blocked_in_alert_mode"
            elif not probe_not_degraded:
                diag["safe_update_reason"] = "probe_acc_degraded"
            else:
                diag["safe_update_reason"] = "update_not_allowed"

        safe_refresh_enable = bool(self.config.get("safe_refresh_enable", False))
        safe_refresh_stale_rounds = int(max(self.config.get("safe_refresh_stale_rounds", 20), 1))
        safe_refresh_margin = float(max(self.config.get("safe_refresh_margin", 0.10), 0.0))
        safe_refresh_ema = float(np.clip(self.config.get("safe_refresh_ema", 0.5), 0.0, 1.0))
        safe_refresh_confirmed_abnormal_ratio_max = float(
            np.clip(self.config.get("safe_refresh_confirmed_abnormal_ratio_max", 0.20), 0.0, 1.0)
        )
        safe_refresh_cts_max = float(self.config.get("safe_refresh_cts_max", 2.0))
        safe_refresh_max_confirmed = int(max(self.config.get("safe_refresh_max_confirmed", 3), 0))
        rounds_since_safe_update = (
            int(round_idx) - int(self.last_safe_update_round)
            if int(self.last_safe_update_round) >= 0
            else int(round_idx) + 1
        )
        confirmed_ratio_for_refresh = float(
            confirmed_abnormal_ratio
            if float(confirmed_abnormal_ratio) >= 0.0
            else effective_abnormal_ratio
        )
        stale_enough = bool(rounds_since_safe_update >= safe_refresh_stale_rounds)
        probe_gain_ok = bool(float(probe_acc) > float(self.last_safe_probe_acc) + safe_refresh_margin)
        collapse_blocked = bool(
            bool(global_gate_rejected)
            or bool(self.recent_alert_mode)
            or float(cts_global_light) > float(safe_refresh_cts_max)
        )
        confirmed_ratio_ok = bool(
            float(confirmed_ratio_for_refresh) <= safe_refresh_confirmed_abnormal_ratio_max
        )
        confirmed_count_ok = bool(int(max(confirmed_suspect_count, 0)) <= safe_refresh_max_confirmed)
        safe_refresh_applied = False
        safe_refresh_reason = "disabled"
        if safe_refresh_enable and (not safe_updated):
            if (
                stale_enough
                and probe_gain_ok
                and (not collapse_blocked)
                and confirmed_ratio_ok
                and confirmed_count_ok
            ):
                with torch.no_grad():
                    safe_params = dict(self.w_safe.named_parameters())
                    curr_params = dict(post_gate_model.named_parameters())
                    for name, p_safe in safe_params.items():
                        if name in curr_params:
                            p_safe.mul_(1.0 - safe_refresh_ema).add_(
                                curr_params[name].detach().to(device=p_safe.device, dtype=p_safe.dtype),
                                alpha=safe_refresh_ema,
                            )
                    safe_buffers = dict(self.w_safe.named_buffers())
                    curr_buffers = dict(post_gate_model.named_buffers())
                    for name, b_safe in safe_buffers.items():
                        if name in curr_buffers:
                            b_safe.copy_(curr_buffers[name].detach().to(device=b_safe.device, dtype=b_safe.dtype))
                safe_updated = True
                safe_refresh_applied = True
                safe_refresh_reason = "stale_recovery"
                self.last_safe_probe_acc = float(probe_acc)
                self.last_safe_update_round = int(round_idx)
                diag["safe_update_reason"] = "ema_refresh_stale"
            elif not stale_enough:
                safe_refresh_reason = "not_stale"
            elif not probe_gain_ok:
                safe_refresh_reason = "probe_margin_not_met"
            elif collapse_blocked:
                safe_refresh_reason = "collapse_guard_blocked"
            elif not confirmed_ratio_ok:
                safe_refresh_reason = "confirmed_abnormal_ratio_too_high"
            elif not confirmed_count_ok:
                safe_refresh_reason = "confirmed_count_too_high"
            else:
                safe_refresh_reason = "not_allowed"
        elif safe_refresh_enable and safe_updated:
            safe_refresh_reason = "skipped_already_updated"
        diag["safe_refresh_applied"] = bool(safe_refresh_applied)
        diag["safe_refresh_reason"] = str(safe_refresh_reason)
        diag["safe_refresh_stale_rounds"] = int(safe_refresh_stale_rounds)
        diag["safe_refresh_margin"] = float(safe_refresh_margin)
        diag["safe_refresh_ema"] = float(safe_refresh_ema)
        diag["safe_refresh_last_update_round"] = int(self.last_safe_update_round)
        diag["safe_refresh_rounds_since_last_update"] = int(rounds_since_safe_update)
        diag["safe_refresh_confirmed_abnormal_ratio"] = float(confirmed_ratio_for_refresh)
        diag["safe_refresh_confirmed_abnormal_ratio_max"] = float(
            safe_refresh_confirmed_abnormal_ratio_max
        )
        diag["safe_refresh_confirmed_count"] = int(max(confirmed_suspect_count, 0))
        diag["safe_refresh_max_confirmed"] = int(safe_refresh_max_confirmed)
        diag["safe_refresh_cts_max"] = float(safe_refresh_cts_max)
        diag["safe_model_update_reason"] = str(diag.get("safe_update_reason", "none"))

        # Promote weak -> full once quality and stability are met.
        full_upgrade_probe_threshold = max(
            float(full_safe_init_probe_min) * float(full_safe_upgrade_probe_ratio),
            float(full_safe_upgrade_fallback_probe_min),
        )
        full_safe_upgrade_relax_start_round = int(
            max(self.config.get("full_safe_upgrade_relax_start_round", 60), 0)
        )
        full_safe_upgrade_relax_probe_min = float(
            max(self.config.get("full_safe_upgrade_relax_probe_min", 0.12), 0.0)
        )
        full_safe_upgrade_use_relaxed_context = bool(
            self.config.get("full_safe_upgrade_use_relaxed_context", True)
        )
        relaxed_upgrade_active = bool(int(self.current_round) >= int(full_safe_upgrade_relax_start_round))
        full_upgrade_probe_threshold_effective = float(full_upgrade_probe_threshold)
        if relaxed_upgrade_active:
            full_upgrade_probe_threshold_effective = min(
                full_upgrade_probe_threshold_effective,
                full_safe_upgrade_relax_probe_min,
            )
        upgrade_context_ok = bool(safe_context_ok_relaxed) if full_safe_upgrade_use_relaxed_context else bool(safe_context_ok)
        safe_upgrade_allowed = bool(
            str(self.safe_model_stage) == "weak"
            and upgrade_context_ok
            and float(probe_acc) >= full_upgrade_probe_threshold_effective
            and int(self.current_round) >= int(full_safe_upgrade_min_wait_round)
            and int(self.consecutive_reject_count) <= int(full_safe_upgrade_max_reject_count)
        )
        diag["safe_upgrade_allowed"] = bool(safe_upgrade_allowed)
        diag["full_safe_upgrade_relax_start_round"] = int(full_safe_upgrade_relax_start_round)
        diag["full_safe_upgrade_relax_probe_min"] = float(full_safe_upgrade_relax_probe_min)
        diag["full_safe_upgrade_use_relaxed_context"] = bool(full_safe_upgrade_use_relaxed_context)
        diag["full_safe_upgrade_relaxed_active"] = bool(relaxed_upgrade_active)
        diag["full_safe_upgrade_threshold_effective"] = float(full_upgrade_probe_threshold_effective)
        diag["full_safe_upgrade_context_ok"] = bool(upgrade_context_ok)
        if safe_upgrade_allowed:
            self.safe_model_stage = "full"
            diag["safe_update_reason"] = "upgraded_to_full"
            diag["safe_upgrade_reason"] = "probe_and_context_ready"
        else:
            if str(self.safe_model_stage) != "weak":
                diag["safe_upgrade_reason"] = "stage_not_weak"
            elif not upgrade_context_ok:
                diag["safe_upgrade_reason"] = "unsafe_context"
            elif float(probe_acc) < full_upgrade_probe_threshold_effective:
                diag["safe_upgrade_reason"] = "probe_below_upgrade_threshold"
            elif int(self.current_round) < int(full_safe_upgrade_min_wait_round):
                diag["safe_upgrade_reason"] = "upgrade_wait_round_not_reached"
            elif int(self.consecutive_reject_count) > int(full_safe_upgrade_max_reject_count):
                diag["safe_upgrade_reason"] = "consecutive_reject_too_high"
            else:
                diag["safe_upgrade_reason"] = "upgrade_not_allowed"
        diag["safe_model_stage_after"] = str(self.safe_model_stage)
        diag["safe_model_update_reason"] = str(diag.get("safe_update_reason", "none"))
        return safe_initialized, safe_updated, float(probe_acc), diag

    def _compute_cts_global_light(
        self,
        *,
        global_model: Module,
        global_update: ModelUpdate,
        clean_dataloader: Optional[DataLoader],
        round_idx: int,
        target_label: Optional[int],
    ) -> float:
        probe_all = self.cts_intent._resolve_probe_set(clean_dataloader=clean_dataloader, metadata={})
        probe_size = min(int(self.config.get("probe_size_light", 128)), int(probe_all.shape[0]))
        if probe_size <= 0:
            return float(self.last_cts_global_light)
        perm = torch.randperm(int(probe_all.shape[0]), generator=self._rng)[:probe_size]
        probe = probe_all.index_select(0, perm.to(device=probe_all.device)).detach()

        trigger_types = [str(t).strip().lower() for t in self.config.get("trigger_types", ["patch", "blend", "color", "occlusion", "sig"])]
        if len(trigger_types) == 0:
            trigger_types = ["patch", "blend"]
        n_trig = min(max(int(self.config.get("num_triggers_light", 2)), 1), len(trigger_types))
        perm_t = torch.randperm(len(trigger_types), generator=self._rng).tolist()
        selected_trigs = [trigger_types[idx] for idx in perm_t[:n_trig]]

        original_trigger_types = list(self.cts_intent.config.get("trigger_types", ["patch", "blend", "sig", "color"]))
        original_probe_size = int(self.cts_intent.config.get("probe_size", 100))
        try:
            self.cts_intent.config["trigger_types"] = selected_trigs
            self.cts_intent.config["probe_size"] = int(probe_size)
            result = self.cts_intent.detect(
                group_updates=OrderedDict([(0, global_update)]),
                global_model=global_model,
                clean_dataloader=clean_dataloader,
                round_idx=int(round_idx),
                metadata={
                    "x_syn": probe,
                    "known_target_label": target_label,
                    "force_known_target": bool(target_label is not None),
                },
            )
            return float(result.group_scores.get(0, 0.0))
        finally:
            self.cts_intent.config["trigger_types"] = original_trigger_types
            self.cts_intent.config["probe_size"] = int(original_probe_size)

    def repair_round_update(
        self,
        *,
        round_idx: int,
        global_model: Module,
        aggregated_update: ModelUpdate,
        clean_dataloader: Optional[DataLoader],
        p_hat: float,
        target_label: Optional[int],
        is_audit_round: bool,
        abnormal_group_ratio: float,
        confirmed_abnormal_ratio: float,
        confirmed_suspect_count: int = 0,
        current_effective_lr: float,
        server_lr: float,
    ) -> Tuple[ModelUpdate, Dict[str, Any]]:
        """Apply Layer-3 repair and return cleaned round update."""

        gate_stats = dict(self.last_gate_stats)
        metrics = {
            "gate_mean": float(gate_stats.get("gate_mean", 1.0)),
            "gate_min": float(gate_stats.get("gate_min", 1.0)),
            "gate_reject_count": int(gate_stats.get("gate_reject_count", 0)),
            "gate_floor_activated": bool(gate_stats.get("gate_floor_activated", False)),
            "effective_ratio": float(gate_stats.get("effective_ratio", 1.0)),
            "num_abnormal_groups": int(gate_stats.get("num_abnormal_groups", 0)),
            "abnormal_group_ratio": float(gate_stats.get("abnormal_group_ratio", float(abnormal_group_ratio))),
            "confirmed_abnormal_ratio": float(
                gate_stats.get("confirmed_abnormal_ratio", float(confirmed_abnormal_ratio))
            ),
            "p_hat": float(p_hat),
            "risk_score_mean": float(self.risk_scores.mean().item()),
            "risk_score_max": float(self.risk_scores.max().item()),
            "risk_aware_source": str(self.config.get("risk_aware_source", "raw")),
            "risk_confirmed_count": 0,
            "safe_signal_source": str(self.config.get("safe_signal_source", "dcbd")),
            "safe_signal_source_effective": "dcbd",
            "safe_update_ratio_source": str(self.config.get("safe_update_ratio_source", "confirmed")),
            "safe_initialized": False,
            "safe_updated": False,
            "safe_model_exists": bool(self.w_safe is not None),
            "safe_model_ready": bool(str(self.safe_model_stage) == "full"),
            "safe_model_stage": str(self.safe_model_stage),
            "safe_bootstrap_mode": bool(str(self.safe_model_stage) == "weak"),
            "bootstrap_repair_enabled": False,
            "global_gate_rejected": False,
        }

        base_update = self.last_gate_update if (is_audit_round and self.last_gate_update is not None and self.gate_enabled()) else aggregated_update
        working_update = self._clone_update(base_update)

        check_interval = max(int(self.config.get("check_interval", 5)), 1)
        should_check = bool((int(round_idx) + 1) % check_interval == 0 or int(round_idx) <= int(self.alert_until_round))
        cts_checked_this_round = False
        if should_check:
            self.last_cts_global_light = self._compute_cts_global_light(
                global_model=global_model,
                global_update=working_update,
                clean_dataloader=clean_dataloader,
                round_idx=int(round_idx),
                target_label=target_label,
            )
            self.last_cts_checked_round = int(round_idx)
            cts_checked_this_round = True

        cts_global_light = float(self.last_cts_global_light)
        cts_ema_beta = float(np.clip(self.config.get("cts_light_ema_beta", 0.9), 0.0, 0.9999))
        cts_std_eps = float(max(self.config.get("cts_light_ema_std_eps", 1e-3), 1e-8))
        if cts_checked_this_round:
            if not bool(self.cts_light_ema_initialized):
                self.cts_light_ema_mean = float(cts_global_light)
                self.cts_light_ema_var = 1.0
                self.cts_light_ema_initialized = True
            else:
                delta = float(cts_global_light - self.cts_light_ema_mean)
                self.cts_light_ema_mean = float(
                    cts_ema_beta * float(self.cts_light_ema_mean)
                    + (1.0 - cts_ema_beta) * float(cts_global_light)
                )
                self.cts_light_ema_var = float(
                    cts_ema_beta * float(self.cts_light_ema_var)
                    + (1.0 - cts_ema_beta) * float(delta * delta)
                )
        cts_ema_std = float(np.sqrt(max(float(self.cts_light_ema_var), cts_std_eps * cts_std_eps)))
        cts_global_light_centered = float(cts_global_light - float(self.cts_light_ema_mean))
        cts_global_light_z = float(cts_global_light_centered / max(float(cts_ema_std), cts_std_eps))
        self.last_cts_global_light_centered = float(cts_global_light_centered)
        self.last_cts_global_light_z = float(cts_global_light_z)

        cts_signal_fresh = bool(cts_checked_this_round or is_audit_round)
        tau_alert = float(self.config.get("tau_alert", 0.6))
        tau_reject = float(self.config.get("tau_reject", 1.5))
        tau_alert_z = float(self.config.get("tau_alert_z", 1.5))
        tau_reject_z = float(self.config.get("tau_reject_z", 3.0))
        alert_horizon = max(int(self.config.get("alert_horizon", 3)), 0)
        reject_confirm_rounds = max(int(self.config.get("reject_confirm_rounds", 2)), 1)
        no_safe_high_risk_branch = str(self.config.get("no_safe_high_risk_branch", "gated")).strip().lower()
        no_safe_alert_scale = float(np.clip(self.config.get("no_safe_alert_scale", 0.70), 0.0, 1.0))
        no_safe_noop_max_rounds = max(int(self.config.get("no_safe_noop_max_rounds", 2)), 0)

        alert_mode_reason = "none"
        if is_audit_round:
            p_hat_alert = float(self.config.get("p_hat_alert", 0.15))
            if abnormal_group_ratio > 0.0 or float(p_hat) > p_hat_alert:
                self.alert_until_round = int(round_idx) + alert_horizon
                alert_mode_reason = "audit_abnormal_or_high_p_hat"
        cts_alert_signal = bool(cts_signal_fresh and (cts_global_light > tau_alert or cts_global_light_z > tau_alert_z))
        if cts_alert_signal:
            self.alert_until_round = max(self.alert_until_round, int(round_idx) + alert_horizon)
            if alert_mode_reason == "none":
                alert_mode_reason = "cts_global_exceeds_tau_alert"
        self.recent_alert_mode = bool(int(round_idx) <= int(self.alert_until_round))
        if self.recent_alert_mode and alert_mode_reason == "none":
            alert_mode_reason = "within_alert_window"

        abnormal_detected = bool(is_audit_round and abnormal_group_ratio > 0.0)
        reject_signal = bool(cts_signal_fresh and (cts_global_light > tau_reject or cts_global_light_z > tau_reject_z))
        if bool(reject_signal):
            self.reject_counter = int(self.reject_counter) + 1
        else:
            self.reject_counter = 0
        hard_reject_ready = bool(
            str(self.safe_model_stage) == "full"
            and int(self.reject_counter) >= int(reject_confirm_rounds)
        )
        reject_triggered = bool(reject_signal and hard_reject_ready)
        reject_allowed = True
        reject_blocked_by_no_safe_model = False
        reject_fallback_branch_used = False
        reject_block_reason = "none"
        final_update_branch_hint = "repair"
        reject_reason = "none"

        if reject_triggered and reject_allowed:
            working_update = self._zero_update_like(working_update)
            metrics["global_gate_rejected"] = True
            reject_reason = "cts_global_exceeds_tau_reject"
            final_update_branch_hint = "reject"
            state = "abnormal"
        elif reject_signal and str(self.safe_model_stage) != "full":
            # No full safe teacher yet: never hard reject. Use controlled fallback.
            reject_allowed = False
            reject_reason = "reject_blocked_no_safe_model"
            reject_blocked_by_no_safe_model = True
            reject_fallback_branch_used = True
            fallback_scale = float(np.clip(self.config.get("reject_fallback_scale_without_safe", 0.25), 0.0, 1.0))
            use_gated = str(no_safe_high_risk_branch) == "gated"
            if (
                (not use_gated)
                and int(no_safe_noop_max_rounds) >= 0
                and int(self.reject_counter) > int(no_safe_noop_max_rounds)
            ):
                # Prevent deadlock: after repeated no-safe high-risk rounds, force gated update.
                use_gated = True
                reject_block_reason = "safe_model_stage_not_full_noop_cap_force_gated"
            if use_gated:
                working_update = self._scale_update(working_update, fallback_scale)
                final_update_branch_hint = "gated"
                if reject_block_reason == "none":
                    reject_block_reason = "safe_model_stage_not_full_gated_fallback"
            else:
                working_update = self._zero_update_like(working_update)
                final_update_branch_hint = "noop"
                if reject_block_reason == "none":
                    reject_block_reason = "safe_model_stage_not_full_noop_fallback"
            state = "alert"
        elif abnormal_detected:
            state = "abnormal"
        elif bool(self.recent_alert_mode):
            if str(self.safe_model_stage) == "none" and no_safe_alert_scale < 0.9999:
                working_update = self._scale_update(working_update, no_safe_alert_scale)
                final_update_branch_hint = "gated"
            state = "alert"
        else:
            final_update_branch_hint = "raw"
            state = "normal"

        steps = int(self.config.get("steps_normal", 1))
        if state == "alert":
            steps = int(self.config.get("steps_alert", 3))
        elif state == "abnormal":
            steps = int(self.config.get("steps_abnormal", 5))

        lambda_min = float(self.config.get("lambda_min", 0.05))
        lambda_medium = float(self.config.get("lambda_medium", 0.5))
        lambda_max = float(self.config.get("lambda_max", 2.0))
        p_max = max(float(self.config.get("p_max", 0.3)), 1e-6)
        lambda_branch = "normal"
        if reject_triggered and reject_allowed:
            lambda_bd = lambda_max
            lambda_branch = "reject"
        elif abnormal_detected:
            lambda_bd = lambda_max * float(np.clip(float(p_hat) / p_max, 0.0, 1.0))
            lambda_branch = "abnormal"
        elif bool(self.recent_alert_mode):
            lambda_bd = lambda_medium
            lambda_branch = "alert"
        else:
            lambda_bd = lambda_min

        lr_ratio = float(self.config.get("repair_lr_ratio_normal", 0.01))
        if state == "alert":
            lr_ratio = float(self.config.get("repair_lr_ratio_alert", 0.03))
        elif state == "abnormal":
            lr_ratio = float(self.config.get("repair_lr_ratio_abnormal", 0.05))
        repair_lr = float(np.clip(
            float(current_effective_lr) * lr_ratio,
            float(self.config.get("repair_lr_min", 5e-5)),
            float(self.config.get("repair_lr_max", 2e-3)),
        ))

        post_gate_model = deepcopy(global_model).to(self.device)
        self._apply_update_to_model(post_gate_model, working_update, scale=float(server_lr))
        safe_initialized, safe_updated, safe_probe_acc, safe_diag = self._maybe_init_or_update_w_safe(
            round_idx=int(round_idx),
            post_gate_model=post_gate_model,
            clean_dataloader=clean_dataloader,
            is_audit_round=is_audit_round,
            abnormal_group_ratio=float(abnormal_group_ratio),
            confirmed_abnormal_ratio=float(confirmed_abnormal_ratio),
            confirmed_suspect_count=int(max(confirmed_suspect_count, 0)),
            cts_global_light=float(cts_global_light),
            p_hat=float(p_hat),
            global_gate_rejected=bool(metrics.get("global_gate_rejected", False)),
        )

        l_kd = 0.0
        l_anti_bd = 0.0
        l_reg = 0.0
        repaired_update = self._clone_update(working_update)
        repair_state_before_safe_check = str(state)
        repair_state = str(state)
        repair_state_after_safe_check = str(state)
        repair_disabled_reason = "none"
        lambda_disabled_reason = "none"
        steps_planned = int(steps)
        steps_after_safe_cap = int(steps)
        steps_after_lambda_cap = int(steps)
        lambda_base_before_safe_scale = float(lambda_bd)
        lambda_safe_scale_ratio = 1.0
        lambda_safe_scale_floor = 1.0
        lambda_cap_reason = "none"
        safe_stage_current = str(self.safe_model_stage)
        bootstrap_repair_enabled = bool(safe_stage_current == "weak")
        # Before the first safe teacher is initialized, skip Layer-3 optimization
        # and keep only Gate/monitoring to avoid unstable early over-reaction.
        if self.w_safe is None:
            enable_no_safe_minimal_fallback = bool(self.config.get("enable_no_safe_minimal_fallback", True))
            if enable_no_safe_minimal_fallback and bool(reject_fallback_branch_used):
                # No teacher yet: use branch fallback (noop/gated) but do not run
                # min-max repair terms that depend on a safe teacher.
                steps = 0
                steps_after_safe_cap = 0
                steps_after_lambda_cap = 0
                lambda_bd = 0.0
                lambda_safe_scale_ratio = 0.0
                lambda_safe_scale_floor = 0.0
                lambda_cap_reason = "safe_model_unavailable_minimal_fallback"
                lambda_disabled_reason = "safe_model_unavailable"
                repair_state = "off"
                repair_state_after_safe_check = "off"
                repair_disabled_reason = "safe_model_unavailable_minimal_fallback"
            else:
                steps = 0
                lambda_bd = 0.0
                steps_after_safe_cap = 0
                steps_after_lambda_cap = 0
                lambda_safe_scale_ratio = 0.0
                lambda_safe_scale_floor = 0.0
                lambda_cap_reason = "safe_model_unavailable"
                lambda_disabled_reason = "safe_model_unavailable"
                repair_state = "off"
                repair_state_after_safe_check = "off"
                repair_disabled_reason = "safe_model_unavailable"
        if self.repair_enabled() and self.w_safe is not None and steps > 0:
            student = post_gate_model
            student.eval()
            for p in student.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.SGD(student.parameters(), lr=repair_lr)

            x_probe = self.cts_intent._resolve_probe_set(clean_dataloader=clean_dataloader, metadata={})
            probe_size = min(int(x_probe.shape[0]), max(32, int(self.config.get("probe_size_light", 128))))
            perm = torch.randperm(int(x_probe.shape[0]), generator=self._rng)[:probe_size]
            x_batch = x_probe.index_select(0, perm.to(device=x_probe.device)).detach().to(self.device)
            with torch.no_grad():
                t_clean_logits = self._extract_logits(self.w_safe(x_batch))

            safe_acc = self._quick_probe_accuracy(self.w_safe, clean_dataloader)
            acc_full_repair = float(self.config.get("acc_full_repair", 0.50))
            acc_floor = float(self.config.get("acc_full_repair_floor", 0.08))
            if (not bootstrap_repair_enabled) and safe_acc < acc_full_repair:
                # Smoothly reduce anti-backdoor strength instead of hard-disabling it.
                denom = max(acc_full_repair - acc_floor, 1e-6)
                ratio = float(np.clip((safe_acc - acc_floor) / denom, 0.0, 1.0))
                ratio_floor_base = float(np.clip(self.config.get("lambda_scale_min_when_safe_low", 0.15), 0.0, 1.0))
                ratio_floor_abnormal = float(np.clip(
                    self.config.get("lambda_scale_min_when_safe_low_abnormal", max(ratio_floor_base, 0.35)),
                    0.0,
                    1.0,
                ))
                ratio_floor = ratio_floor_abnormal if state == "abnormal" else ratio_floor_base
                # Never hard-zero lambda here: keep a conservative anti-backdoor term
                # to avoid KD-only drift when safe teacher is weak.
                scale_ratio = max(ratio, ratio_floor)
                lambda_bd = float(lambda_bd) * scale_ratio
                lambda_safe_scale_ratio = float(scale_ratio)
                lambda_safe_scale_floor = float(ratio_floor)
                lambda_cap_reason = "safe_low_scale"
                # Conservative fallback while safe teacher quality is still low.
                if state == "abnormal":
                    steps = min(int(steps), max(int(self.config.get("steps_cap_when_safe_low_abnormal", 5)), 1))
                    repair_lr *= float(max(self.config.get("repair_lr_scale_when_safe_low_abnormal", 0.80), 0.0))
                else:
                    steps = min(int(steps), max(int(self.config.get("steps_cap_when_safe_low", 3)), 1))
                    repair_lr *= float(max(self.config.get("repair_lr_scale_when_safe_low", 0.60), 0.0))
                steps_after_safe_cap = int(steps)

            if (not bootstrap_repair_enabled) and lambda_bd <= 1e-12:
                # If anti-backdoor term is effectively unavailable, avoid aggressive KD-only updates.
                steps = min(int(steps), max(int(self.config.get("steps_cap_when_lambda_zero", 1)), 1))
                repair_lr *= float(max(self.config.get("repair_lr_scale_when_lambda_zero", 0.35), 0.0))
                lambda_cap_reason = "lambda_near_zero_cap"
            if bootstrap_repair_enabled:
                # Weak-stage bootstrap should stay conservative to avoid driving the
                # model toward a still-immature teacher.
                bootstrap_steps_cap = max(int(self.config.get("bootstrap_steps_cap", 2)), 1)
                if state == "abnormal":
                    bootstrap_steps_cap = max(
                        bootstrap_steps_cap,
                        int(max(self.config.get("bootstrap_steps_cap_abnormal", 3), 1)),
                    )
                steps = min(int(steps), int(bootstrap_steps_cap))
                repair_lr *= float(np.clip(self.config.get("bootstrap_repair_lr_scale", 0.7), 0.0, 1.0))
            steps_after_lambda_cap = int(steps)
            repair_lr = float(np.clip(
                repair_lr,
                float(self.config.get("repair_lr_min", 5e-5)),
                float(self.config.get("repair_lr_max", 2e-3)),
            ))

            augmented = self.augmented_triggers_enabled()
            if self.ablation == "repair_fixed_triggers":
                augmented = False
            trigger_batches = self.build_augmented_triggers(
                x_batch,
                num_triggers=max(int(self.config.get("repair_num_triggers", 4)), 1),
                augmented=bool(augmented),
            )
            temperature = max(float(self.config.get("temperature", 1.0)), 1e-6)
            mu_reg = float(max(self.config.get("mu_reg", 1e-4), 0.0))
            grad_clip = float(max(self.config.get("grad_clip", 1.0), 1e-6))
            safe_param_vec = torch.cat([p.detach().reshape(-1).to(self.device) for p in self.w_safe.parameters()], dim=0)

            if bootstrap_repair_enabled:
                bootstrap_allow_anti_bd = bool(self.config.get("bootstrap_allow_anti_bd", True))
                bootstrap_min_probe_for_anti_bd = float(
                    max(self.config.get("bootstrap_min_probe_for_anti_bd", 0.10), 0.0)
                )
                bootstrap_lambda_ratio = float(
                    np.clip(self.config.get("bootstrap_lambda_ratio", 0.30), 0.0, 1.0)
                )
                bootstrap_lambda_max = float(max(self.config.get("bootstrap_lambda_max", 0.40), 0.0))
                if bootstrap_allow_anti_bd and float(safe_probe_acc) >= bootstrap_min_probe_for_anti_bd:
                    lambda_bd = min(
                        float(lambda_base_before_safe_scale) * bootstrap_lambda_ratio,
                        bootstrap_lambda_max,
                    )
                    lambda_branch = "bootstrap_anti_bd_lite"
                    lambda_cap_reason = "bootstrap_anti_bd_lite"
                else:
                    lambda_bd = 0.0
                    lambda_branch = "bootstrap"
                    lambda_cap_reason = "bootstrap_kd_only"
                repair_state = "bootstrap"
                repair_state_after_safe_check = "bootstrap"
            bootstrap_kd_only = bool(bootstrap_repair_enabled and float(lambda_bd) <= 1e-12)
            for _ in range(max(int(steps), 1)):
                optimizer.zero_grad(set_to_none=True)
                s_clean_logits = self._extract_logits(student(x_batch))
                log_q = F.log_softmax(s_clean_logits / temperature, dim=-1)
                p_ref = F.softmax(t_clean_logits / temperature, dim=-1)
                loss_kd = F.kl_div(log_q, p_ref, reduction="batchmean")
                if bootstrap_kd_only:
                    anti_bd = torch.zeros((), device=self.device, dtype=loss_kd.dtype)
                else:
                    trigger_losses = []
                    for trig_batch in trigger_batches:
                        s_trig_logits = self._extract_logits(student(trig_batch))
                        log_q_trig = F.log_softmax(s_trig_logits / temperature, dim=-1)
                        trigger_losses.append(F.kl_div(log_q_trig, p_ref, reduction="batchmean"))
                    anti_bd = torch.stack(trigger_losses).max() if trigger_losses else torch.zeros((), device=self.device, dtype=loss_kd.dtype)

                student_param_vec = torch.cat([p.reshape(-1) for p in student.parameters()], dim=0)
                reg_term = torch.mean((student_param_vec - safe_param_vec) ** 2)
                loss = loss_kd + float(lambda_bd) * anti_bd + mu_reg * reg_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
                optimizer.step()
                l_kd = float(loss_kd.detach().item())
                l_anti_bd = float(anti_bd.detach().item())
                l_reg = float(reg_term.detach().item())

            repaired_update = self._extract_update_from_models(
                model_before=global_model,
                model_after=student,
                reference_update=aggregated_update,
                server_lr=float(server_lr),
            )

        metrics.update({
            "cts_global_light": float(cts_global_light),
            "cts_global_light_centered": float(cts_global_light_centered),
            "cts_global_light_z": float(cts_global_light_z),
            "cts_checked_this_round": bool(cts_checked_this_round),
            "cts_signal_fresh": bool(cts_signal_fresh),
            "alert_mode": bool(self.recent_alert_mode),
            "alert_mode_reason": str(alert_mode_reason),
            "recent_alert_mode": bool(self.recent_alert_mode),
            "reject_reason": str(reject_reason),
            "reject_counter": int(self.reject_counter),
            "reject_allowed": bool(reject_allowed),
            "reject_blocked_by_no_safe_model": bool(reject_blocked_by_no_safe_model),
            "reject_fallback_branch_used": bool(reject_fallback_branch_used),
            "reject_block_reason": str(reject_block_reason),
            "final_update_branch_hint": str(final_update_branch_hint),
            "repair_state": str(repair_state),
            "repair_state_before_safe_check": str(repair_state_before_safe_check),
            "repair_state_after_safe_check": str(repair_state_after_safe_check),
            "repair_steps_planned": int(steps_planned),
            "repair_steps_after_safe_cap": int(steps_after_safe_cap),
            "repair_steps_after_lambda_cap": int(steps_after_lambda_cap),
            "repair_steps_final": int(steps),
            "repair_steps": int(steps),
            "repair_disabled_reason": str(repair_disabled_reason),
            "lambda_branch": str(lambda_branch),
            "lambda_base_before_safe_scale": float(lambda_base_before_safe_scale),
            "lambda_safe_scale_ratio": float(lambda_safe_scale_ratio),
            "lambda_safe_scale_floor": float(lambda_safe_scale_floor),
            "lambda_cap_reason": str(lambda_cap_reason),
            "lambda_disabled_reason": str(lambda_disabled_reason),
            "lambda_bd": float(lambda_bd),
            "repair_lr": float(repair_lr),
            "l_kd": float(l_kd),
            "l_anti_bd": float(l_anti_bd),
            "l_reg": float(l_reg),
            "safe_initialized": bool(self.w_safe is not None),
            "safe_initialized_this_round": bool(safe_initialized),
            "safe_updated": bool(safe_updated),
            "safe_model_exists": bool(self.w_safe is not None),
            "safe_model_ready": bool(str(self.safe_model_stage) == "full"),
            "safe_model_stage": str(self.safe_model_stage),
            "safe_bootstrap_mode": bool(str(self.safe_model_stage) == "weak"),
            "bootstrap_repair_enabled": bool(bootstrap_repair_enabled),
            "safe_probe_acc": float(safe_probe_acc),
            "safe_init_probe_acc": float(safe_probe_acc),
            "safe_init_reason": str(safe_diag.get("safe_init_reason", "unknown")),
            "safe_init_used_fallback": bool(safe_diag.get("safe_init_used_fallback", False)),
            "safe_init_source_model": str(safe_diag.get("safe_init_source_model", "none")),
            "safe_init_threshold_used": str(safe_diag.get("safe_init_threshold_used", "none")),
            "safe_init_blocking_condition": str(safe_diag.get("safe_init_blocking_condition", "none")),
            "safe_init_context_ok": bool(safe_diag.get("safe_init_context_ok", False)),
            "safe_upgrade_allowed": bool(safe_diag.get("safe_upgrade_allowed", False)),
            "safe_upgrade_reason": str(safe_diag.get("safe_upgrade_reason", "none")),
            "safe_update_reason": str(safe_diag.get("safe_update_reason", "unknown")),
            "safe_model_update_reason": str(safe_diag.get("safe_model_update_reason", safe_diag.get("safe_update_reason", "unknown"))),
            "safe_init_allowed": bool(safe_diag.get("safe_init_allowed", False)),
            "safe_update_allowed": bool(safe_diag.get("safe_update_allowed", False)),
            "safe_update_context_ok": bool(safe_diag.get("safe_update_context_ok", False)),
            "safe_signal_source": str(safe_diag.get("safe_signal_source", self.config.get("safe_signal_source", "dcbd"))),
            "safe_signal_source_effective": str(safe_diag.get("safe_signal_source_effective", "dcbd")),
            "safe_update_ratio_source": str(safe_diag.get("safe_update_ratio_source", self.config.get("safe_update_ratio_source", "confirmed"))),
            "safe_update_ratio_source_effective": str(safe_diag.get("safe_update_ratio_source_effective", "raw")),
            "safe_effective_abnormal_ratio": float(safe_diag.get("safe_effective_abnormal_ratio", abnormal_group_ratio)),
            "safe_raw_abnormal_ratio": float(safe_diag.get("safe_raw_abnormal_ratio", abnormal_group_ratio)),
            "safe_confirmed_abnormal_ratio": float(safe_diag.get("safe_confirmed_abnormal_ratio", confirmed_abnormal_ratio)),
            "safe_refresh_applied": bool(safe_diag.get("safe_refresh_applied", False)),
            "safe_refresh_reason": str(safe_diag.get("safe_refresh_reason", "disabled")),
            "safe_refresh_stale_rounds": int(safe_diag.get("safe_refresh_stale_rounds", self.config.get("safe_refresh_stale_rounds", 20))),
            "safe_refresh_margin": float(safe_diag.get("safe_refresh_margin", self.config.get("safe_refresh_margin", 0.10))),
            "safe_refresh_ema": float(safe_diag.get("safe_refresh_ema", self.config.get("safe_refresh_ema", 0.5))),
            "safe_refresh_last_update_round": int(safe_diag.get("safe_refresh_last_update_round", self.last_safe_update_round)),
            "safe_refresh_rounds_since_last_update": int(safe_diag.get("safe_refresh_rounds_since_last_update", 0)),
            "safe_refresh_confirmed_abnormal_ratio": float(safe_diag.get("safe_refresh_confirmed_abnormal_ratio", confirmed_abnormal_ratio)),
            "safe_refresh_confirmed_abnormal_ratio_max": float(safe_diag.get("safe_refresh_confirmed_abnormal_ratio_max", self.config.get("safe_refresh_confirmed_abnormal_ratio_max", 0.20))),
            "safe_refresh_confirmed_count": int(safe_diag.get("safe_refresh_confirmed_count", int(max(confirmed_suspect_count, 0)))),
            "safe_refresh_max_confirmed": int(safe_diag.get("safe_refresh_max_confirmed", self.config.get("safe_refresh_max_confirmed", 3))),
            "safe_refresh_cts_max": float(safe_diag.get("safe_refresh_cts_max", self.config.get("safe_refresh_cts_max", 2.0))),
            "confirmed_gate_enabled": bool(gate_stats.get("confirmed_gate_enabled", self.config.get("confirmed_gate_enable", False))),
            "confirmed_gate_mean": float(gate_stats.get("confirmed_gate_mean", 1.0)),
            "confirmed_gate_min": float(gate_stats.get("confirmed_gate_min", 1.0)),
            "confirmed_gate_reject_count": int(gate_stats.get("confirmed_gate_reject_count", 0)),
            "confirmed_gate_threshold": float(gate_stats.get("confirmed_gate_threshold", self.config.get("confirmed_gate_threshold", 0.20))),
            "confirmed_gate_beta": float(gate_stats.get("confirmed_gate_beta", self.config.get("confirmed_gate_beta", 8.0))),
            "confirmed_gate_reason": str(gate_stats.get("confirmed_gate_reason", "disabled")),
        })

        assert_abnormal_steps_planned = bool(
            (repair_state != "abnormal")
            or (int(steps_planned) == int(self.config.get("steps_abnormal", steps_planned)))
        )
        assert_lambda_cap_reason_present = bool(
            abs(float(lambda_bd) - float(lambda_base_before_safe_scale)) <= 1e-10
            or str(lambda_cap_reason) != "none"
        )
        assert_safe_init_reason_present = bool((not safe_initialized) or str(safe_diag.get("safe_init_reason", "")) not in {"", "none"})
        assert_safe_update_reason_present = bool((not safe_updated) or str(safe_diag.get("safe_update_reason", "")) not in {"", "none"})
        metrics.update({
            "assert_abnormal_steps_planned": bool(assert_abnormal_steps_planned),
            "assert_lambda_cap_reason_present": bool(assert_lambda_cap_reason_present),
            "assert_safe_init_reason_present": bool(assert_safe_init_reason_present),
            "assert_safe_update_reason_present": bool(assert_safe_update_reason_present),
        })
        if bool(metrics.get("global_gate_rejected", False)):
            self.consecutive_reject_count = int(self.consecutive_reject_count) + 1
        else:
            self.consecutive_reject_count = 0
        metrics["consecutive_reject_count"] = int(self.consecutive_reject_count)
        self.safe_initialized = bool(self.w_safe is not None)
        self.safe_updated = bool(safe_updated)
        self.last_repair_metrics = metrics
        return repaired_update, metrics

    @staticmethod
    def _extract_logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            if isinstance(outputs[0], torch.Tensor):
                return outputs[0]
        raise TypeError("Model forward must return logits tensor or tuple/list with logits first.")
