"""Top-level controller for the C3S-Guard federated defense."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from defense.c3s_guard.cts_intent import CTSIntent
from defense.c3s_guard.dgc_clean import DGCClean, build_backdoor_subspace_from_group_diff
from defense.c3s_guard.dgc_repair import DGCRepair
from defense.c3s_guard.s3_loc import S3Loc, localize_s3_clients
from defense.c3s_guard.trigger_proxy import TriggerProxyBank
from defense.c3s_guard.utils import C3SGuardDecision, CTSIntentResult, DGCCleanResult, ModelUpdate


class C3SGuard:
    """Coordinate CTS-Intent, S3-Loc, and DGC-Clean across FL rounds."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Module,
        clean_dataloader: Optional[DataLoader] = None,
        trigger_proxy_bank: Optional[TriggerProxyBank] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the C3S-Guard controller.

        Args:
            config: Dictionary containing all module hyper-parameters.
            model: BackdoorBench-compatible classification model.
            clean_dataloader: Optional clean data loader for probing and references.
            trigger_proxy_bank: Optional proxy bank shared with CTS-Intent.
            device: Device used by the defense stack.
        """

        merged_config = self.default_config()
        merged_config.update(config or {})
        self.config = merged_config

        self.device = self._resolve_device(device, model)
        self.model = model.to(self.device)
        self.clean_dataloader = clean_dataloader
        self.trigger_proxy_bank = trigger_proxy_bank

        self.num_clients = int(self.config["num_clients"])
        self.base_audit_period = int(self.config["audit_period"])
        self.audit_period = self.base_audit_period
        self.audit_rounds = int(self.config["audit_rounds"])
        self.audit_group_size = int(self.config["audit_group_size"])
        self.k_min = int(self.config["k_min"])
        self.backup_ratio = float(self.config["backup_ratio"])
        self.suspicious_weight = float(self.config["suspicious_weight"])
        self.degenerate_p_hat_floor = float(self.config["degenerate_p_hat_floor"])
        self.audit_from_selected_only = bool(self.config["audit_from_selected_only"])
        self.cleaning_enabled = bool(self.config.get("enable_cleaning", True))

        self.cts_intent = CTSIntent(
            config=self.config.get("cts_intent", {}),
            model=self.model,
            trigger_proxy_bank=self.trigger_proxy_bank,
            device=self.device,
        )
        self.s3_loc = S3Loc(
            config=self.config.get("s3_loc", {}),
            num_clients=self.num_clients,
            group_size=self.audit_group_size,
            audit_rounds=self.audit_rounds,
            device=self.device,
        )
        dgc_model_ref = self.model if self.cleaning_enabled else deepcopy(self.model).cpu()
        dgc_device = self.device if self.cleaning_enabled else torch.device("cpu")
        self.dgc_clean = DGCClean(
            config=self.config.get("dgc_clean", {}),
            model=dgc_model_ref,
            device=dgc_device,
        )
        dgc_repair_config = dict(self.config.get("dgc_repair", {}))
        if "confirmed_gate_enable" not in dgc_repair_config:
            dgc_repair_config["confirmed_gate_enable"] = bool(
                self.config.get("c3s_confirmed_gate_enable", False)
            )
        if "confirmed_gate_beta" not in dgc_repair_config:
            dgc_repair_config["confirmed_gate_beta"] = float(
                self.config.get("c3s_confirmed_gate_beta", 8.0)
            )
        if "confirmed_gate_threshold" not in dgc_repair_config:
            dgc_repair_config["confirmed_gate_threshold"] = float(
                self.config.get("c3s_confirmed_gate_threshold", 0.20)
            )
        if "risk_aware_sampling" not in dgc_repair_config:
            dgc_repair_config["risk_aware_sampling"] = bool(
                self.config.get("risk_aware_enable", False)
            )
        if "risk_aware_source" not in dgc_repair_config:
            dgc_repair_config["risk_aware_source"] = str(
                self.config.get("risk_aware_source", "confirmed")
            )
        if "risk_gamma" not in dgc_repair_config:
            dgc_repair_config["risk_gamma"] = float(
                max(self.config.get("risk_aware_gamma", 1.0), 0.0)
            )
        if "risk_p_min" not in dgc_repair_config:
            dgc_repair_config["risk_p_min"] = float(
                self.config.get("risk_aware_p_min", 0.5)
            )
        if "safe_signal_source" not in dgc_repair_config:
            dgc_repair_config["safe_signal_source"] = str(
                self.config.get("dgc_repair_safe_signal_source", "dcbd")
            )
        self.dgc_repair = DGCRepair(
            config=dgc_repair_config,
            model=self.model,
            cts_intent=self.cts_intent,
            device=self.device,
        )

        self.current_round: int = -1
        self.audit_history: List[Dict[str, Any]] = []
        self.A_matrix: np.ndarray = np.zeros((0, self.num_clients), dtype=np.float32)
        self.b_g_history: List[bool] = []
        self.suspicious_clients: Set[int] = set()
        self.ema_model: Module = deepcopy(self.dgc_clean.ema_model)
        if self.cleaning_enabled:
            self.ema_model = self.ema_model.to(self.device)
        self.y_star_estimate: int = -1

        self.selected_clients: List[int] = []
        self.current_audit_plan: List[Dict[str, Any]] = []
        self.current_group_updates: "OrderedDict[int, ModelUpdate]" = OrderedDict()
        self.current_group_clients: Dict[int, List[int]] = {}
        self.current_dropout_counts: List[int] = []
        self.current_cts_scores: List[float] = []
        self.current_localization_signal_scores: List[float] = []
        self.current_b_g: List[bool] = []
        self.current_p_hat: float = 0.0
        self.current_degenerate: bool = False
        self.current_suspicious_set: List[int] = []
        self.current_confirmed_suspects: List[int] = []
        self.current_fct_stats: Dict[int, Dict[str, Any]] = {}
        self.current_fct_fallback_reason: str = "disabled"
        self.current_cleaning_p_hat: float = 0.0
        self.current_cleaning_confidence: float = 0.0
        self.current_localization_reliable: bool = False
        self.current_dense_mode_active: bool = False
        self.current_dense_p_hat: float = 0.0
        self.current_cleaning_accept_score: float = 0.0
        self.current_cleaning_tier: str = "none"
        self.current_cleaning_reject_reason: str = "none"
        self.current_effective_rho: float = 0.0
        self.current_reweight_strength: float = 1.0
        self.current_audit_active: bool = False
        self.current_decision: Optional[C3SGuardDecision] = None
        self.current_gate_update: Optional[ModelUpdate] = None
        self.current_repair_metrics: Dict[str, Any] = dict(self.dgc_repair.last_repair_metrics)
        self.current_consensus_candidates: List[int] = []
        self.current_B_tilde: Optional[torch.Tensor] = None
        self.B_tilde_updated_at: int = -1
        self.B_tilde_diagnostics: Dict[str, Any] = {}
        self.btilde_quality_history: List[Dict[str, Any]] = []
        self.w_init_flat: torch.Tensor = self.dgc_clean._flatten_model_parameters(self.model).detach().to(self.device).clone()
        self.weight_purify_history: List[Dict[str, Any]] = []
        self.weight_purify_stats: Dict[str, float] = {
            "total_purify_calls": 0.0,
            "total_backdoor_energy_fraction": 0.0,
            "mean_backdoor_energy_fraction": 0.0,
            "last_round": -1.0,
            "last_strength": 0.0,
            "last_b_tilde_rank": 0.0,
        }
        self.continuous_proj_history: Dict[str, Dict[str, float]] = {}
        self.continuous_proj_stats: Dict[str, float] = {
            "total_rounds_projected": 0.0,
            "total_norm_removed": 0.0,
            "total_reduction_ratio": 0.0,
            "rounds_without_B_tilde": 0.0,
            "B_tilde_update_count": 0.0,
            "first_B_tilde_round": -1.0,
        }
        self._probe_set: Optional[torch.Tensor] = None
        self._backup_members: Dict[int, List[int]] = {}
        self._historical_score_sum: Dict[int, float] = {}
        self._historical_score_weight: Dict[int, float] = {}
        self._historical_top_count: Dict[int, float] = {}
        self._historical_anomalous_audits: int = 0
        self._historical_reliable_audits: int = 0
        self._risk_candidate_precision_history: List[float] = []
        self.current_risk_sampling_enabled: bool = False
        self.current_risk_activation_reason: str = "disabled"
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.config["seed"]))
        # Keep FCT sampling deterministic but independent from audit-group sampling.
        self._fct_rng = torch.Generator(device="cpu")
        self._fct_rng.manual_seed(int(self.config["seed"]) + 100003)
        # Track per-client FCT confirmation outcomes across audits for stability gating.
        self._fct_confirm_history: Dict[int, List[int]] = {}

    @staticmethod
    def default_config() -> Dict[str, Any]:
        """Return a minimal nested config template for C3S-Guard."""

        return {
            "seed": 1234,
            "num_clients": 100,
            "audit_period": 20,
            "audit_rounds": 20,
            "audit_group_size": 10,
            "audit_group_pair_penalty": 1.0,
            "audit_group_size_penalty": 0.05,
            "k_min": 5,
            "backup_ratio": 0.3,
            "suspicious_topk": 1,
            "suspicious_weight": 0.5,
            "degenerate_p_hat_floor": 0.30,
            "enable_client_reweighting": True,
            "localization_reliable_min_anomalous_groups": 2,
            "localization_reliable_max_positive_ratio": 0.12,
            "localization_reliable_min_cts_z": 1.4,
            "localization_reliable_min_consistency": 0.50,
            "localization_reliable_min_rank_score": 0.30,
            "localization_reliable_min_selected_consistency": 0.55,
            
            "reweight_min_score_gap": 0.20,
            "reweight_min_score_ratio": 1.10,
            "reweight_min_top_score": 2.0,
            "reweight_min_anomalous_groups": 2,
            "reweight_max_positive_ratio": 0.12,
            "ranking_min_score_gap": 0.50,
            "ranking_min_score_ratio": 1.18,
            "ranking_min_top_score": 2.30,
            "ranking_min_anomalous_groups": 1,
            "ranking_max_positive_ratio": 0.16,
            "ranking_fallback_min_cts_z": 1.70,
            "ranking_fallback_min_consistency": 0.70,
            "reliable_pool_min_score_gap": 0.35,
            "reliable_pool_min_score_ratio": 1.12,
            "reliable_pool_min_top_score": 2.30,
            "reliable_pool_min_anomalous_groups": 1,
            "reliable_pool_min_consensus_overlap": 2,
            "reliable_pool_candidate_topk": 5,
            "reliable_pool_consensus_topk": 3,
            "reliable_pool_allow_fallback": False,
            "reliable_pool_fallback_min_cts_z": 1.70,
            "reliable_pool_fallback_min_consistency": 0.70,
            "reliable_pool_strong_min_anomalous_groups": 2,
            "reliable_pool_strong_max_positive_ratio": 0.10,
            "reliable_pool_strong_min_top_score": 4.0,
            "reliable_pool_strong_min_score_gap": 0.90,
            "reliable_pool_strong_min_score_ratio": 1.24,
            "reliable_pool_strong_min_historical_audits": 3,
            "reliable_pool_strong_min_historical_mean": 0.80,
            "reliable_pool_weak_min_historical_audits": 2,
            "reliable_pool_weak_min_historical_mean": 0.75,
            "reliable_pool_weak_min_top_score": 2.8,
            "reliable_pool_weak_min_score_gap": 0.18,
            "reliable_pool_weak_min_score_ratio": 1.05,
            "reliable_pool_weak_min_consensus_overlap": 1,
            "reliable_pool_ranking_min_historical_audits": 3,
            "reliable_pool_ranking_min_top_score": 3.0,
            "reliable_pool_ranking_min_score_gap": 0.75,
            "reliable_pool_ranking_min_score_ratio": 1.25,
            "reliable_pool_ranking_min_historical_mean": 0.55,
            "enable_consensus_reweighting": False,
            "reweight_topk": 3,
            "enable_provisional_reweighting": False,
            "provisional_reweight_topk": 2,
            "provisional_reweight_min_anomalous_groups": 2,
            "provisional_reweight_max_positive_ratio": 0.10,
            "provisional_reweight_min_top_score": 2.0,
            "provisional_reweight_min_score_gap": 0.25,
            "provisional_reweight_min_score_ratio": 1.10,
            "provisional_reweight_require_nonfallback": True,
            "enable_ranking_reweighting": False,
            "ranking_reweight_topk": 2,
            "ranking_reweight_min_historical_audits": 3,
            "ranking_reweight_min_anomalous_groups": 1,
            "ranking_reweight_max_positive_ratio": 0.12,
            "ranking_reweight_min_top_score": 2.0,
            "ranking_reweight_min_top_persistence": 0.10,
            "ranking_reweight_min_historical_mean": 0.20,
            "ranking_reweight_min_consensus_overlap": 1,
            "ranking_reweight_enable_strong_override": True,
            "ranking_reweight_override_min_historical_audits": 4,
            "ranking_reweight_override_min_top_score": 3.5,
            "ranking_reweight_override_min_score_gap": 0.50,
            "ranking_reweight_override_min_score_ratio": 1.20,
            "ranking_reweight_enable_local_override": True,
            "ranking_reweight_local_override_min_historical_audits": 2,
            "ranking_reweight_local_override_min_top_score": 1.90,
            "ranking_reweight_local_override_min_score_gap": 0.18,
            "ranking_reweight_local_override_min_score_ratio": 1.08,
            "ranking_reweight_local_override_require_nonfallback": False,
            "ranking_reweight_require_nonfallback": False,
            "enable_fallback_reweighting": False,
            "fallback_reweight_topk": 1,
            "fallback_reweight_min_anomalous_groups": 1,
            "fallback_reweight_max_positive_ratio": 0.12,
            "fallback_reweight_min_top_score": 2.0,
            "fallback_reweight_min_score_gap": 0.20,
            "fallback_reweight_min_score_ratio": 1.10,
            "fallback_reweight_min_historical_audits": 2,
            "fallback_reweight_min_top_frequency": 1,
            "fallback_reweight_require_history_support": True,
            "fallback_reweight_require_consensus_overlap": True,
            "fallback_reweight_require_nonfallback": False,
            "consensus_min_anomalous_audits": 2,
            "consensus_min_top_frequency": 2,
            "consensus_score_boost": 0.5,
            "consensus_min_current_ratio": 0.50,
            "consensus_min_current_ratio_hard": 0.20,
            "consensus_min_historical_mean": 0.12,
            "consensus_persistence_cap": 0.85,
            "consensus_history_topk": 10,
            "consensus_candidate_topk": 10,
            "reweight_min_reliable_audits": 3,
            "reweight_min_consensus_overlap": 1,
            "reweight_selected_only": True,
            "reweight_allow_no_consensus_with_strong_signal": False,
            "reweight_no_consensus_min_top_score": 2.4,
            "reweight_no_consensus_min_score_gap": 0.30,
            "reweight_no_consensus_min_score_ratio": 1.10,
            "reweight_no_consensus_min_historical_audits": 2,
            "consensus_reweight_require_signal": True,
            "consensus_reweight_require_current_overlap": True,
            "consensus_history_min_anomalous_groups": 1,
            "consensus_history_max_positive_ratio": 0.10,
            "consensus_history_min_top_score": 3.0,
            "consensus_history_min_score_gap": 0.75,
            "consensus_history_min_score_ratio": 1.20,
            "consensus_history_allow_fallback": False,
            "consensus_history_relaxed_enabled": True,
            "consensus_history_relaxed_min_anomalous_groups": 1,
            "consensus_history_relaxed_max_positive_ratio": 0.20,
            "consensus_history_relaxed_min_top_score": 1.20,
            "consensus_history_relaxed_require_reliable": True,
            "consensus_history_relaxed_allow_fallback": True,
            "consensus_history_quality_gate_enabled": True,
            "consensus_history_quality_min_candidate_count": 2,
            "consensus_history_quality_min_selected_consistency": 0.64,
            "consensus_history_quality_min_rank_score": 0.50,
            "consensus_history_quality_min_consensus_count": 0,
            
            "consensus_history_fallback_min_cts_z": 1.70,
            "consensus_history_fallback_min_consistency": 0.70,
            "consensus_history_bootstrap_max_positive_ratio": 0.12,
            "consensus_history_bootstrap_min_top_score": 2.80,
            "consensus_history_bootstrap_min_score_gap": 0.18,
            "consensus_history_bootstrap_min_score_ratio": 1.05,
            "consensus_history_score_decay": 0.97,
            "consensus_history_top_decay": 0.95,
            "consensus_history_decay_warmup": 2,
            "consensus_relax_after_audits": 0,
            "retain_reweight_on_unreliable": True,
            "retain_reweight_min_anomalous_groups": 1,
            "retain_reweight_min_consensus_overlap": 1,
            "retain_reweight_intersection_only": True,
            "stability_blend_weight": 0.35,
            "stability_persistence_boost": 0.20,
            "stability_min_top_frequency": 2,
            "stability_min_persistence": 0.15,
            "stability_reliable_min_score": 2.2,
            "stability_enable_dynamic_gate": True,
            "stability_gate_weak_scale": 0.10,
            "stability_gate_disable_when_no_anomaly": True,
            "stability_gate_min_anomalous_groups": 1,
            "stability_gate_min_positive_ratio": 0.02,
            "stability_gate_min_current_max": 0.20,
            "stability_gate_min_score_gap": 0.05,
            "stability_gate_min_score_ratio": 1.06,
            "stability_gate_min_cts_z": 2.20,
            "stability_gate_min_consistency": 0.55,
            "reliable_consensus_weight": 0.35,
            "reliable_history_weight": 0.15,
            "reliable_persistence_weight": 0.10,
            "reliable_min_support": 0.25,
            "reliable_low_support_penalty": 0.75,
            "reliable_min_current_ratio": 0.55,
            "reliable_max_boost": 0.40,
            "target_label": None,
            "force_known_target": False,
            "cleaning_min_p_hat": 0.06,
            "cleaning_min_p_hat_when_anomalous": 0.10,
            "cleaning_max_p_hat_when_unreliable": 0.08,
            "cleaning_max_p_hat_with_weak_signal": 0.05,
            "cleaning_consistency_threshold": 0.92,
            "cleaning_cts_z_threshold": 3.0,
            "cleaning_cts_score_threshold": 0.90,
            "cleaning_consistency_threshold_floor": 0.40,
            "cleaning_cts_z_threshold_floor": 0.80,
            "cleaning_cts_score_threshold_floor": 0.01,
            "cleaning_consistency_threshold_quantile": 0.65,
            "cleaning_cts_z_threshold_quantile": 0.85,
            "cleaning_cts_score_threshold_quantile": 0.85,
            "cleaning_reliability_consistency_source": "candidate_selected",
            "cleaning_reliability_relaxed_consistency_floor": 0.32,
            "cleaning_reliability_require_two_signals": True,
            "cleaning_reliability_disallow_when_reasons": ["consistency_below_threshold", "cts_z_below_threshold"],
            "cleaning_cts_z_clip": 12.0,
            "cleaning_mad_sigma_floor": 0.01,
            "cleaning_cts_score_norm_threshold": 1.2,
            "repair_p_hat_cap_when_unreliable": 0.12,
            "repair_abnormal_ratio_cap_when_unreliable": 0.10,
            "hist_keep_min_consistency": 0.45,
            "hist_keep_min_oriented_diff": 0.003,
            "hist_keep_allow_abs_fallback": True,
            "hist_keep_min_abs_diff": 0.002,
            "hist_keep_min_abs_signal": 0.20,
            "hist_keep_enable_soft_support": True,
            "hist_keep_soft_min_mad_z": 1.20,
            "hist_keep_soft_min_consistency": 0.42,
            "dcbd_absolute_signal_priority": True,
            "dcbd_hist_keep_min_tsc": 0.0,
            "dcbd_hist_keep_min_consistency": 0.34,
            "dcbd_hist_keep_min_dcbd_score": 0.0,
            "dcbd_hist_keep_min_dcbd_z": 1.0,
            "dcbd_tsc_hard_gate_enable": False,
            "dcbd_tsc_hard_gate_min": 0.0,
            "dcbd_tsc_hard_gate_use_abs": False,
            "dcbd_tsc_hard_gate_apply_anomalous": True,
            "dcbd_tsc_hard_gate_apply_hist_keep": True,
            "dcbd_tsc_hard_gate_apply_candidate": True,
            "dcbd_reliability_use_absolute_score": True,
            "candidate_require_hist_keep": True,
            "candidate_min_consistency": 0.45,
            "candidate_min_oriented_diff": 0.003,
            "candidate_min_oriented_signal": 0.05,
            "candidate_use_two_sided_signal": True,
            "candidate_allow_abs_fallback": True,
            "candidate_min_abs_diff": 0.002,
            "candidate_min_abs_signal": 0.20,
            "candidate_min_consistency_abs": 0.40,
            "dcbd_candidate_min_tsc": 0.0,
            "dcbd_candidate_min_consistency": 0.34,
            "dcbd_candidate_min_dcbd_score": 0.0,
            "dcbd_candidate_min_dcbd_z": 1.0,
            "dcbd_candidate_rank_tsc_weight": 1.0,
            "dcbd_candidate_rank_dcbd_z_weight": 0.25,
            "candidate_allow_current_when_history_blocked": True,
            "candidate_history_block_min_cts_z": 0.80,
            "candidate_history_block_min_anomalous_groups": 1,
            "candidate_consistency_topk_for_reliability": 3,
            "cts_signal_bridge_enable": True,
            "cts_signal_bridge_min_candidate_count": 1,
            "cts_signal_bridge_min_consistency": 0.42,
            "cts_signal_bridge_min_score": 0.20,
            "cts_signal_bridge_min_cts_z": 0.80,
            "cts_signal_bridge_min_rank_score": 0.02,
            "reweight_quality_gate_enabled": True,
            "reweight_quality_min_candidate_count": 2,
            "reweight_quality_min_selected_consistency": 0.64,
            "reweight_quality_min_rank_score": 0.50,
            "reweight_quality_min_consensus_count": 0,
            
            "cleaning_skip_unreliable": True,
            "cleaning_unreliable_min_confidence": 0.55,
            "cleaning_p_hat_cap_unreliable": 0.02,
            "cleaning_p_hat_cap_without_reweight": 0.015,
            "cleaning_min_p_hat_to_apply": 0.01,
            "cleaning_skip_without_reweight": True,
            "cleaning_without_reweight_require_localization": True,
            "cleaning_without_reweight_min_confidence": 0.70,
            "cleaning_without_reweight_min_p_hat": 0.018,
            "cleaning_min_clean_acc_before_apply": 0.0,
            "cleaning_min_clean_acc_without_reweight": 0.0,
            "cleaning_policy_version": "v2_continuous_tiered",
            "cleaning_accept_threshold_weak": 0.35,
            "cleaning_accept_threshold_medium": 0.55,
            "cleaning_accept_min_suspicious_count": 1,
            "cleaning_accept_min_consensus_ratio": 0.10,
            "repair_clean_accept_min_candidate_score": 0.20,
            "repair_clean_accept_min_candidate_score_unreliable": 0.60,
            "repair_clean_accept_min_p_hat_unreliable": 0.08,
            "repair_clean_accept_allow_unreliable_when_strong": False,
            "repair_clean_accept_min_consistency_unreliable": 0.45,
            "repair_clean_accept_min_cts_z_unreliable": 0.90,
           
            "repair_clean_accept_require_localization_reliable": True,
            "repair_clean_accept_allow_unreliable_with_highconf_confirmed": True,
            "repair_clean_accept_min_highconf_confirmed": 2,
            "repair_clean_accept_highconf_min_fct_z": 1.8,
            "repair_clean_accept_highconf_min_pairs": 4,
            "repair_clean_accept_disallow_unreliable_if_reasons": [
                "consistency_below_threshold",
                "cts_z_below_threshold",
            ],
            "cleaning_accept_w_loc": 0.40,
            "cleaning_accept_w_cand": 0.30,
            "cleaning_accept_w_cons": 0.20,
            "cleaning_accept_w_phat": 0.10,
            "cleaning_accept_top_score_ref": 3.0,
            "cleaning_accept_score_gap_ref": 0.5,
            "cleaning_accept_score_ratio_ref": 1.2,
            "p_hat_correction_mode": "group_ratio",
            "p_hat_correction_factor": 1.0,
            "p_hat_correction_max": 0.35,
            "p_hat_correction_auto_factor_max": 3.0,
            "cleaning_min_p_hat_for_accept": 0.01,
            "cleaning_force_weak_when_allowed": True,
            "cleaning_force_apply_in_auto": True,
            "cleaning_rho_w_p_hat": 0.60,
            "cleaning_rho_w_loc": 0.25,
            "cleaning_rho_w_cons": 0.15,
            "cleaning_rho_max_predicted": 0.20,
            "cleaning_rho_max_weak": 0.10,
            "cleaning_rho_max_medium": 0.20,
            "cleaning_rho_max_strong": 0.35,
            "cleaning_rho_min_weak": 0.03,
            "cleaning_rho_min_medium": 0.08,
            "cleaning_rho_min_strong": 0.15,
            "cleaning_rho_reference": 0.20,
            "cleaning_weak_distill_steps": 1,
            "cleaning_medium_distill_steps": 2,
            "cleaning_strong_distill_steps": 3,
            "dense_mode_enabled": True,
            "dense_mode_positive_ratio_max": 0.20,
            "dense_mode_min_cts_z": 2.0,
            "dense_mode_min_consistency": 0.40,
            "dense_mode_anchor_scale": 1.25,
            "dense_mode_hard_weight": 0.15,
            "dense_mode_soft_weight": 0.70,
            "dense_mode_consensus_weight": 0.15,
            "dense_mode_min_p_hat": 0.08,
            "dense_mode_max_p_hat": 0.95,
            "dense_mode_cleaning_relax": True,
            "dense_mode_cleaning_min_confidence": 0.25,
            "dense_mode_cleaning_min_p_hat": 0.03,
            "dense_mode_allow_without_reweight": True,
            "audit_from_selected_only": False,
            "cts_group_update_scale": "mean",
            "cts_group_norm_align": True,
            "cts_group_norm_ratio": 0.75,
            "cts_group_norm_min_scale": 0.05,
            "cts_group_norm_allow_upscale": False,
            "cts_reprobe_on_collapse": True,
            "cts_reprobe_max_abs_threshold": 0.02,
            "cts_reprobe_std_threshold": 0.01,
            "cts_reprobe_boost_factor": 4.0,
            "cts_reprobe_max_scale": 1.0,
            "dgc_continuous_projection": True,
            "dgc_continuous_projection_strength": 1.0,
            "dgc_weight_purification": False,
            "dgc_weight_purify_strength": 0.3,
            "dgc_repair_enable": True,
            "btilde_rank": 10,
            "btilde_min_anomalous": 2,
            "btilde_min_energy_ratio": 0.3,
            "s3_history_accumulation_gate": True,
            "s3_history_disable_gate_when_cleaning_disabled": True,
            "s3_history_min_anomalous_groups": 1,
            "s3_history_min_cts_z": 1.0,
            "s3_history_min_signal_std": 0.5,
            "s3_history_allow_abs_cts_fallback": True,
            "s3_history_min_abs_cts_fallback": 0.015,
            "s3_history_min_signal_std_fallback": 0.02,
            "enable_cleaning": True,
            "repair_require_confirmed": False,
            "repair_require_confirmed_alert_override": True,
            "repair_require_confirmed_alert_min_p_hat": 0.05,
            "repair_block_when_safe_weak": False,
            "repair_block_when_low_confidence": False,
            "repair_cleaned_min_accept_score": 0.15,
            "c3s_confirmed_gate_enable": False,
            "c3s_confirmed_gate_beta": 8.0,
            "c3s_confirmed_gate_threshold": 0.20,
            "fct_enable": False,
            "fct_topn": 6,
            "fct_num_pairs": 6,
            "fct_z_threshold": 1.0,
            "fct_use_matched_controls": True,
            "fct_group_size": 0,
            "fct_clean_pool_strategy": "bottom",
            "fct_clean_pool_bottom_ratio": 0.60,
            "fct_diff_winsor_quantile": 0.10,
            "fct_use_mad_z": True,
            "fct_mad_scale_eps": 1e-6,
            "fct_skip_on_high_contamination": True,
            "fct_skip_max_contamination_risk": 0.25,
            "fct_stability_window": 3,
            "fct_stability_min_pass": 2,
            "fct_stability_enable": True,
            "localization_signal_source": "auto",  # auto | oriented | dcbd | group_behavior
            "reweight_require_confirmed_suspects": True,
            "reweight_require_confirmed_min_count": 1,
            "reweight_require_fct_quality_gate": True,
            "reweight_fct_min_confirmation_rate": 0.35,
            "reweight_fct_min_mean_z": 1.20,
            "reweight_fct_min_mean_diff": 0.00,
            "reweight_fct_max_clean_pool_contamination_risk": 0.40,
            "risk_confirmed_highconf_enable": True,
            "risk_confirmed_highconf_min_fct_z": 1.5,
            "risk_confirmed_highconf_min_pairs": 4,
            "risk_confirmed_highconf_require_localization_reliable": True,
            "risk_aware_enable": False,
            "risk_aware_source": "confirmed",
            "risk_aware_gamma": 1.0,
            "risk_aware_p_min": 0.5,
            "risk_aware_require_fct": False,
            "risk_aware_raw_precision_min": 0.30,
            "risk_aware_proxy_cv_max": 0.50,
            "repair_candidate_require_confirmation": True,
            "repair_candidate_min_safe_acc": 0.45,
            "repair_candidate_shadow_only": False,
            "repair_candidate_allow_weak_safe_override": True,
            "repair_candidate_weak_safe_override_min_confirmed": 2,
            "repair_candidate_weak_safe_override_min_safe_acc": 0.25,
            "repair_candidate_weak_safe_override_require_alert": True,
            "dgc_repair_safe_signal_source": "dcbd",
            "dgc_repair": {},
            "cts_intent": {},
            "s3_loc": {
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
            },
            "dgc_clean": {
                "num_gradient_samples": 10,
                "gradient_probe_batch_size": 10,
                "distill_epochs": 3,
                "ema_gamma": 0.999,
                "lambda_min": 0.1,
                "lambda_max": 1.0,
                "lambda_p0": 0.05,
                "lambda_p1": 0.30,
            },
        }

    def reset(self) -> None:
        """Reset transient controller state and audit buffers."""

        self.current_round = -1
        self.audit_history = []
        self.A_matrix = np.zeros((0, self.num_clients), dtype=np.float32)
        self.b_g_history = []
        self.suspicious_clients = set()
        self.selected_clients = []
        self.current_audit_plan = []
        self.current_group_updates = OrderedDict()
        self.current_group_clients = {}
        self.current_dropout_counts = []
        self.current_cts_scores = []
        self.current_localization_signal_scores = []
        self.current_b_g = []
        self.current_p_hat = 0.0
        self.current_degenerate = False
        self.current_suspicious_set = []
        self.current_confirmed_suspects = []
        self.current_fct_stats = {}
        self.current_fct_fallback_reason = "disabled"
        self.current_cleaning_p_hat = 0.0
        self.current_cleaning_confidence = 0.0
        self.current_localization_reliable = False
        self.current_dense_mode_active = False
        self.current_dense_p_hat = 0.0
        self.current_cleaning_accept_score = 0.0
        self.current_cleaning_tier = "none"
        self.current_cleaning_reject_reason = "none"
        self.current_effective_rho = 0.0
        self.current_reweight_strength = 1.0
        self.current_audit_active = False
        self.current_decision = None
        self.current_gate_update = None
        self.current_repair_metrics = dict(self.dgc_repair.last_repair_metrics)
        self.current_consensus_candidates = []
        self.current_B_tilde = None
        self.B_tilde_updated_at = -1
        self.B_tilde_diagnostics = {}
        self.btilde_quality_history = []
        self.weight_purify_history = []
        self.weight_purify_stats = {
            "total_purify_calls": 0.0,
            "total_backdoor_energy_fraction": 0.0,
            "mean_backdoor_energy_fraction": 0.0,
            "last_round": -1.0,
            "last_strength": 0.0,
            "last_b_tilde_rank": 0.0,
        }
        self.continuous_proj_history = {}
        self.continuous_proj_stats = {
            "total_rounds_projected": 0.0,
            "total_norm_removed": 0.0,
            "total_reduction_ratio": 0.0,
            "rounds_without_B_tilde": 0.0,
            "B_tilde_update_count": 0.0,
            "first_B_tilde_round": -1.0,
        }
        self._risk_candidate_precision_history = []
        self.current_risk_sampling_enabled = False
        self.current_risk_activation_reason = "disabled"
        self.y_star_estimate = -1
        self.audit_period = self.base_audit_period
        self._probe_set = None
        self._backup_members = {}
        self._historical_score_sum = {}
        self._historical_score_weight = {}
        self._historical_top_count = {}
        self._historical_anomalous_audits = 0
        self._historical_reliable_audits = 0
        self._fct_confirm_history = {}
        self.s3_loc.reset_cycle()
        self.dgc_clean.reset()
        self.dgc_repair.reset()
        self.ema_model = deepcopy(self.dgc_clean.ema_model)
        if self.cleaning_enabled:
            self.ema_model = self.ema_model.to(self.device)

    def should_audit(self, round_idx: int) -> bool:
        """
        Decide whether the current round should trigger a full audit cycle.

        Args:
            round_idx: One-based or zero-based global round index, as defined by
                the training loop.
        """

        if self.audit_period <= 0:
            return False
        return (int(round_idx) + 1) % int(self.audit_period) == 0

    def on_round_start(self, round_idx: int, global_model: Module) -> None:
        """Begin a new federated round and synchronize controller state."""

        self.current_round = int(round_idx)
        self.model = global_model.to(self.device)
        self.cts_intent.model = self.model
        if self.cleaning_enabled:
            self.dgc_clean.model = self.model
        self.current_audit_active = self.should_audit(round_idx)
        self.current_group_updates = OrderedDict()
        self.current_group_clients = {}
        self.current_dropout_counts = []
        self.current_cts_scores = []
        self.current_localization_signal_scores = []
        self.current_b_g = []
        self.current_p_hat = 0.0
        self.current_degenerate = False
        self.current_suspicious_set = []
        self.current_confirmed_suspects = []
        self.current_fct_stats = {}
        self.current_fct_fallback_reason = "disabled"
        self.current_cleaning_p_hat = 0.0
        self.current_cleaning_confidence = 0.0
        self.current_localization_reliable = False
        self.current_dense_mode_active = False
        self.current_dense_p_hat = 0.0
        self.current_cleaning_accept_score = 0.0
        self.current_cleaning_tier = "none"
        self.current_cleaning_reject_reason = "none"
        self.current_effective_rho = 0.0
        self.current_reweight_strength = 1.0
        self.current_decision = None
        self.current_gate_update = None
        self.current_repair_metrics = dict(self.dgc_repair.last_repair_metrics)
        self.current_consensus_candidates = []
        self.current_audit_plan = []
        self._backup_members = {}
        self.dgc_repair.on_round_start(int(round_idx))

    def on_clients_selected(self, selected_clients: Sequence[int]) -> None:
        """Store the currently selected clients for the round."""

        self.selected_clients = [int(client_id) for client_id in selected_clients]

    def sample_clients_risk_aware(self, clients_per_round: int) -> List[int]:
        """Sample clients with DGC-Repair risk-aware probabilities."""

        k = int(clients_per_round)
        if k >= int(self.num_clients):
            return list(range(int(self.num_clients)))
        if k <= 0:
            return []
        risk_aware_cfg_enabled = bool(self.config.get("risk_aware_enable", False))
        if (not risk_aware_cfg_enabled) or (not bool(self.current_risk_sampling_enabled)):
            perm = torch.randperm(int(self.num_clients), generator=self._rng).tolist()
            return [int(client_id) for client_id in perm[:k]]
        return self.dgc_repair.sample_clients(k)

    def get_round_repair_metrics(self) -> Dict[str, Any]:
        """Expose per-round DGC-Repair diagnostics for logging."""

        return dict(self.current_repair_metrics)

    def create_audit_groups(self) -> List[List[int]]:
        """Create the audit grouping plan for the current round.

        Returns a flat list of primary groups. Backup members are stored in
        ``self.current_audit_plan`` and ``self._backup_members``.
        """

        if not self.current_audit_active:
            self.current_audit_plan = []
            self._backup_members = {}
            return []

        backup_size = int(np.ceil(self.audit_group_size * self.backup_ratio))
        all_client_ids = list(range(self.num_clients))
        source_pool = list(self.selected_clients)
        if (not self.audit_from_selected_only) or len(source_pool) < self.audit_group_size:
            source_pool = all_client_ids

        if len(source_pool) < self.k_min:
            raise ValueError("Not enough clients to form audit groups with the configured k_min.")

        planned_groups: List[List[int]] = []
        self.current_audit_plan = []
        self._backup_members = {}
        exposure_count: Dict[int, int] = {int(client_id): 0 for client_id in source_pool}
        pair_count: Dict[Tuple[int, int], int] = {}
        pair_penalty = float(self.config.get("audit_group_pair_penalty", 1.0))
        size_penalty = float(self.config.get("audit_group_size_penalty", 0.05))

        for audit_round in range(self.audit_rounds):
            group_count = max(1, int(np.ceil(len(source_pool) / self.audit_group_size)))
            round_groups: List[List[int]] = [[] for _ in range(group_count)]

            round_pool_perm = torch.randperm(len(source_pool), generator=self._rng).tolist()
            round_pool = [source_pool[index] for index in round_pool_perm]
            round_pool.sort(key=lambda client_id: (exposure_count.get(int(client_id), 0), int(client_id)))

            for client_id in round_pool:
                client_id = int(client_id)
                candidate_group_ids = [
                    group_idx for group_idx, members in enumerate(round_groups)
                    if len(members) < self.audit_group_size
                ]
                if len(candidate_group_ids) == 0:
                    break

                best_group_ids: List[int] = []
                best_score: Optional[float] = None
                for group_idx in candidate_group_ids:
                    members = round_groups[group_idx]
                    pair_overlap = 0.0
                    for member_id in members:
                        pair_key = (min(client_id, int(member_id)), max(client_id, int(member_id)))
                        pair_overlap += float(pair_count.get(pair_key, 0))
                    placement_score = pair_penalty * pair_overlap + size_penalty * float(len(members))
                    if best_score is None or placement_score < best_score - 1e-12:
                        best_score = placement_score
                        best_group_ids = [group_idx]
                    elif abs(placement_score - best_score) <= 1e-12:
                        best_group_ids.append(group_idx)

                if len(best_group_ids) == 1:
                    chosen_group_idx = int(best_group_ids[0])
                else:
                    chosen_offset = int(
                        torch.randint(0, len(best_group_ids), (1,), generator=self._rng).item()
                    )
                    chosen_group_idx = int(best_group_ids[chosen_offset])
                round_groups[chosen_group_idx].append(client_id)

            for primary in round_groups:
                if len(primary) < self.k_min:
                    topup_pool = [int(client_id) for client_id in source_pool if int(client_id) not in primary]
                    while len(primary) < self.k_min and len(primary) < self.audit_group_size and len(topup_pool) > 0:
                        best_idx = -1
                        best_score: Optional[float] = None
                        for idx, client_id in enumerate(topup_pool):
                            pair_overlap = 0.0
                            for member_id in primary:
                                pair_key = (min(int(client_id), int(member_id)), max(int(client_id), int(member_id)))
                                pair_overlap += float(pair_count.get(pair_key, 0))
                            placement_score = (
                                float(exposure_count.get(int(client_id), 0))
                                + pair_penalty * pair_overlap
                                + size_penalty * float(len(primary))
                            )
                            if best_score is None or placement_score < best_score - 1e-12:
                                best_score = placement_score
                                best_idx = idx
                        if best_idx < 0:
                            break
                        primary.append(int(topup_pool.pop(best_idx)))

                if len(primary) < self.k_min:
                    continue

                for client_id in primary:
                    exposure_count[int(client_id)] = int(exposure_count.get(int(client_id), 0)) + 1
                for left_idx in range(len(primary)):
                    left_id = int(primary[left_idx])
                    for right_idx in range(left_idx + 1, len(primary)):
                        right_id = int(primary[right_idx])
                        pair_key = (min(left_id, right_id), max(left_id, right_id))
                        pair_count[pair_key] = int(pair_count.get(pair_key, 0)) + 1

                remaining_candidates = [client_id for client_id in all_client_ids if client_id not in primary]
                if len(remaining_candidates) > 0 and backup_size > 0:
                    backup_perm = torch.randperm(len(remaining_candidates), generator=self._rng).tolist()
                    backups = [
                        remaining_candidates[index]
                        for index in backup_perm[: min(backup_size, len(remaining_candidates))]
                    ]
                else:
                    backups = []

                group_id = len(planned_groups)
                planned_groups.append(list(primary))
                self._backup_members[group_id] = backups
                self.current_audit_plan.append(
                    {
                        "group_id": group_id,
                        "audit_round": audit_round,
                        "primary": list(primary),
                        "backup": list(backups),
                    }
                )

        return planned_groups

    def on_group_aggregated(self, group_id: int, group_update: ModelUpdate, group_clients: Sequence[int]) -> None:
        """Register one secure-aggregation group observation."""

        if len(group_clients) == 0:
            return
        if len(group_clients) < self.k_min:
            return
        self.current_group_updates[int(group_id)] = self._clone_update(group_update)
        self.current_group_clients[int(group_id)] = [int(client_id) for client_id in group_clients]
        self.current_dropout_counts.append(len(group_clients))

    def on_aggregation_complete(
        self,
        global_update: ModelUpdate,
        global_model: Module,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> C3SGuardDecision:
        """Finish the round: audit if needed and clean the global update."""

        return self.after_aggregation(
            round_idx=self.current_round,
            aggregated_update=global_update,
            global_model=global_model,
            clean_dataloader=self.clean_dataloader,
            metadata=metadata,
        )

    def get_client_weight(self, client_id: int) -> float:
        """Return the aggregation weight multiplier for one client."""

        if not bool(self.config.get("enable_client_reweighting", True)):
            return 1.0
        if int(client_id) not in self.suspicious_clients:
            return 1.0
        strength = float(max(0.0, min(1.0, self.current_reweight_strength)))
        return 1.0 - strength * (1.0 - float(self.suspicious_weight))

    def _select_reweight_clients(self, ranked_clients: Sequence[int], topk: int) -> Set[int]:
        """Select reweight targets with optional restriction to currently selected clients."""

        topk = max(1, int(topk))
        ranked = [int(client_id) for client_id in ranked_clients]
        if len(ranked) == 0:
            return set()

        if not bool(self.config.get("reweight_selected_only", True)):
            return set(ranked[:topk])

        selected_set = set(int(client_id) for client_id in self.selected_clients)
        selected_ranked = [client_id for client_id in ranked if client_id in selected_set]
        return set(selected_ranked[:topk])

    def before_aggregation(
        self,
        round_idx: int,
        group_updates: Mapping[int, ModelUpdate],
        group_members: Optional[Mapping[int, Sequence[int]]] = None,
        global_model: Optional[Module] = None,
        clean_dataloader: Optional[DataLoader] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> C3SGuardDecision:
        """
        Hook executed before the server aggregates the current round.

        Args:
            round_idx: Current global round index.
            group_updates: Securely aggregated group updates ``Delta_G`` keyed by
                group id. The server does not observe individual client updates.
            group_members: Optional mapping from group id to participating client
                ids when the audit protocol reveals group composition.
            global_model: Optional model snapshot before applying this round.
            clean_dataloader: Optional clean loader overriding the default one.
            metadata: Optional round-level metadata, e.g. selected clients or
                sampling diagnostics.

        Returns:
            A ``C3SGuardDecision`` describing detection and localization outcomes
            that can influence the aggregation policy for this round.
        """

        self.current_round = int(round_idx)
        if global_model is not None:
            self.model = global_model.to(self.device)
            self.cts_intent.model = self.model
            if self.cleaning_enabled:
                self.dgc_clean.model = self.model

        ordered_group_updates = OrderedDict((int(group_id), group_updates[group_id]) for group_id in group_updates.keys())
        self.current_group_updates = OrderedDict(
            (group_id, self._clone_update(update)) for group_id, update in ordered_group_updates.items()
        )
        self.current_group_clients = {
            int(group_id): [int(client_id) for client_id in members]
            for group_id, members in (group_members or {}).items()
        }
        self.current_dropout_counts = [
            len(self.current_group_clients.get(group_id, []))
            for group_id in self.current_group_updates.keys()
        ]

        if not self.current_audit_active and not self.should_audit(round_idx):
            decision = C3SGuardDecision(round_idx=round_idx, audit_triggered=False)
            self.current_decision = decision
            return decision

        decision = self._run_audit_cycle(
            round_idx=round_idx,
            global_model=global_model,
            clean_dataloader=clean_dataloader,
            metadata=metadata,
        )
        self.current_decision = decision
        return decision

    def after_aggregation(
        self,
        round_idx: int,
        aggregated_update: ModelUpdate,
        global_model: Optional[Module] = None,
        clean_dataloader: Optional[DataLoader] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> C3SGuardDecision:
        """
        Hook executed after the server forms the round-level aggregated update.

        Args:
            round_idx: Current global round index.
            aggregated_update: The round-level aggregated update to be accepted
                or cleaned before model application.
            global_model: Optional model snapshot before applying the update.
            clean_dataloader: Optional clean loader overriding the default one.
            metadata: Optional round-level metadata, including prior decisions.

        Returns:
            A ``C3SGuardDecision`` whose cleaning field may contain the sanitized
            update returned by DGC-Clean.
        """

        metadata = dict(metadata or {})
        if global_model is not None:
            self.model = global_model.to(self.device)
            self.cts_intent.model = self.model
            if self.cleaning_enabled:
                self.dgc_clean.model = self.model

        if self.current_audit_active and len(self.current_group_updates) > 0:
            audit_metadata = dict(metadata)
            audit_metadata["global_update_norm"] = float(self._update_norm(aggregated_update))
            if self.current_decision is None or not self.current_decision.audit_triggered:
                self.current_decision = self._run_audit_cycle(
                    round_idx=round_idx,
                    global_model=global_model,
                    clean_dataloader=clean_dataloader,
                    metadata=audit_metadata,
                )
        else:
            self.current_decision = C3SGuardDecision(round_idx=round_idx, audit_triggered=False)

        if (
            bool(self.config.get("dgc_repair_enable", True))
            and bool(self.dgc_repair.enabled)
            and bool(self.cleaning_enabled)
        ):
            return self._after_aggregation_repair(
                round_idx=round_idx,
                aggregated_update=aggregated_update,
                global_model=global_model,
                clean_dataloader=clean_dataloader,
                metadata=metadata,
            )

        effective_p_hat = max(self.current_p_hat, self.current_cleaning_p_hat)
        if self.current_degenerate:
            effective_p_hat = max(effective_p_hat, self.degenerate_p_hat_floor)
            self.audit_period = max(1, self.base_audit_period // 2)
        else:
            self.audit_period = self.base_audit_period

        selected_client_set = set(int(client_id) for client_id in self.selected_clients)
        effective_applied_clients = set(
            int(client_id)
            for client_id in self.suspicious_clients
            if int(client_id) in selected_client_set
        )
        reweighting_applied = bool(
            bool(self.config.get("enable_client_reweighting", True))
            and len(effective_applied_clients) > 0
        )
        oracle_cleaning_rho_value = metadata.get("oracle_cleaning_rho")
        oracle_cleaning_rho = None
        if oracle_cleaning_rho_value is not None:
            oracle_cleaning_rho = float(max(0.0, min(1.0, oracle_cleaning_rho_value)))
        cleaning_allowed_base = bool(
            self.current_audit_active
            and self.y_star_estimate >= 0
            and bool(self.config.get("enable_cleaning", True))
        )
        cleaning_allowed = bool(cleaning_allowed_base)
        latest_audit: Optional[Dict[str, Any]] = None
        latest_s3_debug: Dict[str, Any] = {}
        if self.current_audit_active and len(self.audit_history) > 0:
            latest_audit = self.audit_history[-1]
            if int(latest_audit.get("round", -1)) == int(round_idx):
                latest_s3_debug = dict(latest_audit.get("s3_debug", {}))
            else:
                latest_audit = None
                latest_s3_debug = {}
        top_score = float(latest_s3_debug.get("top_score", 0.0))
        score_gap = float(latest_s3_debug.get("score_gap", 0.0))
        score_ratio = float(latest_s3_debug.get("score_ratio", 1.0))
        consensus_count = int(len(latest_s3_debug.get("consensus_candidates", [])))
        suspicious_count = int(len(self.current_suspicious_set))
        accept_info = self._compute_cleaning_acceptance(
            p_hat=effective_p_hat,
            oracle_cleaning_rho=oracle_cleaning_rho,
            localization_reliable=bool(self.current_localization_reliable),
            top_score=top_score,
            score_gap=score_gap,
            score_ratio=score_ratio,
            consensus_count=consensus_count,
            suspicious_count=suspicious_count,
        )
        cleaning_accept_score = float(accept_info.get("accept_score", 0.0))
        cleaning_tier = str(accept_info.get("tier", "none"))
        cleaning_reject_reason = str(accept_info.get("reject_reason", "none"))
        r_loc = float(accept_info.get("r_loc", 0.0))
        r_cons = float(accept_info.get("r_cons", 0.0))

        base_rho = (
            float(self.config.get("cleaning_rho_w_p_hat", 0.60)) * float(effective_p_hat)
            + float(self.config.get("cleaning_rho_w_loc", 0.25)) * float(r_loc)
            + float(self.config.get("cleaning_rho_w_cons", 0.15)) * float(r_cons)
        )
        rho_max_pred = float(self.config.get("cleaning_rho_max_predicted", 0.20))
        effective_rho = max(0.0, min(rho_max_pred, float(base_rho)))
        if cleaning_tier == "weak":
            effective_rho = min(effective_rho, float(self.config.get("cleaning_rho_max_weak", 0.10)))
            if bool(self.config.get("cleaning_force_weak_when_allowed", True)):
                effective_rho = max(effective_rho, float(self.config.get("cleaning_rho_min_weak", 0.03)))
        elif cleaning_tier == "medium":
            effective_rho = min(effective_rho, float(self.config.get("cleaning_rho_max_medium", 0.20)))
            effective_rho = max(effective_rho, float(self.config.get("cleaning_rho_min_medium", 0.08)))
        elif cleaning_tier == "strong":
            effective_rho = min(max(effective_rho, float(self.config.get("cleaning_rho_min_strong", 0.15))), float(self.config.get("cleaning_rho_max_strong", 0.35)))
        else:
            effective_rho = 0.0
        if oracle_cleaning_rho is not None:
            effective_rho = float(max(0.0, min(1.0, oracle_cleaning_rho)))
        effective_rho_planned = float(effective_rho)

        rho_reference = max(float(self.config.get("cleaning_rho_reference", 0.20)), 1e-6)
        self.current_reweight_strength = self._clip01(float(effective_rho) / rho_reference) if cleaning_tier != "none" else 1.0

        clean_acc_before_value = metadata.get("clean_acc_before")
        clean_acc_before = (
            float(clean_acc_before_value)
            if clean_acc_before_value is not None
            else None
        )
        if cleaning_allowed and clean_acc_before is not None:
            if clean_acc_before < float(self.config.get("cleaning_min_clean_acc_before_apply", 0.0)):
                cleaning_allowed = False
                cleaning_reject_reason = "projected_damage_too_high"
        if cleaning_allowed and float(effective_p_hat) < float(self.config.get("cleaning_min_p_hat_to_apply", 0.01)):
            cleaning_allowed = False
            cleaning_reject_reason = "p_hat_too_small"
        if cleaning_allowed and cleaning_tier == "none":
            cleaning_allowed = False
            if cleaning_reject_reason == "none":
                cleaning_reject_reason = "predicted_benefit_too_low"

        if (not cleaning_allowed_base) and cleaning_reject_reason == "none":
            if not bool(self.config.get("enable_cleaning", True)):
                cleaning_reject_reason = "cleaning_disabled_by_config"
            elif self.y_star_estimate < 0:
                cleaning_reject_reason = "target_unavailable"
            elif not self.current_audit_active:
                cleaning_reject_reason = "no_audit_triggered"
            else:
                cleaning_reject_reason = "candidate_count_too_small"

        distill_steps_override = 0
        if cleaning_tier == "weak":
            distill_steps_override = int(self.config.get("cleaning_weak_distill_steps", 1))
        elif cleaning_tier == "medium":
            distill_steps_override = int(self.config.get("cleaning_medium_distill_steps", 2))
        elif cleaning_tier == "strong":
            distill_steps_override = int(self.config.get("cleaning_strong_distill_steps", 3))
        projection_strength = self._clip01(float(effective_rho) / rho_reference) if effective_rho > 0.0 else 0.0

        if cleaning_allowed:
            probe_set = self._get_probe_set(clean_dataloader=clean_dataloader)
            suspicious_subspace: Optional[Dict[str, Any]] = None
            subspace_method = str(self.dgc_clean.config.get("subspace_method", "parameter_grad")).strip().lower()
            if subspace_method in {"group_diff", "group_diff_svd"} and self.current_B_tilde is not None:
                suspicious_subspace = {
                    "basis": self.current_B_tilde.detach().to(self.device),
                    "rank": int(self.current_B_tilde.shape[1]) if self.current_B_tilde.ndim == 2 else 0,
                    "num_vectors": int(self.current_B_tilde.shape[1]) if self.current_B_tilde.ndim == 2 else 0,
                    "sample_mode": "group_diff_svd",
                }
            cleaning_result = self.dgc_clean.clean_aggregated_update(
                aggregated_update=aggregated_update,
                suspicious_subspace=suspicious_subspace,
                round_idx=round_idx,
                metadata={
                    "y_star": self.y_star_estimate,
                    "p_hat": effective_p_hat,
                    "x_syn": probe_set,
                    "cleaning_confidence": self.current_cleaning_confidence,
                    "localization_reliable": self.current_localization_reliable,
                    "reweighting_applied": reweighting_applied,
                    "effective_rho": float(effective_rho),
                    "cleaning_tier": str(cleaning_tier),
                    "distill_steps_override": int(distill_steps_override),
                    "projection_strength": float(projection_strength),
                },
            )
            if (
                cleaning_tier != "none"
                and float(cleaning_result.aux_stats.get("projection_norm_after", cleaning_result.aux_stats.get("projection_norm", 0.0))) <= 1e-12
                and cleaning_reject_reason == "none"
            ):
                cleaning_reject_reason = "no_viable_projection"
            self.ema_model = deepcopy(self.dgc_clean.ema_model)
        else:
            cleaning_result = None

        clean_candidate_count = int(suspicious_count)
        clean_candidate_available = bool(clean_candidate_count > 0)
        clean_candidate_score_best = float(max(float(top_score), 0.0))
        clean_accept_raw_score = float(cleaning_accept_score)
        clean_accept_final = bool(cleaning_allowed and cleaning_result is not None)
        clean_accept_block_flags = self._default_clean_accept_block_flags()
        if clean_accept_final:
            clean_accept_block_reason = "none"
            clean_accept_path = "legacy_cleaning"
        else:
            if not clean_candidate_available:
                clean_accept_block_reason = "no_candidate_group"
            elif cleaning_result is None:
                clean_accept_block_reason = "cleaning_result_unavailable"
            elif str(cleaning_reject_reason).strip().lower() != "none":
                clean_accept_block_reason = str(cleaning_reject_reason)
            else:
                clean_accept_block_reason = "cleaning_blocked_unknown"
            clean_accept_block_reason, clean_accept_block_flags = self._normalize_clean_accept_block_reason(
                base_reason=str(clean_accept_block_reason),
                candidate_available=bool(clean_candidate_available),
                candidate_score_best=float(clean_candidate_score_best),
                cts_signal_reliable=bool(self.current_localization_reliable),
                safe_teacher_ready=bool(self.current_repair_metrics.get("safe_model_ready", False)),
                reject_override=False,
            )
            clean_accept_path = f"blocked:{clean_accept_block_reason}"
        if (not clean_accept_final) and str(cleaning_reject_reason).strip().lower() == "none":
            cleaning_reject_reason = str(clean_accept_block_reason)

        if self.current_audit_active and len(self.audit_history) > 0:
            last_audit = self.audit_history[-1]
            if int(last_audit.get("round", -1)) == int(round_idx):
                last_audit["cleaning_effective_p_hat"] = float(effective_p_hat)
                last_audit["cleaning_allowed"] = bool(cleaning_allowed)
                last_audit["cleaning_reweighting_applied"] = bool(reweighting_applied)
                last_audit["cleaning_accept_score"] = float(cleaning_accept_score)
                last_audit["cleaning_tier"] = str(cleaning_tier)
                last_audit["cleaning_accept_threshold_weak"] = float(accept_info.get("weak_threshold", self.config.get("cleaning_accept_threshold_weak", 0.35)))
                last_audit["cleaning_accept_threshold_medium"] = float(accept_info.get("medium_threshold", self.config.get("cleaning_accept_threshold_medium", 0.55)))
                last_audit["cleaning_accept_margin_to_weak"] = float(accept_info.get("margin_to_weak", 0.0))
                last_audit["cleaning_accept_margin_to_medium"] = float(accept_info.get("margin_to_medium", 0.0))
                last_audit["cleaning_reject_reason"] = str(
                    "none"
                    if clean_accept_final
                    else (
                        cleaning_reject_reason
                        if str(cleaning_reject_reason).strip().lower() != "none"
                        else str(clean_accept_block_reason)
                    )
                )
                last_audit["clean_candidate_available"] = bool(clean_candidate_available)
                last_audit["clean_candidate_count"] = int(clean_candidate_count)
                last_audit["clean_candidate_score_best"] = float(clean_candidate_score_best)
                last_audit["clean_accept_raw_score"] = float(clean_accept_raw_score)
                last_audit["clean_accept_block_reason"] = str(clean_accept_block_reason)
                last_audit["clean_accept_block_flags"] = {
                    str(key): bool(value) for key, value in clean_accept_block_flags.items()
                }
                last_audit["clean_accept_final"] = float(1.0 if clean_accept_final else 0.0)
                last_audit["clean_accept_path"] = str(clean_accept_path)
                last_audit["effective_rho_planned"] = float(effective_rho_planned)
                last_audit["effective_rho"] = float(effective_rho if cleaning_allowed else 0.0)
                last_audit["effective_projection_planned"] = float(projection_strength)
                last_audit["effective_projection_applied"] = float(
                    projection_strength if cleaning_result is not None else 0.0
                )
                last_audit["effective_reweighting_applied"] = float(
                    self.current_reweight_strength if bool(self.config.get("enable_client_reweighting", True)) else 0.0
                )
                if cleaning_result is not None:
                    actual_distill_steps = int(
                        cleaning_result.aux_stats.get("distill_stats", {}).get("distill_steps", 0)
                    )
                else:
                    actual_distill_steps = 0
                last_audit["effective_distillation_steps"] = int(actual_distill_steps)
                last_audit["force_weak_cleaning"] = bool(
                    bool(self.config.get("cleaning_force_apply_in_auto", True))
                    and cleaning_allowed
                    and cleaning_tier in ("weak", "medium", "strong")
                )
                last_audit["cleaning_policy_version"] = str(
                    self.config.get("cleaning_policy_version", "v2_continuous_tiered")
                )
                last_audit["oracle_cleaning_rho"] = (
                    float(oracle_cleaning_rho) if oracle_cleaning_rho is not None else None
                )
                last_audit["oracle_cleaning_strength_scan_id"] = str(
                    metadata.get("oracle_cleaning_strength_scan_id", "")
                )
                if cleaning_result is not None:
                    last_audit["cleaning_projection_norm_before"] = float(
                        cleaning_result.aux_stats.get("projection_norm_before", cleaning_result.aux_stats.get("projection_norm", 0.0))
                    )
                    last_audit["cleaning_projection_norm_after"] = float(
                        cleaning_result.aux_stats.get("projection_norm_after", cleaning_result.aux_stats.get("projection_norm", 0.0))
                    )
                    last_audit["clean_model_delta_norm"] = float(
                        cleaning_result.aux_stats.get("clean_model_delta_norm", 0.0)
                    )
                if clean_acc_before is not None:
                    last_audit["clean_acc_before"] = float(clean_acc_before)

        self.current_cleaning_accept_score = float(cleaning_accept_score)
        self.current_cleaning_tier = str(cleaning_tier)
        self.current_cleaning_reject_reason = str(
            "none"
            if clean_accept_final
            else (
                cleaning_reject_reason
                if str(cleaning_reject_reason).strip().lower() != "none"
                else str(clean_accept_block_reason)
            )
        )
        self.current_effective_rho = float(effective_rho if cleaning_allowed else 0.0)

        decision = self.current_decision or C3SGuardDecision(round_idx=round_idx, audit_triggered=False)
        decision.cleaning = cleaning_result
        if cleaning_result is not None:
            decision.aggregation_action = "cleaned"
        self.current_decision = decision
        return decision

    def _after_aggregation_repair(
        self,
        *,
        round_idx: int,
        aggregated_update: ModelUpdate,
        global_model: Optional[Module],
        clean_dataloader: Optional[DataLoader],
        metadata: Mapping[str, Any],
    ) -> C3SGuardDecision:
        """DGC-Repair path: gate + risk-aware signals + server-side min-max repair."""

        decision = self.current_decision or C3SGuardDecision(round_idx=round_idx, audit_triggered=False)
        effective_p_hat = float(max(self.current_p_hat, self.current_cleaning_p_hat))
        confirmed_suspects = [int(client_id) for client_id in self.current_confirmed_suspects]
        confirmed_set = {int(client_id) for client_id in confirmed_suspects}
        abnormal_ratio = 0.0
        confirmed_abnormal_ratio = -1.0
        if self.current_audit_active and len(self.audit_history) > 0:
            latest = self.audit_history[-1]
            if int(latest.get("round", -1)) == int(round_idx):
                abnormal_ratio = float(latest.get("abnormal_group_ratio", latest.get("positive_ratio_current", 0.0)))
                group_members = latest.get("group_members", {})
                group_ids = [int(group_id) for group_id in latest.get("group_ids", [])]
                if len(group_ids) > 0:
                    if len(confirmed_set) <= 0:
                        confirmed_abnormal_ratio = 0.0
                    else:
                        confirmed_abnormal_groups = 0
                        for group_id in group_ids:
                            members = group_members.get(group_id, group_members.get(str(group_id), []))
                            member_set = {int(client_id) for client_id in members}
                            group_is_abnormal = bool(
                                group_id in latest.get("anomalous_groups", [])
                                or str(group_id) in latest.get("anomalous_groups", [])
                            )
                            if group_is_abnormal and len(member_set.intersection(confirmed_set)) > 0:
                                confirmed_abnormal_groups += 1
                        confirmed_abnormal_ratio = float(
                            confirmed_abnormal_groups / max(len(group_ids), 1)
                        )
        if not bool(self.current_localization_reliable):
            p_hat_cap_unreliable = float(
                max(self.config.get("repair_p_hat_cap_when_unreliable", 0.12), 0.0)
            )
            abnormal_cap_unreliable = float(
                np.clip(self.config.get("repair_abnormal_ratio_cap_when_unreliable", 0.10), 0.0, 1.0)
            )
            effective_p_hat = float(min(effective_p_hat, p_hat_cap_unreliable))
            abnormal_ratio = float(min(abnormal_ratio, abnormal_cap_unreliable))
            if float(confirmed_abnormal_ratio) >= 0.0:
                confirmed_abnormal_ratio = float(min(confirmed_abnormal_ratio, abnormal_cap_unreliable))

        current_effective_lr = float(metadata.get("current_effective_lr", metadata.get("round_lr", 0.01)))
        server_lr = float(metadata.get("server_lr", 1.0))
        repaired_update, repair_metrics = self.dgc_repair.repair_round_update(
            round_idx=int(round_idx),
            global_model=self.model if global_model is None else global_model,
            aggregated_update=aggregated_update,
            clean_dataloader=clean_dataloader or self.clean_dataloader,
            p_hat=float(effective_p_hat),
            target_label=(int(self.y_star_estimate) if self.y_star_estimate >= 0 else None),
            is_audit_round=bool(self.current_audit_active),
            abnormal_group_ratio=float(abnormal_ratio),
            confirmed_abnormal_ratio=float(confirmed_abnormal_ratio),
            confirmed_suspect_count=int(len(confirmed_suspects)),
            current_effective_lr=float(current_effective_lr),
            server_lr=float(server_lr),
        )
        repair_metrics["repair_effective_p_hat_input"] = float(effective_p_hat)
        repair_metrics["repair_effective_abnormal_ratio_input"] = float(abnormal_ratio)
        repair_metrics["repair_effective_confirmed_abnormal_ratio_input"] = float(confirmed_abnormal_ratio)
        repair_metrics["repair_localization_reliable"] = bool(self.current_localization_reliable)

        self.current_repair_metrics = dict(repair_metrics)
        repair_hint = str(repair_metrics.get("final_update_branch_hint", "repair"))
        clean_accept_proxy_score = float(np.clip(
            0.4 * float(effective_p_hat / max(self.config.get("dense_mode_min_p_hat", 0.08), 1e-6))
            + 0.3 * float(repair_metrics.get("effective_ratio", 1.0))
            + 0.3 * float(max(repair_metrics.get("cts_global_light", 0.0), 0.0)),
            0.0,
        1.0,
        ))
        repair_candidate_require_confirmation = bool(
            self.config.get(
                "repair_candidate_require_confirmation",
                self.config.get("repair_require_confirmed", True),
            )
        )
        repair_candidate_min_safe_acc = float(
            max(self.config.get("repair_candidate_min_safe_acc", 0.45), 0.0)
        )
        repair_candidate_shadow_only = bool(
            self.config.get("repair_candidate_shadow_only", False)
        )
        repair_require_confirmed = bool(self.config.get("repair_require_confirmed", False))
        repair_require_confirmed_alert_override = bool(
            self.config.get("repair_require_confirmed_alert_override", True)
        )
        repair_require_confirmed_alert_min_p_hat = float(
            max(self.config.get("repair_require_confirmed_alert_min_p_hat", 0.05), 0.0)
        )
        repair_block_when_safe_weak = bool(self.config.get("repair_block_when_safe_weak", False))
        repair_block_when_low_confidence = bool(self.config.get("repair_block_when_low_confidence", False))
        repair_cleaned_min_accept_score = float(max(self.config.get("repair_cleaned_min_accept_score", 0.15), 0.0))
        repair_clean_accept_require_localization_reliable = bool(
            self.config.get("repair_clean_accept_require_localization_reliable", True)
        )
        repair_clean_accept_allow_unreliable_with_highconf_confirmed = bool(
            self.config.get("repair_clean_accept_allow_unreliable_with_highconf_confirmed", True)
        )
        repair_clean_accept_min_highconf_confirmed = int(
            max(self.config.get("repair_clean_accept_min_highconf_confirmed", 2), 0)
        )
        repair_clean_accept_highconf_min_fct_z = float(
            self.config.get("repair_clean_accept_highconf_min_fct_z", 1.8)
        )
        repair_clean_accept_highconf_min_pairs = int(
            max(self.config.get("repair_clean_accept_highconf_min_pairs", 4), 0)
        )
        lambda_max_cfg = max(float(self.dgc_repair.config.get("lambda_max", 2.0)), 1e-6)
        repair_lambda_ratio = float(
            np.clip(float(repair_metrics.get("lambda_bd", 0.0)) / lambda_max_cfg, 0.0, 1.0)
        )
        highconf_confirmed_suspects: List[int] = []
        for client_id in confirmed_suspects:
            stats = self.current_fct_stats.get(int(client_id), {})
            fct_z_value = float(stats.get("fct_z", 0.0))
            num_pairs_valid = int(stats.get("num_pairs_valid", stats.get("num_pairs", 0)))
            if (
                fct_z_value >= repair_clean_accept_highconf_min_fct_z
                and num_pairs_valid >= repair_clean_accept_highconf_min_pairs
            ):
                highconf_confirmed_suspects.append(int(client_id))
        highconf_confirmed_count = int(len(highconf_confirmed_suspects))
        safe_probe_acc = float(repair_metrics.get("safe_probe_acc", 0.0))
        safe_model_ready = bool(repair_metrics.get("safe_model_ready", False))
        safe_model_stage = str(repair_metrics.get("safe_model_stage", "none")).strip().lower()
        repair_state_now = str(repair_metrics.get("repair_state", "unknown")).strip().lower()
        repair_alert_like = bool(
            bool(repair_metrics.get("alert_mode", False))
            or repair_state_now in {"alert", "abnormal"}
        )
        cleaning_benefit_proxy = float(
            np.clip(
                0.5 * float(max(repair_metrics.get("cts_global_light", 0.0), 0.0))
                + 0.5 * float(repair_lambda_ratio),
                0.0,
                1.0,
            )
        )
        cleaning_damage_proxy = float(
            np.clip(repair_candidate_min_safe_acc - safe_probe_acc, 0.0, 1.0)
        )
        block_shadow_only = bool(repair_candidate_shadow_only)
        block_no_confirmed_base = bool(repair_candidate_require_confirmation and len(confirmed_suspects) == 0)
        allow_no_confirmed_override = bool(
            block_no_confirmed_base
            and repair_require_confirmed_alert_override
            and repair_alert_like
            and float(effective_p_hat) >= repair_require_confirmed_alert_min_p_hat
        )
        block_no_confirmed = bool(block_no_confirmed_base and (not allow_no_confirmed_override))
        block_safe_model_weak = bool(safe_probe_acc < repair_candidate_min_safe_acc)
        weak_safe_override_enabled = bool(
            self.config.get("repair_candidate_allow_weak_safe_override", True)
        )
        weak_safe_override_min_confirmed = int(
            max(self.config.get("repair_candidate_weak_safe_override_min_confirmed", 2), 0)
        )
        weak_safe_override_min_safe_acc = float(
            max(self.config.get("repair_candidate_weak_safe_override_min_safe_acc", 0.25), 0.0)
        )
        weak_safe_override_require_alert = bool(
            self.config.get("repair_candidate_weak_safe_override_require_alert", True)
        )
        weak_safe_override_applied = False
        if block_safe_model_weak and weak_safe_override_enabled:
            weak_stage_ok = bool(safe_model_stage == "weak")
            confirmed_ok = bool(len(confirmed_suspects) >= weak_safe_override_min_confirmed)
            safe_floor_ok = bool(safe_probe_acc >= weak_safe_override_min_safe_acc)
            alert_ok = bool((not weak_safe_override_require_alert) or repair_alert_like)
            if weak_stage_ok and confirmed_ok and safe_floor_ok and alert_ok:
                block_safe_model_weak = False
                weak_safe_override_applied = True
        block_repair_confidence_low = bool(cleaning_benefit_proxy <= cleaning_damage_proxy)
        localization_reliable_now = bool(self.current_localization_reliable)
        allow_unreliable_localization_override = bool(
            (not localization_reliable_now)
            and repair_clean_accept_allow_unreliable_with_highconf_confirmed
            and highconf_confirmed_count >= repair_clean_accept_min_highconf_confirmed
        )
        block_localization_unreliable = bool(
            repair_clean_accept_require_localization_reliable
            and (not localization_reliable_now)
            and (not allow_unreliable_localization_override)
        )
        cleaned_candidate_allowed = bool(
            (not block_shadow_only)
            and (not block_no_confirmed)
            and (not block_safe_model_weak)
            and (not block_repair_confidence_low)
        )
        if block_shadow_only:
            cleaned_candidate_block_reason = "shadow_only_mode"
            repair_candidate_block_reason = "shadow_only_mode"
        elif block_no_confirmed:
            cleaned_candidate_block_reason = "no_confirmed_suspect"
            repair_candidate_block_reason = "require_confirmation_not_met"
        elif block_safe_model_weak:
            cleaned_candidate_block_reason = "safe_model_weak"
            repair_candidate_block_reason = "safe_acc_too_low"
        elif block_repair_confidence_low:
            cleaned_candidate_block_reason = "repair_confidence_low"
            repair_candidate_block_reason = "benefit_less_than_damage"
        elif weak_safe_override_applied:
            cleaned_candidate_block_reason = "allowed_by_weak_safe_override"
            repair_candidate_block_reason = "allowed"
        else:
            cleaned_candidate_block_reason = "none"
            repair_candidate_block_reason = "allowed"
        repair_metrics["confirmed_suspects"] = [int(client_id) for client_id in confirmed_suspects]
        repair_metrics["confirmed_suspect_count"] = int(len(confirmed_suspects))
        repair_metrics["highconf_confirmed_suspects"] = [int(client_id) for client_id in highconf_confirmed_suspects]
        repair_metrics["highconf_confirmed_count"] = int(highconf_confirmed_count)
        repair_metrics["repair_candidate_require_confirmation"] = bool(
            repair_candidate_require_confirmation
        )
        repair_metrics["repair_candidate_min_safe_acc"] = float(repair_candidate_min_safe_acc)
        repair_metrics["repair_candidate_shadow_only"] = bool(repair_candidate_shadow_only)
        repair_metrics["repair_candidate_allowed"] = bool(cleaned_candidate_allowed)
        repair_metrics["repair_candidate_block_reason"] = str(repair_candidate_block_reason)
        repair_metrics["cleaning_benefit_proxy"] = float(cleaning_benefit_proxy)
        repair_metrics["cleaning_damage_proxy"] = float(cleaning_damage_proxy)
        repair_metrics["cleaned_candidate_allowed"] = bool(cleaned_candidate_allowed)
        repair_metrics["cleaned_candidate_block_reason"] = str(cleaned_candidate_block_reason)
        repair_metrics["cleaned_candidate_block_no_confirmed"] = bool(block_no_confirmed)
        repair_metrics["cleaned_candidate_block_no_confirmed_base"] = bool(block_no_confirmed_base)
        repair_metrics["cleaned_candidate_allow_no_confirmed_override"] = bool(allow_no_confirmed_override)
        repair_metrics["repair_alert_like"] = bool(repair_alert_like)
        repair_metrics["repair_require_confirmed_alert_override"] = bool(
            repair_require_confirmed_alert_override
        )
        repair_metrics["repair_require_confirmed_alert_min_p_hat"] = float(
            repair_require_confirmed_alert_min_p_hat
        )
        repair_metrics["cleaned_candidate_block_safe_model_weak"] = bool(block_safe_model_weak)
        repair_metrics["cleaned_candidate_allow_weak_safe_override"] = bool(
            weak_safe_override_enabled
        )
        repair_metrics["cleaned_candidate_weak_safe_override_applied"] = bool(
            weak_safe_override_applied
        )
        repair_metrics["cleaned_candidate_weak_safe_override_min_confirmed"] = int(
            weak_safe_override_min_confirmed
        )
        repair_metrics["cleaned_candidate_weak_safe_override_min_safe_acc"] = float(
            weak_safe_override_min_safe_acc
        )
        repair_metrics["cleaned_candidate_weak_safe_override_require_alert"] = bool(
            weak_safe_override_require_alert
        )
        repair_metrics["cleaned_candidate_block_repair_confidence_low"] = bool(block_repair_confidence_low)
        repair_metrics["cleaned_candidate_block_localization_unreliable"] = bool(
            block_localization_unreliable
        )
        repair_metrics["repair_clean_accept_require_localization_reliable"] = bool(
            repair_clean_accept_require_localization_reliable
        )
        repair_metrics["repair_clean_accept_allow_unreliable_with_highconf_confirmed"] = bool(
            repair_clean_accept_allow_unreliable_with_highconf_confirmed
        )
        repair_metrics["repair_clean_accept_min_highconf_confirmed"] = int(
            repair_clean_accept_min_highconf_confirmed
        )
        repair_metrics["repair_clean_accept_highconf_min_fct_z"] = float(
            repair_clean_accept_highconf_min_fct_z
        )
        repair_metrics["repair_clean_accept_highconf_min_pairs"] = int(
            repair_clean_accept_highconf_min_pairs
        )
        repair_metrics["repair_clean_accept_localization_reliable_now"] = bool(localization_reliable_now)
        repair_metrics["repair_clean_accept_allow_unreliable_localization_override"] = bool(
            allow_unreliable_localization_override
        )
        repair_metrics["repair_cleaned_min_accept_score"] = float(repair_cleaned_min_accept_score)
        repair_metrics["repair_cleaned_accept_score_proxy"] = float(clean_accept_proxy_score)
        lambda_max_cfg = max(float(self.dgc_repair.config.get("lambda_max", 2.0)), 1e-6)
        repair_lambda_ratio = float(
            np.clip(float(repair_metrics.get("lambda_bd", 0.0)) / lambda_max_cfg, 0.0, 1.0)
        )
        repair_metrics["repair_lambda_ratio"] = float(repair_lambda_ratio)
        self.current_cleaning_accept_score = float(clean_accept_proxy_score)
        self.current_cleaning_tier = "repair"
        self.current_cleaning_reject_reason = "none"
        self.current_effective_rho = float(repair_lambda_ratio)

        cleaning_result = DGCCleanResult(
            round_idx=int(round_idx),
            cleaned_update=repaired_update,
            removed_energy=None,
            suspicious_subspace_rank=None,
            aux_stats={
                "method": "dgc_repair",
                "repair_steps": int(repair_metrics.get("repair_steps", 0)),
                "lambda_bd": float(repair_metrics.get("lambda_bd", 0.0)),
                "repair_lr": float(repair_metrics.get("repair_lr", 0.0)),
                "l_kd": float(repair_metrics.get("l_kd", 0.0)),
                "l_anti_bd": float(repair_metrics.get("l_anti_bd", 0.0)),
                "l_reg": float(repair_metrics.get("l_reg", 0.0)),
                "cts_global_light": float(repair_metrics.get("cts_global_light", 0.0)),
                "cts_global_light_centered": float(repair_metrics.get("cts_global_light_centered", 0.0)),
                "cts_global_light_z": float(repair_metrics.get("cts_global_light_z", 0.0)),
                "cts_checked_this_round": bool(repair_metrics.get("cts_checked_this_round", False)),
                "recent_alert_mode": bool(repair_metrics.get("recent_alert_mode", False)),
                "global_gate_rejected": bool(repair_metrics.get("global_gate_rejected", False)),
                "safe_initialized": bool(repair_metrics.get("safe_initialized", False)),
                "safe_initialized_this_round": bool(repair_metrics.get("safe_initialized_this_round", False)),
                "safe_updated": bool(repair_metrics.get("safe_updated", False)),
                "safe_model_exists": bool(repair_metrics.get("safe_model_exists", False)),
                "safe_model_ready": bool(repair_metrics.get("safe_model_ready", False)),
                "safe_model_stage": str(repair_metrics.get("safe_model_stage", "none")),
                "safe_bootstrap_mode": bool(repair_metrics.get("safe_bootstrap_mode", False)),
                "bootstrap_repair_enabled": bool(repair_metrics.get("bootstrap_repair_enabled", False)),
                "safe_init_source_model": str(repair_metrics.get("safe_init_source_model", "none")),
                "safe_upgrade_allowed": bool(repair_metrics.get("safe_upgrade_allowed", False)),
                "safe_upgrade_reason": str(repair_metrics.get("safe_upgrade_reason", "none")),
                "safe_model_update_reason": str(repair_metrics.get("safe_model_update_reason", "none")),
                "safe_refresh_applied": bool(repair_metrics.get("safe_refresh_applied", False)),
                "safe_refresh_reason": str(repair_metrics.get("safe_refresh_reason", "disabled")),
                "safe_refresh_stale_rounds": int(repair_metrics.get("safe_refresh_stale_rounds", 0)),
                "safe_refresh_margin": float(repair_metrics.get("safe_refresh_margin", 0.0)),
                "safe_refresh_ema": float(repair_metrics.get("safe_refresh_ema", 0.0)),
                "safe_update_ratio_source": str(repair_metrics.get("safe_update_ratio_source", "confirmed")),
                "safe_update_ratio_source_effective": str(repair_metrics.get("safe_update_ratio_source_effective", "raw")),
                "safe_effective_abnormal_ratio": float(repair_metrics.get("safe_effective_abnormal_ratio", 0.0)),
                "safe_raw_abnormal_ratio": float(repair_metrics.get("safe_raw_abnormal_ratio", 0.0)),
                "safe_confirmed_abnormal_ratio": float(repair_metrics.get("safe_confirmed_abnormal_ratio", -1.0)),
                "reject_counter": int(repair_metrics.get("reject_counter", 0)),
                "reject_block_reason": str(repair_metrics.get("reject_block_reason", "none")),
                "final_update_branch_hint": str(repair_metrics.get("final_update_branch_hint", "raw")),
                "confirmed_suspect_count": int(repair_metrics.get("confirmed_suspect_count", 0)),
                "highconf_confirmed_count": int(repair_metrics.get("highconf_confirmed_count", 0)),
                "confirmed_abnormal_ratio": float(repair_metrics.get("confirmed_abnormal_ratio", -1.0)),
                "repair_candidate_allowed": bool(repair_metrics.get("repair_candidate_allowed", True)),
                "repair_candidate_block_reason": str(repair_metrics.get("repair_candidate_block_reason", "allowed")),
                "repair_candidate_require_confirmation": bool(
                    repair_metrics.get("repair_candidate_require_confirmation", True)
                ),
                "repair_candidate_min_safe_acc": float(
                    repair_metrics.get("repair_candidate_min_safe_acc", 0.45)
                ),
                "repair_candidate_shadow_only": bool(
                    repair_metrics.get("repair_candidate_shadow_only", False)
                ),
                "cleaning_benefit_proxy": float(repair_metrics.get("cleaning_benefit_proxy", 0.0)),
                "cleaning_damage_proxy": float(repair_metrics.get("cleaning_damage_proxy", 0.0)),
                "cleaned_candidate_allowed": bool(repair_metrics.get("cleaned_candidate_allowed", True)),
                "cleaned_candidate_block_reason": str(repair_metrics.get("cleaned_candidate_block_reason", "none")),
                "cleaned_candidate_block_no_confirmed": bool(
                    repair_metrics.get("cleaned_candidate_block_no_confirmed", False)
                ),
                "cleaned_candidate_block_safe_model_weak": bool(
                    repair_metrics.get("cleaned_candidate_block_safe_model_weak", False)
                ),
                "cleaned_candidate_block_repair_confidence_low": bool(
                    repair_metrics.get("cleaned_candidate_block_repair_confidence_low", False)
                ),
                "cleaned_candidate_block_localization_unreliable": bool(
                    repair_metrics.get("cleaned_candidate_block_localization_unreliable", False)
                ),
                "repair_cleaned_min_accept_score": float(
                    repair_metrics.get("repair_cleaned_min_accept_score", 0.0)
                ),
                "repair_cleaned_accept_score_proxy": float(
                    repair_metrics.get("repair_cleaned_accept_score_proxy", 0.0)
                ),
                "repair_clean_accept_require_localization_reliable": bool(
                    repair_metrics.get("repair_clean_accept_require_localization_reliable", True)
                ),
                "repair_clean_accept_allow_unreliable_with_highconf_confirmed": bool(
                    repair_metrics.get(
                        "repair_clean_accept_allow_unreliable_with_highconf_confirmed",
                        True,
                    )
                ),
                "repair_clean_accept_min_highconf_confirmed": int(
                    repair_metrics.get("repair_clean_accept_min_highconf_confirmed", 2)
                ),
                "repair_clean_accept_highconf_min_fct_z": float(
                    repair_metrics.get("repair_clean_accept_highconf_min_fct_z", 1.8)
                ),
                "repair_clean_accept_highconf_min_pairs": int(
                    repair_metrics.get("repair_clean_accept_highconf_min_pairs", 4)
                ),
                "repair_clean_accept_localization_reliable_now": bool(
                    repair_metrics.get("repair_clean_accept_localization_reliable_now", False)
                ),
                "repair_clean_accept_allow_unreliable_localization_override": bool(
                    repair_metrics.get(
                        "repair_clean_accept_allow_unreliable_localization_override",
                        False,
                    )
                ),
            },
        )
        decision.cleaning = cleaning_result
        decision.aggregation_action = str(repair_metrics.get("final_update_branch_hint", "repair"))
        if decision.aux_stats is None:
            decision.aux_stats = {}
        decision.aux_stats["dgc_repair_metrics"] = dict(repair_metrics)
        self.current_decision = decision

        if self.current_audit_active and len(self.audit_history) > 0:
            latest = self.audit_history[-1]
            if int(latest.get("round", -1)) == int(round_idx):
                clean_candidate_count = int(len(latest.get("candidate_group_ids", [])))
                clean_candidate_available = bool(clean_candidate_count > 0)
                clean_candidate_score_best = float(latest.get("clean_candidate_score_best", 0.0))
                clean_accept_raw_score = float(clean_accept_proxy_score)
                cts_signal_reliable = bool(
                    latest.get(
                        "cts_signal_reliable",
                        latest.get("cts_debug", {}).get("cts_signal_reliable", False),
                    )
                )
                cts_signal_unreliable_reason = str(
                    latest.get(
                        "cts_signal_unreliable_reason",
                        latest.get("cts_debug", {}).get("cts_signal_unreliable_reason", "none"),
                    )
                )
                cts_reliability_consistency_candidates = [
                    latest.get("cts_reliability_consistency", None),
                    latest.get("candidate_selected_consistency", None),
                    latest.get("candidate_topk_consistency_mean", None),
                    latest.get(
                        "mean_consistency",
                        latest.get("cts_debug", {}).get("mean_consistency", 0.0),
                    ),
                ]
                cts_reliability_consistency = 0.0
                for _cons_val in cts_reliability_consistency_candidates:
                    if _cons_val is None:
                        continue
                    try:
                        cts_reliability_consistency = float(_cons_val)
                        break
                    except (TypeError, ValueError):
                        continue
                cts_reliability_consistency_source = str(
                    latest.get(
                        "cts_reliability_consistency_source",
                        latest.get("cts_debug", {}).get(
                            "cts_reliability_consistency_source",
                            "global_mean",
                        ),
                    )
                )
                cts_z_value = float(
                    latest.get(
                        "cts_z_value",
                        latest.get("cts_debug", {}).get("cts_z_value", 0.0),
                    )
                )
                min_candidate_score = float(
                    max(self.config.get("repair_clean_accept_min_candidate_score", 0.20), 0.0)
                )
                min_candidate_score_unreliable = float(
                    max(self.config.get("repair_clean_accept_min_candidate_score_unreliable", 0.90), 0.0)
                )
                min_p_hat_unreliable = float(
                    max(self.config.get("repair_clean_accept_min_p_hat_unreliable", 0.10), 0.0)
                )
                min_consistency_unreliable = float(
                    np.clip(self.config.get("repair_clean_accept_min_consistency_unreliable", 0.45), 0.0, 1.0)
                )
                min_cts_z_unreliable = float(
                    max(self.config.get("repair_clean_accept_min_cts_z_unreliable", 0.90), 0.0)
                )
                
                candidate_source = str(latest.get("candidate_source", "none")).strip().lower()
                fallback_like_candidate_source = bool(
                    candidate_source in {"fallback_selected_groups", "history_kept_bridge"}
                    or bool(latest.get("candidate_from_hist_keep_bridge", False))
                )
                disallow_unreliable_reasons = {
                    str(reason).strip().lower()
                    for reason in self.config.get(
                        "repair_clean_accept_disallow_unreliable_if_reasons",
                        ["consistency_below_threshold", "cts_z_below_threshold"],
                    )
                    if str(reason).strip()
                }
                allow_unreliable_when_strong = bool(
                    self.config.get("repair_clean_accept_allow_unreliable_when_strong", True)
                )
                candidate_score_ok = bool(clean_candidate_score_best >= min_candidate_score)
                unreliable_reason_tokens = {
                    token.strip().lower()
                    for token in str(cts_signal_unreliable_reason).split("|")
                    if token.strip()
                }
                unreliable_reason_disallowed = bool(
                    len(disallow_unreliable_reasons.intersection(unreliable_reason_tokens)) > 0
                )
                
                unreliable_strong_ok = bool(
                    allow_unreliable_when_strong
                    and (not unreliable_reason_disallowed)
                    and (not fallback_like_candidate_source)
                    and clean_candidate_score_best >= min_candidate_score_unreliable
                    
                    and (
                        (
                            cts_reliability_consistency >= min_consistency_unreliable
                            and cts_z_value >= min_cts_z_unreliable
                        )
                        or (
                            cts_reliability_consistency >= (min_consistency_unreliable + 0.10)
                            and float(self.current_cleaning_p_hat) >= min_p_hat_unreliable
                        )
                    )
                )
                reliability_gate_ok = bool(cts_signal_reliable or unreliable_strong_ok)
                cleaning_allowed = bool(cleaning_result is not None and clean_candidate_available)
                clean_accept_final = bool(
                    cleaning_allowed
                    and bool(cleaned_candidate_allowed)
                    and repair_hint in {"repair", "gated"}
                    and candidate_score_ok
                    and reliability_gate_ok
                )
                clean_accept_block_flags = self._default_clean_accept_block_flags()
                if clean_accept_final:
                    clean_accept_block_reason = "none"
                    clean_accept_path = f"repair_path:{repair_hint}"
                else:
                    if not clean_candidate_available:
                        clean_accept_block_reason = "no_candidate_group"
                    elif cleaning_result is None:
                        clean_accept_block_reason = "cleaned_update_unavailable"
                    elif not bool(cleaned_candidate_allowed):
                        clean_accept_block_reason = str(
                            repair_metrics.get("cleaned_candidate_block_reason", "cleaned_candidate_blocked")
                        )
                    elif str(repair_hint).strip().lower() in {"reject", "noop", "raw", "raw_fallback", "rollback"}:
                        clean_accept_block_reason = f"blocked_by_{str(repair_hint).strip().lower()}"
                    elif not candidate_score_ok:
                        clean_accept_block_reason = "candidate_exists_but_score_too_low"
                    elif not reliability_gate_ok:
                        clean_accept_block_reason = "blocked_by_cts_signal_unreliable"
                    else:
                        clean_accept_block_reason = "repair_blocked_unknown"
                    clean_accept_block_reason, clean_accept_block_flags = self._normalize_clean_accept_block_reason(
                        base_reason=str(clean_accept_block_reason),
                        candidate_available=bool(clean_candidate_available),
                        candidate_score_best=float(clean_candidate_score_best),
                        cts_signal_reliable=bool(cts_signal_reliable),
                        safe_teacher_ready=bool(repair_metrics.get("safe_model_ready", False)),
                        reject_override=bool(
                            str(repair_hint).strip().lower() in {"reject", "rollback", "raw", "raw_fallback", "noop"}
                        ),
                    )
                    clean_accept_path = f"blocked:{clean_accept_block_reason}"
                self.current_cleaning_accept_score = float(1.0 if clean_accept_final else clean_accept_proxy_score)
                self.current_cleaning_reject_reason = (
                    "none"
                    if clean_accept_final
                    else str(clean_accept_block_reason)
                )
                latest["repair_steps"] = int(repair_metrics.get("repair_steps", 0))
                latest["repair_steps_final"] = int(repair_metrics.get("repair_steps_final", repair_metrics.get("repair_steps", 0)))
                latest["repair_state"] = str(repair_metrics.get("repair_state", "unknown"))
                latest["repair_state_before_safe_check"] = str(repair_metrics.get("repair_state_before_safe_check", "unknown"))
                latest["repair_state_after_safe_check"] = str(repair_metrics.get("repair_state_after_safe_check", "unknown"))
                latest["repair_steps_planned"] = int(repair_metrics.get("repair_steps_planned", 0))
                latest["repair_steps_after_safe_cap"] = int(repair_metrics.get("repair_steps_after_safe_cap", 0))
                latest["repair_steps_after_lambda_cap"] = int(repair_metrics.get("repair_steps_after_lambda_cap", 0))
                latest["repair_disabled_reason"] = str(repair_metrics.get("repair_disabled_reason", "none"))
                latest["lambda_branch"] = str(repair_metrics.get("lambda_branch", "normal"))
                latest["lambda_base_before_safe_scale"] = float(repair_metrics.get("lambda_base_before_safe_scale", 0.0))
                latest["lambda_safe_scale_ratio"] = float(repair_metrics.get("lambda_safe_scale_ratio", 0.0))
                latest["lambda_safe_scale_floor"] = float(repair_metrics.get("lambda_safe_scale_floor", 0.0))
                latest["lambda_cap_reason"] = str(repair_metrics.get("lambda_cap_reason", "none"))
                latest["lambda_disabled_reason"] = str(repair_metrics.get("lambda_disabled_reason", "none"))
                latest["lambda_bd"] = float(repair_metrics.get("lambda_bd", 0.0))
                latest["repair_lr"] = float(repair_metrics.get("repair_lr", 0.0))
                latest["l_kd"] = float(repair_metrics.get("l_kd", 0.0))
                latest["l_anti_bd"] = float(repair_metrics.get("l_anti_bd", 0.0))
                latest["l_reg"] = float(repair_metrics.get("l_reg", 0.0))
                latest["cts_global_light"] = float(repair_metrics.get("cts_global_light", 0.0))
                latest["cts_global_light_centered"] = float(repair_metrics.get("cts_global_light_centered", 0.0))
                latest["cts_global_light_z"] = float(repair_metrics.get("cts_global_light_z", 0.0))
                latest["cts_checked_this_round"] = bool(repair_metrics.get("cts_checked_this_round", False))
                latest["cts_signal_fresh"] = bool(repair_metrics.get("cts_signal_fresh", False))
                latest["alert_mode"] = bool(repair_metrics.get("alert_mode", False))
                latest["alert_mode_reason"] = str(repair_metrics.get("alert_mode_reason", "none"))
                latest["recent_alert_mode"] = bool(repair_metrics.get("recent_alert_mode", False))
                latest["safe_updated"] = bool(repair_metrics.get("safe_updated", False))
                latest["safe_initialized"] = bool(repair_metrics.get("safe_initialized", False))
                latest["safe_initialized_this_round"] = bool(repair_metrics.get("safe_initialized_this_round", False))
                latest["safe_model_exists"] = bool(repair_metrics.get("safe_model_exists", False))
                latest["safe_model_ready"] = bool(repair_metrics.get("safe_model_ready", False))
                latest["safe_model_stage"] = str(repair_metrics.get("safe_model_stage", "none"))
                latest["safe_bootstrap_mode"] = bool(repair_metrics.get("safe_bootstrap_mode", False))
                latest["bootstrap_repair_enabled"] = bool(repair_metrics.get("bootstrap_repair_enabled", False))
                latest["safe_probe_acc"] = float(repair_metrics.get("safe_probe_acc", 0.0))
                latest["safe_init_probe_acc"] = float(repair_metrics.get("safe_init_probe_acc", repair_metrics.get("safe_probe_acc", 0.0)))
                latest["safe_init_reason"] = str(repair_metrics.get("safe_init_reason", "unknown"))
                latest["safe_init_source_model"] = str(repair_metrics.get("safe_init_source_model", "none"))
                latest["safe_update_reason"] = str(repair_metrics.get("safe_update_reason", "unknown"))
                latest["safe_model_update_reason"] = str(
                    repair_metrics.get("safe_model_update_reason", repair_metrics.get("safe_update_reason", "unknown"))
                )
                latest["safe_init_allowed"] = bool(repair_metrics.get("safe_init_allowed", False))
                latest["safe_update_allowed"] = bool(repair_metrics.get("safe_update_allowed", False))
                latest["safe_init_used_fallback"] = bool(repair_metrics.get("safe_init_used_fallback", False))
                latest["safe_init_threshold_used"] = str(repair_metrics.get("safe_init_threshold_used", "none"))
                latest["safe_init_blocking_condition"] = str(repair_metrics.get("safe_init_blocking_condition", "none"))
                latest["safe_init_context_ok"] = bool(repair_metrics.get("safe_init_context_ok", False))
                latest["safe_upgrade_allowed"] = bool(repair_metrics.get("safe_upgrade_allowed", False))
                latest["safe_upgrade_reason"] = str(repair_metrics.get("safe_upgrade_reason", "none"))
                latest["safe_update_context_ok"] = bool(repair_metrics.get("safe_update_context_ok", False))
                latest["safe_signal_source"] = str(
                    repair_metrics.get("safe_signal_source", "dcbd")
                )
                latest["safe_signal_source_effective"] = str(
                    repair_metrics.get("safe_signal_source_effective", "dcbd")
                )
                latest["safe_refresh_applied"] = bool(repair_metrics.get("safe_refresh_applied", False))
                latest["safe_refresh_reason"] = str(repair_metrics.get("safe_refresh_reason", "disabled"))
                latest["safe_refresh_stale_rounds"] = int(repair_metrics.get("safe_refresh_stale_rounds", 0))
                latest["safe_refresh_margin"] = float(repair_metrics.get("safe_refresh_margin", 0.0))
                latest["safe_refresh_ema"] = float(repair_metrics.get("safe_refresh_ema", 0.0))
                latest["safe_refresh_last_update_round"] = int(
                    repair_metrics.get("safe_refresh_last_update_round", -1)
                )
                latest["safe_refresh_rounds_since_last_update"] = int(
                    repair_metrics.get("safe_refresh_rounds_since_last_update", 0)
                )
                latest["safe_refresh_confirmed_count"] = int(
                    repair_metrics.get("safe_refresh_confirmed_count", 0)
                )
                latest["safe_refresh_max_confirmed"] = int(
                    repair_metrics.get("safe_refresh_max_confirmed", 3)
                )
                latest["safe_update_ratio_source"] = str(
                    repair_metrics.get("safe_update_ratio_source", "confirmed")
                )
                latest["safe_update_ratio_source_effective"] = str(
                    repair_metrics.get("safe_update_ratio_source_effective", "raw")
                )
                latest["safe_effective_abnormal_ratio"] = float(
                    repair_metrics.get("safe_effective_abnormal_ratio", abnormal_ratio)
                )
                latest["safe_raw_abnormal_ratio"] = float(
                    repair_metrics.get("safe_raw_abnormal_ratio", abnormal_ratio)
                )
                latest["safe_confirmed_abnormal_ratio"] = float(
                    repair_metrics.get("safe_confirmed_abnormal_ratio", confirmed_abnormal_ratio)
                )
                latest["repair_effective_p_hat_input"] = float(
                    repair_metrics.get("repair_effective_p_hat_input", effective_p_hat)
                )
                latest["repair_effective_abnormal_ratio_input"] = float(
                    repair_metrics.get("repair_effective_abnormal_ratio_input", abnormal_ratio)
                )
                latest["repair_effective_confirmed_abnormal_ratio_input"] = float(
                    repair_metrics.get("repair_effective_confirmed_abnormal_ratio_input", confirmed_abnormal_ratio)
                )
                latest["repair_localization_reliable"] = bool(
                    repair_metrics.get("repair_localization_reliable", self.current_localization_reliable)
                )
                latest["global_gate_rejected"] = bool(repair_metrics.get("global_gate_rejected", False))
                latest["reject_reason"] = str(repair_metrics.get("reject_reason", "none"))
                latest["reject_allowed"] = bool(repair_metrics.get("reject_allowed", True))
                latest["reject_blocked_by_no_safe_model"] = bool(repair_metrics.get("reject_blocked_by_no_safe_model", False))
                latest["reject_fallback_branch_used"] = bool(repair_metrics.get("reject_fallback_branch_used", False))
                latest["reject_block_reason"] = str(repair_metrics.get("reject_block_reason", "none"))
                latest["reject_counter"] = int(repair_metrics.get("reject_counter", 0))
                latest["consecutive_reject_count"] = int(repair_metrics.get("consecutive_reject_count", 0))
                latest["selection_candidate"] = "cleaned"
                latest["final_update_branch_hint"] = str(repair_metrics.get("final_update_branch_hint", "repair"))
                latest["final_update_branch"] = str(repair_metrics.get("final_update_branch_hint", "repair"))
                latest["confirmed_suspects"] = [int(client_id) for client_id in confirmed_suspects]
                latest["confirmed_suspect_count"] = int(len(confirmed_suspects))
                latest["highconf_confirmed_suspects"] = [
                    int(client_id)
                    for client_id in repair_metrics.get("highconf_confirmed_suspects", [])
                ]
                latest["highconf_confirmed_count"] = int(
                    repair_metrics.get("highconf_confirmed_count", 0)
                )
                latest["repair_candidate_allowed"] = bool(
                    repair_metrics.get("repair_candidate_allowed", True)
                )
                latest["repair_candidate_block_reason"] = str(
                    repair_metrics.get("repair_candidate_block_reason", "allowed")
                )
                latest["repair_candidate_require_confirmation"] = bool(
                    repair_metrics.get("repair_candidate_require_confirmation", True)
                )
                latest["repair_candidate_min_safe_acc"] = float(
                    repair_metrics.get("repair_candidate_min_safe_acc", 0.45)
                )
                latest["repair_candidate_shadow_only"] = bool(
                    repair_metrics.get("repair_candidate_shadow_only", False)
                )
                latest["cleaned_candidate_allowed"] = bool(repair_metrics.get("cleaned_candidate_allowed", True))
                latest["cleaned_candidate_block_reason"] = str(
                    repair_metrics.get("cleaned_candidate_block_reason", "none")
                )
                latest["cleaned_candidate_block_no_confirmed"] = bool(
                    repair_metrics.get("cleaned_candidate_block_no_confirmed", False)
                )
                latest["cleaned_candidate_block_safe_model_weak"] = bool(
                    repair_metrics.get("cleaned_candidate_block_safe_model_weak", False)
                )
                latest["cleaned_candidate_block_repair_confidence_low"] = bool(
                    repair_metrics.get("cleaned_candidate_block_repair_confidence_low", False)
                )
                latest["cleaned_candidate_block_localization_unreliable"] = bool(
                    repair_metrics.get("cleaned_candidate_block_localization_unreliable", False)
                )
                latest["repair_clean_accept_require_localization_reliable"] = bool(
                    repair_metrics.get("repair_clean_accept_require_localization_reliable", True)
                )
                latest["repair_clean_accept_allow_unreliable_with_highconf_confirmed"] = bool(
                    repair_metrics.get(
                        "repair_clean_accept_allow_unreliable_with_highconf_confirmed",
                        True,
                    )
                )
                latest["repair_clean_accept_min_highconf_confirmed"] = int(
                    repair_metrics.get("repair_clean_accept_min_highconf_confirmed", 2)
                )
                latest["repair_clean_accept_highconf_min_fct_z"] = float(
                    repair_metrics.get("repair_clean_accept_highconf_min_fct_z", 1.8)
                )
                latest["repair_clean_accept_highconf_min_pairs"] = int(
                    repair_metrics.get("repair_clean_accept_highconf_min_pairs", 4)
                )
                
                
                latest["repair_clean_accept_localization_reliable_now"] = bool(
                    repair_metrics.get("repair_clean_accept_localization_reliable_now", False)
                )
                latest["repair_clean_accept_allow_unreliable_localization_override"] = bool(
                    repair_metrics.get(
                        "repair_clean_accept_allow_unreliable_localization_override",
                        False,
                    )
                )
                latest["cleaning_allowed"] = bool(cleaning_allowed)
                latest["cleaning_accepted"] = bool(clean_accept_final)
                latest["cleaning_reject_reason"] = (
                    "none" if clean_accept_final else str(clean_accept_block_reason)
                )
                latest["clean_candidate_available"] = bool(clean_candidate_available)
                latest["clean_candidate_count"] = int(clean_candidate_count)
                latest["clean_candidate_score_best"] = float(clean_candidate_score_best)
                latest["clean_accept_raw_score"] = float(clean_accept_raw_score)
                latest["clean_accept_block_reason"] = str(clean_accept_block_reason)
                latest["clean_accept_block_flags"] = {
                    str(key): bool(value) for key, value in clean_accept_block_flags.items()
                }
                latest["clean_accept_candidate_score_ok"] = bool(candidate_score_ok)
                latest["clean_accept_reliability_gate_ok"] = bool(reliability_gate_ok)
                latest["clean_accept_unreliable_strong_ok"] = bool(unreliable_strong_ok)
                latest["clean_accept_unreliable_reason_disallowed"] = bool(unreliable_reason_disallowed)
                latest["cleaning_benefit_proxy"] = float(
                    repair_metrics.get("cleaning_benefit_proxy", latest.get("cleaning_benefit_proxy", 0.0))
                )
                latest["cleaning_damage_proxy"] = float(
                    repair_metrics.get("cleaning_damage_proxy", latest.get("cleaning_damage_proxy", 0.0))
                )
                latest["clean_accept_candidate_fallback_like"] = bool(fallback_like_candidate_source)
                latest["clean_accept_mean_consistency"] = float(cts_reliability_consistency)
                latest["clean_accept_consistency_source"] = str(cts_reliability_consistency_source)
                latest["clean_accept_cts_z_value"] = float(cts_z_value)
                latest["clean_accept_final"] = float(1.0 if clean_accept_final else 0.0)
                latest["clean_accept_path"] = str(clean_accept_path)
                latest["cleaning_accept_score"] = float(self.current_cleaning_accept_score)
                latest["cleaning_tier"] = "repair" if clean_accept_final else "none"
                latest["effective_rho_planned"] = float(repair_lambda_ratio)
                latest["effective_rho"] = float(repair_lambda_ratio if clean_accept_final else 0.0)
                latest["effective_projection_planned"] = float(repair_lambda_ratio)
                latest["effective_projection_applied"] = float(
                    repair_lambda_ratio if clean_accept_final else 0.0
                )
                latest["effective_distillation_steps"] = int(
                    repair_metrics.get("repair_steps_final", repair_metrics.get("repair_steps", 0))
                    if clean_accept_final
                    else 0
                )
                latest["repair_lambda_ratio"] = float(repair_lambda_ratio)
                latest["repair_alert_like"] = bool(repair_alert_like)
                latest["repair_require_confirmed_alert_override"] = bool(
                    repair_require_confirmed_alert_override
                )
                latest["repair_require_confirmed_alert_min_p_hat"] = float(
                    repair_require_confirmed_alert_min_p_hat
                )
                latest["cleaned_candidate_block_no_confirmed_base"] = bool(block_no_confirmed_base)
                latest["cleaned_candidate_allow_no_confirmed_override"] = bool(
                    allow_no_confirmed_override
                )
                latest["assert_abnormal_steps_planned"] = bool(repair_metrics.get("assert_abnormal_steps_planned", True))
                latest["assert_lambda_cap_reason_present"] = bool(repair_metrics.get("assert_lambda_cap_reason_present", True))
                latest["assert_safe_init_reason_present"] = bool(repair_metrics.get("assert_safe_init_reason_present", True))
                latest["assert_safe_update_reason_present"] = bool(repair_metrics.get("assert_safe_update_reason_present", True))
        # Keep the latest enriched metrics snapshot for external round logger.
        self.current_repair_metrics = dict(repair_metrics)
        return decision

    def state_dict(self) -> Dict[str, Any]:
        """Serialize the controller and all sub-module states."""

        return {
            "config": dict(self.config),
            "current_round": self.current_round,
            "audit_history": list(self.audit_history),
            "A_matrix": self.A_matrix.copy(),
            "b_g_history": list(self.b_g_history),
            "suspicious_clients": sorted(self.suspicious_clients),
            "current_confirmed_suspects": [int(client_id) for client_id in self.current_confirmed_suspects],
            "current_fct_stats": {
                int(client_id): dict(stats) for client_id, stats in self.current_fct_stats.items()
            },
            "current_fct_fallback_reason": str(self.current_fct_fallback_reason),
            "historical_score_sum": dict(self._historical_score_sum),
            "historical_score_weight": dict(self._historical_score_weight),
            "historical_top_count": dict(self._historical_top_count),
            "historical_anomalous_audits": int(self._historical_anomalous_audits),
            "historical_reliable_audits": int(self._historical_reliable_audits),
            "fct_confirm_history": {
                int(client_id): [int(v) for v in values]
                for client_id, values in self._fct_confirm_history.items()
            },
            "risk_candidate_precision_history": [float(v) for v in self._risk_candidate_precision_history],
            "current_risk_sampling_enabled": bool(self.current_risk_sampling_enabled),
            "current_risk_activation_reason": str(self.current_risk_activation_reason),
            "y_star_estimate": self.y_star_estimate,
            "audit_period": self.audit_period,
            "current_B_tilde": (
                self.current_B_tilde.detach().cpu() if self.current_B_tilde is not None else None
            ),
            "B_tilde_updated_at": int(self.B_tilde_updated_at),
            "B_tilde_diagnostics": dict(self.B_tilde_diagnostics),
            "btilde_quality_history": list(self.btilde_quality_history),
            "w_init_flat": self.w_init_flat.detach().cpu(),
            "weight_purify_history": list(self.weight_purify_history),
            "weight_purify_stats": dict(self.weight_purify_stats),
            "continuous_proj_stats": dict(self.continuous_proj_stats),
            "continuous_proj_history": dict(self.continuous_proj_history),
            "dgc_repair_risk_scores": self.dgc_repair.risk_scores.detach().cpu(),
            "dgc_repair_metrics": dict(self.current_repair_metrics),
            "cts_intent": self.cts_intent.state_dict(),
            "s3_loc": self.s3_loc.state_dict(),
            "dgc_clean": self.dgc_clean.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the controller and all sub-module states."""

        self.config = {**self.default_config(), **dict(state_dict.get("config", {}))}
        self.cleaning_enabled = bool(self.config.get("enable_cleaning", True))
        self.current_round = int(state_dict.get("current_round", -1))
        self.audit_history = list(state_dict.get("audit_history", []))
        self.A_matrix = np.asarray(
            state_dict.get("A_matrix", np.zeros((0, self.num_clients), dtype=np.float32)),
            dtype=np.float32,
        )
        self.b_g_history = [bool(value) for value in state_dict.get("b_g_history", [])]
        self.suspicious_clients = {int(client_id) for client_id in state_dict.get("suspicious_clients", [])}
        self.current_confirmed_suspects = [
            int(client_id) for client_id in state_dict.get("current_confirmed_suspects", [])
        ]
        self.current_fct_stats = {
            int(client_id): dict(stats)
            for client_id, stats in state_dict.get("current_fct_stats", {}).items()
        }
        self.current_fct_fallback_reason = str(
            state_dict.get("current_fct_fallback_reason", self.current_fct_fallback_reason)
        )
        self._historical_score_sum = {
            int(client_id): float(score)
            for client_id, score in state_dict.get("historical_score_sum", {}).items()
        }
        self._historical_score_weight = {
            int(client_id): float(weight)
            for client_id, weight in state_dict.get("historical_score_weight", {}).items()
        }
        self._historical_top_count = {
            int(client_id): float(count)
            for client_id, count in state_dict.get("historical_top_count", {}).items()
        }
        self._historical_anomalous_audits = int(state_dict.get("historical_anomalous_audits", 0))
        self._historical_reliable_audits = int(state_dict.get("historical_reliable_audits", 0))
        self._fct_confirm_history = {
            int(client_id): [int(v) for v in values]
            for client_id, values in state_dict.get("fct_confirm_history", {}).items()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        }
        self._risk_candidate_precision_history = [
            float(value) for value in state_dict.get("risk_candidate_precision_history", [])
        ]
        self.current_risk_sampling_enabled = bool(
            state_dict.get("current_risk_sampling_enabled", False)
        )
        self.current_risk_activation_reason = str(
            state_dict.get("current_risk_activation_reason", "disabled")
        )
        self.y_star_estimate = int(state_dict.get("y_star_estimate", -1))
        self.audit_period = int(state_dict.get("audit_period", self.base_audit_period))
        b_tilde = state_dict.get("current_B_tilde")
        if b_tilde is None:
            self.current_B_tilde = None
        else:
            self.current_B_tilde = torch.as_tensor(b_tilde, device=self.device).float()
        self.B_tilde_updated_at = int(state_dict.get("B_tilde_updated_at", -1))
        self.B_tilde_diagnostics = dict(state_dict.get("B_tilde_diagnostics", {}))
        self.btilde_quality_history = list(state_dict.get("btilde_quality_history", []))
        w_init_flat = state_dict.get("w_init_flat")
        if w_init_flat is None:
            self.w_init_flat = self.dgc_clean._flatten_model_parameters(self.model).detach().to(self.device).clone()
        else:
            self.w_init_flat = torch.as_tensor(w_init_flat, device=self.device).float().detach().clone()
        self.weight_purify_history = list(state_dict.get("weight_purify_history", []))
        self.weight_purify_stats = dict(state_dict.get("weight_purify_stats", self.weight_purify_stats))
        self.continuous_proj_stats = dict(state_dict.get("continuous_proj_stats", self.continuous_proj_stats))
        self.continuous_proj_history = dict(state_dict.get("continuous_proj_history", {}))
        risk_scores = state_dict.get("dgc_repair_risk_scores")
        if risk_scores is not None:
            self.dgc_repair.risk_scores = torch.as_tensor(risk_scores).float().cpu()
        self.current_repair_metrics = dict(state_dict.get("dgc_repair_metrics", self.current_repair_metrics))
        self.cts_intent.load_state_dict(state_dict.get("cts_intent", {}))
        self.s3_loc.load_state_dict(state_dict.get("s3_loc", {}))
        self.dgc_clean.load_state_dict(state_dict.get("dgc_clean", {}))
        self.ema_model = deepcopy(self.dgc_clean.ema_model)
        if self.cleaning_enabled:
            self.ema_model = self.ema_model.to(self.device)

    def _run_audit_cycle(
        self,
        round_idx: int,
        global_model: Optional[Module],
        clean_dataloader: Optional[DataLoader],
        metadata: Optional[Mapping[str, Any]],
    ) -> C3SGuardDecision:
        metadata = dict(metadata or {})
        if len(self.current_group_updates) == 0:
            return C3SGuardDecision(round_idx=round_idx, audit_triggered=False)

        probe_set = self._get_probe_set(clean_dataloader=clean_dataloader)
        group_update_mapping_raw = OrderedDict(self.current_group_updates)
        cts_group_update_scale = str(self.config.get("cts_group_update_scale", "mean")).strip().lower()
        if cts_group_update_scale not in {"sum", "mean"}:
            cts_group_update_scale = "mean"
        cts_group_norm_align = bool(self.config.get("cts_group_norm_align", True))
        cts_group_norm_ratio = max(float(self.config.get("cts_group_norm_ratio", 0.75)), 1e-6)
        cts_group_norm_min_scale = max(float(self.config.get("cts_group_norm_min_scale", 0.05)), 1e-6)
        cts_group_norm_allow_upscale = bool(self.config.get("cts_group_norm_allow_upscale", False))
        global_update_norm = max(float(metadata.get("global_update_norm", 0.0)), 0.0)
        group_update_mapping: "OrderedDict[int, ModelUpdate]" = OrderedDict()
        group_eval_scale_map: Dict[int, float] = {}
        group_eval_norm_map: Dict[int, float] = {}
        for group_id, update in group_update_mapping_raw.items():
            if cts_group_update_scale == "mean":
                member_count = max(int(len(self.current_group_clients.get(int(group_id), []))), 1)
                base_scale = 1.0 / float(member_count)
            else:
                base_scale = 1.0
            scaled_update = self._scale_update(update, float(base_scale))
            effective_scale = float(base_scale)
            if cts_group_norm_align and global_update_norm > 0.0:
                scaled_norm = float(self._update_norm(scaled_update))
                if scaled_norm > 0.0:
                    target_norm = cts_group_norm_ratio * global_update_norm
                    align_scale = target_norm / max(scaled_norm, 1e-12)
                    if not cts_group_norm_allow_upscale:
                        align_scale = min(align_scale, 1.0)
                    align_scale = max(float(align_scale), cts_group_norm_min_scale)
                    scaled_update = self._scale_update(scaled_update, float(align_scale))
                    effective_scale *= float(align_scale)
            group_update_mapping[int(group_id)] = scaled_update
            group_eval_scale_map[int(group_id)] = float(effective_scale)
            group_eval_norm_map[int(group_id)] = float(self._update_norm(scaled_update))
        detection_metadata = {
            "w_t": global_model.state_dict() if global_model is not None else self.model.state_dict(),
            "x_syn": probe_set,
            "known_target_label": metadata.get("target_label", self.config.get("target_label")),
            "force_known_target": metadata.get(
                "force_known_target",
                self.config.get("force_known_target", False),
            ),
        }
        detection = self.cts_intent.detect(
            group_updates=group_update_mapping,
            global_model=global_model,
            clean_dataloader=clean_dataloader,
            round_idx=round_idx,
            metadata=detection_metadata,
        )

        def _detection_collapse_stats(result: CTSIntentResult) -> Tuple[float, float]:
            group_order = result.aux_stats.get("group_order", list(group_update_mapping.keys()))
            scores: List[float] = []
            for gid in group_order:
                value = float(result.group_scores.get(int(gid), 0.0))
                if np.isfinite(value):
                    scores.append(value)
            if len(scores) == 0:
                return 0.0, 0.0
            arr = np.asarray(scores, dtype=np.float32)
            return float(np.max(np.abs(arr))), float(np.std(arr))

        reprobe_applied = False
        reprobe_reason = "not_triggered"
        reprobe_boost_used = 1.0
        reprobe_max_abs_before, reprobe_std_before = _detection_collapse_stats(detection)
        reprobe_max_abs_after = reprobe_max_abs_before
        reprobe_std_after = reprobe_std_before
        if bool(self.config.get("cts_reprobe_on_collapse", True)):
            collapse_abs_thr = max(float(self.config.get("cts_reprobe_max_abs_threshold", 0.02)), 0.0)
            collapse_std_thr = max(float(self.config.get("cts_reprobe_std_threshold", 0.01)), 0.0)
            collapse_detected = bool(
                reprobe_max_abs_before <= collapse_abs_thr
                or reprobe_std_before <= collapse_std_thr
            )
            if collapse_detected:
                boost_factor = max(float(self.config.get("cts_reprobe_boost_factor", 4.0)), 1.0)
                max_scale_cap = max(float(self.config.get("cts_reprobe_max_scale", 1.0)), 1e-6)
                boosted_updates: "OrderedDict[int, ModelUpdate]" = OrderedDict()
                boosted_scale_map: Dict[int, float] = {}
                for group_id, update in group_update_mapping.items():
                    gid = int(group_id)
                    curr_scale = max(float(group_eval_scale_map.get(gid, 1.0)), 1e-12)
                    extra = float(boost_factor)
                    extra = min(extra, max_scale_cap / curr_scale)
                    extra = max(extra, 1.0)
                    boosted_updates[gid] = self._scale_update(update, extra)
                    boosted_scale_map[gid] = float(curr_scale * extra)
                boosted_detection = self.cts_intent.detect(
                    group_updates=boosted_updates,
                    global_model=global_model,
                    clean_dataloader=clean_dataloader,
                    round_idx=round_idx,
                    metadata=detection_metadata,
                )
                boosted_max_abs, boosted_std = _detection_collapse_stats(boosted_detection)
                improved = bool(
                    boosted_max_abs >= reprobe_max_abs_before * 1.25
                    or boosted_std >= reprobe_std_before * 1.25
                )
                if improved:
                    detection = boosted_detection
                    group_update_mapping = boosted_updates
                    group_eval_scale_map = boosted_scale_map
                    group_eval_norm_map = {
                        gid: float(self._update_norm(update))
                        for gid, update in group_update_mapping.items()
                    }
                    reprobe_applied = True
                    reprobe_reason = "collapsed_and_improved"
                    reprobe_boost_used = float(boost_factor)
                    reprobe_max_abs_after = float(boosted_max_abs)
                    reprobe_std_after = float(boosted_std)
                else:
                    reprobe_reason = "collapsed_but_not_improved"
                    reprobe_max_abs_after = float(boosted_max_abs)
                    reprobe_std_after = float(boosted_std)
            else:
                reprobe_reason = "not_collapsed"

        ordered_group_ids = detection.aux_stats.get("group_order", list(group_update_mapping.keys()))
        b_g_map = detection.aux_stats.get("b_g", {})
        self.current_cts_scores = [float(detection.group_scores[group_id]) for group_id in ordered_group_ids]
        anomaly_tail = str(detection.aux_stats.get("anomaly_tail", "right")).strip().lower()
        if anomaly_tail not in {"left", "right"}:
            anomaly_tail = "right"
        sign = 1.0 if anomaly_tail == "right" else -1.0
        tail_alignment_required = bool(
            metadata.get("force_known_target", self.config.get("force_known_target", False))
        )
        z_cts_dir_map = detection.aux_stats.get("z_cts_dir", {})
        b_g_right_map = detection.aux_stats.get("b_g_right", {})
        b_g_left_map = detection.aux_stats.get("b_g_left", {})
        right_mad_z_map = detection.aux_stats.get("right_mad_z", {})
        left_mad_z_map = detection.aux_stats.get("left_mad_z", {})
        anomaly_score_map = detection.aux_stats.get("anomaly_scores", {})
        cts_diff_map_raw = detection.aux_stats.get("diff_group_scores", {})

        def _map_get_bool_local(mapping: Any, group_id: int, default: bool = False) -> bool:
            if isinstance(mapping, Mapping):
                if group_id in mapping:
                    return bool(mapping[group_id])
                key = str(group_id)
                if key in mapping:
                    return bool(mapping[key])
            return bool(default)

        def _map_get_float_local(mapping: Any, group_id: int, default: float = 0.0) -> float:
            if isinstance(mapping, Mapping):
                if group_id in mapping:
                    return float(mapping[group_id])
                key = str(group_id)
                if key in mapping:
                    return float(mapping[key])
            return float(default)

        def _get_oriented_signal(group_id: int, fallback_score: float) -> float:
            if isinstance(z_cts_dir_map, Mapping):
                if group_id in z_cts_dir_map:
                    return float(z_cts_dir_map[group_id])
                key = str(group_id)
                if key in z_cts_dir_map:
                    return float(z_cts_dir_map[key])
            return float(sign * fallback_score)

        def _group_direction_sign(group_id: int) -> float:
            gid = int(group_id)
            right_selected = _map_get_bool_local(b_g_right_map, gid, False)
            left_selected = _map_get_bool_local(b_g_left_map, gid, False)
            if right_selected and not left_selected:
                return 1.0
            if left_selected and not right_selected:
                return -1.0
            right_strength = _map_get_float_local(right_mad_z_map, gid, 0.0)
            left_strength = _map_get_float_local(left_mad_z_map, gid, 0.0)
            if right_strength > left_strength + 1e-6:
                return 1.0
            if left_strength > right_strength + 1e-6:
                return -1.0
            diff_val = _map_get_float_local(cts_diff_map_raw, gid, 0.0)
            if abs(diff_val) > 1e-12:
                return 1.0 if diff_val >= 0.0 else -1.0
            return float(sign)

        def _group_two_sided_strength(group_id: int) -> float:
            gid = int(group_id)
            return float(
                max(
                    abs(_map_get_float_local(z_cts_dir_map, gid, 0.0)),
                    _map_get_float_local(anomaly_score_map, gid, 0.0),
                    _map_get_float_local(right_mad_z_map, gid, 0.0),
                    _map_get_float_local(left_mad_z_map, gid, 0.0),
                )
            )

        def _group_oriented_diff(group_id: int, diff_val: float) -> float:
            if tail_alignment_required:
                return float(sign * float(diff_val))
            return float(_group_direction_sign(int(group_id)) * float(diff_val))

        def _group_tail_soft_mad_z(group_id: int) -> float:
            gid = int(group_id)
            if not tail_alignment_required:
                return float(
                    max(
                        _map_get_float_local(right_mad_z_map, gid, 0.0),
                        _map_get_float_local(left_mad_z_map, gid, 0.0),
                    )
                )
            if sign >= 0.0:
                return float(_map_get_float_local(right_mad_z_map, gid, 0.0))
            return float(_map_get_float_local(left_mad_z_map, gid, 0.0))

        def _group_tail_signal_strength(group_id: int) -> float:
            gid = int(group_id)
            if not tail_alignment_required:
                return float(_group_two_sided_strength(gid))
            fallback_score = float(detection.group_scores.get(gid, 0.0))
            return float(max(0.0, _get_oriented_signal(gid, fallback_score)))

        right_anomalous_count = int(
            sum(1 for group_id in ordered_group_ids if _map_get_bool_local(b_g_right_map, int(group_id), False))
        )
        left_anomalous_count = int(
            sum(1 for group_id in ordered_group_ids if _map_get_bool_local(b_g_left_map, int(group_id), False))
        )

        localization_signal_source_requested = str(
            self.config.get("localization_signal_source", "auto")
        ).strip().lower()
        if localization_signal_source_requested not in {"auto", "oriented", "dcbd", "group_behavior"}:
            localization_signal_source_requested = "auto"
        dcbd_behavior_map = detection.aux_stats.get("dcbd_score", {})
        tsc_target_map = detection.aux_stats.get("tsc_target", {})
        dcbd_anomaly_z_map = detection.aux_stats.get("dcbd_anomaly_z_score", {})
        dcbd_anomaly_decision_map = detection.aux_stats.get("dcbd_anomaly_decision", {})
        group_behavior_map = detection.aux_stats.get("group_behavior_score", {})
        cts_mode_effective = str(detection.aux_stats.get("cts_mode_effective", "raw")).strip().lower()
        dcbd_absolute_signal_priority = bool(
            cts_mode_effective == "dcbd"
            and bool(self.config.get("dcbd_absolute_signal_priority", True))
        )

        def _group_tsc_target(group_id: int) -> float:
            gid = int(group_id)
            return float(_map_get_float_local(tsc_target_map, gid, 0.0))

        def _group_dcbd_score(group_id: int) -> float:
            gid = int(group_id)
            fallback_score = float(detection.group_scores.get(gid, 0.0))
            return float(_map_get_float_local(dcbd_behavior_map, gid, fallback_score))

        def _group_dcbd_anomaly_z(group_id: int) -> float:
            gid = int(group_id)
            return float(max(0.0, _map_get_float_local(dcbd_anomaly_z_map, gid, 0.0)))

        def _group_dcbd_anomaly(group_id: int) -> bool:
            gid = int(group_id)
            return bool(_map_get_bool_local(dcbd_anomaly_decision_map, gid, False))

        dcbd_tsc_hard_gate_enable = bool(
            dcbd_absolute_signal_priority
            and bool(self.config.get("dcbd_tsc_hard_gate_enable", False))
        )
        dcbd_tsc_hard_gate_min = float(
            self.config.get("dcbd_tsc_hard_gate_min", 0.0)
        )
        dcbd_tsc_hard_gate_use_abs = bool(
            self.config.get("dcbd_tsc_hard_gate_use_abs", False)
        )
        dcbd_tsc_hard_gate_apply_anomalous = bool(
            self.config.get("dcbd_tsc_hard_gate_apply_anomalous", True)
        )
        dcbd_tsc_hard_gate_apply_hist_keep = bool(
            self.config.get("dcbd_tsc_hard_gate_apply_hist_keep", True)
        )
        dcbd_tsc_hard_gate_apply_candidate = bool(
            self.config.get("dcbd_tsc_hard_gate_apply_candidate", True)
        )

        def _group_tsc_gate_pass(group_id: int) -> bool:
            if not dcbd_tsc_hard_gate_enable:
                return True
            tsc_val = float(_group_tsc_target(int(group_id)))
            tsc_ref = float(abs(tsc_val)) if dcbd_tsc_hard_gate_use_abs else float(tsc_val)
            return bool(tsc_ref >= dcbd_tsc_hard_gate_min)

        localization_signal_source_effective = str(localization_signal_source_requested)
        if localization_signal_source_requested == "auto":
            if cts_mode_effective == "dcbd" and isinstance(dcbd_behavior_map, Mapping):
                localization_signal_source_effective = "dcbd"
            else:
                localization_signal_source_effective = "oriented"
        signal_map_for_localization: Any = None
        if localization_signal_source_effective == "dcbd":
            signal_map_for_localization = dcbd_behavior_map
        elif localization_signal_source_effective == "group_behavior":
            signal_map_for_localization = group_behavior_map
        self.current_localization_signal_scores = []
        for group_id in ordered_group_ids:
            gid = int(group_id)
            oriented_value = float(_get_oriented_signal(gid, float(detection.group_scores[group_id])))
            if isinstance(signal_map_for_localization, Mapping):
                signal_value = float(_map_get_float_local(signal_map_for_localization, gid, oriented_value))
            else:
                signal_value = float(oriented_value)
            self.current_localization_signal_scores.append(float(signal_value))
        raw_anomalous_groups = [
            int(group_id) for group_id in ordered_group_ids if bool(_map_get_bool_local(b_g_map, int(group_id), False))
        ]
        tsc_hard_gate_filtered_groups: List[int] = []
        if dcbd_tsc_hard_gate_enable and dcbd_tsc_hard_gate_apply_anomalous:
            b_g_map_effective: Dict[int, bool] = {}
            for group_id in ordered_group_ids:
                gid = int(group_id)
                is_raw_anomaly = bool(_map_get_bool_local(b_g_map, gid, False))
                if is_raw_anomaly and (not _group_tsc_gate_pass(gid)):
                    b_g_map_effective[gid] = False
                    tsc_hard_gate_filtered_groups.append(gid)
                else:
                    b_g_map_effective[gid] = bool(is_raw_anomaly)
            b_g_map = b_g_map_effective
        self.current_b_g = [bool(_map_get_bool_local(b_g_map, int(group_id), False)) for group_id in ordered_group_ids]
        self.y_star_estimate = int(detection.aux_stats.get("y_star", -1))

        btilde_update_event: Dict[str, Any] = {
            "updated": False,
            "status": "skipped",
            "reason": "subspace_method_not_group_diff",
            "method": str(self.dgc_clean.config.get("subspace_method", "parameter_grad")),
            "round": int(round_idx),
        }
        subspace_method = str(self.dgc_clean.config.get("subspace_method", "parameter_grad")).strip().lower()
        if subspace_method in {"group_diff", "group_diff_svd"}:
            anomalous_group_ids = [int(group_id) for group_id in ordered_group_ids if bool(b_g_map.get(group_id, False))]
            normal_group_ids = [int(group_id) for group_id in ordered_group_ids if not bool(b_g_map.get(group_id, False))]
            anomalous_updates_flat = [
                self.dgc_clean._flatten_update(group_update_mapping_raw[group_id]).detach()
                for group_id in anomalous_group_ids
                if group_id in group_update_mapping_raw
            ]
            normal_updates_flat = [
                self.dgc_clean._flatten_update(group_update_mapping_raw[group_id]).detach()
                for group_id in normal_group_ids
                if group_id in group_update_mapping_raw
            ]
            b_tilde_basis, b_tilde_diag = build_backdoor_subspace_from_group_diff(
                anomalous_group_updates=anomalous_updates_flat,
                normal_group_updates=normal_updates_flat,
                m=int(self.config.get("btilde_rank", self.dgc_clean.config.get("group_diff_rank", 10))),
                min_anomalous=int(self.config.get("btilde_min_anomalous", self.dgc_clean.config.get("group_diff_min_anomalous", 2))),
                min_energy_ratio=float(self.config.get("btilde_min_energy_ratio", self.dgc_clean.config.get("group_diff_min_energy_ratio", 0.3))),
                device=str(self.device),
            )
            if b_tilde_basis is not None:
                self.current_B_tilde = b_tilde_basis.detach().to(self.device)
                self.B_tilde_updated_at = int(round_idx)
                self.B_tilde_diagnostics = dict(b_tilde_diag)
                self.continuous_proj_stats["B_tilde_update_count"] = float(
                    self.continuous_proj_stats.get("B_tilde_update_count", 0.0) + 1.0
                )
                if float(self.continuous_proj_stats.get("first_B_tilde_round", -1.0)) < 0.0:
                    self.continuous_proj_stats["first_B_tilde_round"] = float(int(round_idx))
                history_item = {
                    "round": int(round_idx),
                    "method": str(b_tilde_diag.get("method", "group_diff_svd")),
                    "energy_ratio": float(b_tilde_diag.get("energy_ratio", 0.0)),
                    "num_anomalous_used": int(b_tilde_diag.get("num_anomalous", len(anomalous_updates_flat))),
                    "num_normal_used": int(b_tilde_diag.get("num_normal", len(normal_updates_flat))),
                    "actual_rank": int(b_tilde_diag.get("actual_rank", 0)),
                    "status": str(b_tilde_diag.get("status", "success")),
                }
                self.btilde_quality_history.append(history_item)
                btilde_update_event = {
                    "updated": True,
                    "status": str(b_tilde_diag.get("status", "success")),
                    "reason": "updated",
                    "method": str(b_tilde_diag.get("method", "group_diff_svd")),
                    "round": int(round_idx),
                    "num_anomalous": int(b_tilde_diag.get("num_anomalous", len(anomalous_updates_flat))),
                    "num_normal": int(b_tilde_diag.get("num_normal", len(normal_updates_flat))),
                    "actual_rank": int(b_tilde_diag.get("actual_rank", 0)),
                    "energy_ratio": float(b_tilde_diag.get("energy_ratio", 0.0)),
                }
            else:
                self.B_tilde_diagnostics = dict(b_tilde_diag)
                btilde_update_event = {
                    "updated": False,
                    "status": "skipped",
                    "reason": str(b_tilde_diag.get("reason", "unknown")),
                    "method": "group_diff_svd",
                    "round": int(round_idx),
                    "num_anomalous": int(len(anomalous_updates_flat)),
                    "num_normal": int(len(normal_updates_flat)),
                }

        current_A_rows = []
        ordered_dropout_counts = []
        ordered_group_assignments = []
        for group_id in ordered_group_ids:
            clients = self.current_group_clients.get(group_id, [])
            row = np.zeros(self.num_clients, dtype=np.float32)
            if len(clients) > 0:
                row[np.asarray(clients, dtype=np.int64)] = 1.0
            current_A_rows.append(row)
            ordered_dropout_counts.append(len(clients))
            ordered_group_assignments.append(list(clients))

        current_A = np.stack(current_A_rows, axis=0) if len(current_A_rows) > 0 else np.zeros((0, self.num_clients), dtype=np.float32)
        pre_anomalous_group_count = int(sum(1 for flag in self.current_b_g if bool(flag)))
        pre_signal_std = float(np.std(np.asarray(self.current_localization_signal_scores, dtype=np.float32))) if len(self.current_localization_signal_scores) > 0 else 0.0
        pre_mad_median = float(detection.aux_stats.get("mad_median", 0.0))
        pre_mad_sigma = float(detection.aux_stats.get("mad_sigma", 0.0))
        pre_max_abs_cts = (
            float(max(abs(float(score)) for score in self.current_cts_scores))
            if len(self.current_cts_scores) > 0
            else 0.0
        )
        pre_cts_z = float(pre_max_abs_cts / max(pre_mad_sigma, 1e-6))
        history_gate_enabled = bool(self.config.get("s3_history_accumulation_gate", True))
        if (not bool(self.cleaning_enabled)) and bool(
            self.config.get("s3_history_disable_gate_when_cleaning_disabled", True)
        ):
            # Detection-only mode: avoid hard history-gating that can discard too many
            # audits and artificially depress localization/AUC estimates.
            history_gate_enabled = False
        history_min_anomalous = int(self.config.get("s3_history_min_anomalous_groups", 1))
        history_min_cts_z = float(self.config.get("s3_history_min_cts_z", 1.0))
        history_min_signal_std = float(self.config.get("s3_history_min_signal_std", 0.5))
        history_allow_abs_cts_fallback = bool(
            self.config.get("s3_history_allow_abs_cts_fallback", True)
        )
        history_min_abs_cts_fallback = float(
            max(self.config.get("s3_history_min_abs_cts_fallback", 0.015), 0.0)
        )
        history_min_signal_std_fallback = float(
            max(self.config.get("s3_history_min_signal_std_fallback", 0.02), 0.0)
        )
        history_accumulated = bool(current_A.shape[0] > 0)
        history_skip_reason = "accepted"
        if history_accumulated and history_gate_enabled:
            if pre_anomalous_group_count < history_min_anomalous:
                history_accumulated = False
                history_skip_reason = "too_few_anomalous_groups"
            elif pre_cts_z < history_min_cts_z:
                if (
                    history_allow_abs_cts_fallback
                    and pre_anomalous_group_count >= history_min_anomalous
                    and float(pre_max_abs_cts) >= history_min_abs_cts_fallback
                    and float(pre_signal_std) >= history_min_signal_std_fallback
                ):
                    history_accumulated = True
                    history_skip_reason = "accepted_via_abs_cts_fallback"
                else:
                    history_accumulated = False
                    history_skip_reason = "cts_z_too_low"
            elif pre_signal_std < history_min_signal_std:
                if (
                    history_allow_abs_cts_fallback
                    and pre_anomalous_group_count >= history_min_anomalous
                    and float(pre_max_abs_cts) >= history_min_abs_cts_fallback
                    and float(pre_signal_std) >= history_min_signal_std_fallback
                ):
                    history_accumulated = True
                    history_skip_reason = "accepted_via_abs_cts_fallback"
                else:
                    history_accumulated = False
                    history_skip_reason = "signal_std_too_low"
        group_score_mapping = {
            int(group_id): float(signal)
            for group_id, signal in zip(ordered_group_ids, self.current_localization_signal_scores)
        }
        diff_score_mapping = cts_diff_map_raw
        consistency_map_hist = detection.aux_stats.get("consistency", {})
        hist_keep_min_consistency = float(
            np.clip(self.config.get("hist_keep_min_consistency", 0.45), 0.0, 1.0)
        )
        hist_keep_min_oriented_diff = float(
            max(self.config.get("hist_keep_min_oriented_diff", 0.003), 0.0)
        )
        hist_keep_allow_abs_fallback = bool(
            self.config.get("hist_keep_allow_abs_fallback", True)
        )
        hist_keep_min_abs_diff = float(
            max(self.config.get("hist_keep_min_abs_diff", 0.002), 0.0)
        )
        hist_keep_min_abs_signal = float(
            max(self.config.get("hist_keep_min_abs_signal", 0.20), 0.0)
        )
        hist_keep_enable_soft_support = bool(
            self.config.get("hist_keep_enable_soft_support", True)
        )
        hist_keep_soft_min_mad_z = float(
            max(self.config.get("hist_keep_soft_min_mad_z", 1.20), 0.0)
        )
        hist_keep_soft_min_consistency = float(
            np.clip(self.config.get("hist_keep_soft_min_consistency", 0.42), 0.0, 1.0)
        )
        dcbd_hist_keep_min_tsc = float(self.config.get("dcbd_hist_keep_min_tsc", 0.0))
        dcbd_hist_keep_min_consistency = float(
            np.clip(self.config.get("dcbd_hist_keep_min_consistency", 0.34), 0.0, 1.0)
        )
        dcbd_hist_keep_min_dcbd_score = float(
            self.config.get("dcbd_hist_keep_min_dcbd_score", 0.0)
        )
        dcbd_hist_keep_min_dcbd_z = float(
            max(self.config.get("dcbd_hist_keep_min_dcbd_z", 1.0), 0.0)
        )
        hist_kept_group_ids: List[int] = []
        if history_accumulated:
            for group_id in ordered_group_ids:
                gid = int(group_id)
                cons_val = float(
                    consistency_map_hist.get(gid, consistency_map_hist.get(str(gid), 0.0))
                )
                selected_or_soft_positive = bool(b_g_map.get(gid, False))
                if (not selected_or_soft_positive) and hist_keep_enable_soft_support:
                    soft_mad_z = float(_group_tail_soft_mad_z(gid))
                    selected_or_soft_positive = bool(
                        soft_mad_z >= hist_keep_soft_min_mad_z
                        and cons_val >= hist_keep_soft_min_consistency
                    )
                if (not selected_or_soft_positive) and (not dcbd_absolute_signal_priority):
                    continue
                diff_val = float(
                    diff_score_mapping.get(gid, diff_score_mapping.get(str(gid), 0.0))
                )
                oriented_diff = float(_group_oriented_diff(gid, diff_val))
                abs_diff = float(abs(diff_val))
                signal_strength = float(_group_tail_signal_strength(gid))
                keep_group = False
                if dcbd_absolute_signal_priority:
                    tsc_val = float(_group_tsc_target(gid))
                    dcbd_score_val = float(_group_dcbd_score(gid))
                    dcbd_z_val = float(_group_dcbd_anomaly_z(gid))
                    tsc_gate_ok = bool(
                        (not dcbd_tsc_hard_gate_apply_hist_keep) or _group_tsc_gate_pass(gid)
                    )
                    dcbd_keep_primary = bool(
                        tsc_val >= dcbd_hist_keep_min_tsc
                        and cons_val >= max(hist_keep_min_consistency, dcbd_hist_keep_min_consistency)
                    )
                    dcbd_keep_support = bool(
                        (
                            _group_dcbd_anomaly(gid)
                            or (
                                dcbd_score_val >= dcbd_hist_keep_min_dcbd_score
                                and dcbd_z_val >= dcbd_hist_keep_min_dcbd_z
                            )
                        )
                        and cons_val >= min(
                            max(hist_keep_min_consistency, dcbd_hist_keep_min_consistency),
                            hist_keep_soft_min_consistency,
                        )
                    )
                    keep_group = bool(tsc_gate_ok and (dcbd_keep_primary or dcbd_keep_support))
                else:
                    keep_primary = bool(
                        oriented_diff >= hist_keep_min_oriented_diff
                        and cons_val >= hist_keep_min_consistency
                    )
                    keep_abs = bool(
                        hist_keep_allow_abs_fallback
                        and (not tail_alignment_required)
                        and abs_diff >= hist_keep_min_abs_diff
                        and signal_strength >= hist_keep_min_abs_signal
                        and cons_val >= min(hist_keep_min_consistency, hist_keep_soft_min_consistency)
                    )
                    keep_group = bool(keep_primary or keep_abs)
                if keep_group:
                    hist_kept_group_ids.append(gid)

        if history_gate_enabled:
            if history_accumulated:
                cts_hist_keep_rule_path = "history_gate_pass"
            else:
                cts_hist_keep_rule_path = f"history_gate_blocked:{history_skip_reason}"
        else:
            cts_hist_keep_rule_path = "history_gate_disabled"

        if history_accumulated:
            if current_A.shape[0] > 0:
                self.A_matrix = np.concatenate([self.A_matrix, current_A], axis=0)
                self.b_g_history.extend(self.current_b_g)
            self.s3_loc.accumulate_observation(
                round_idx=round_idx,
                group_assignments=ordered_group_assignments,
                group_scores=group_score_mapping,
                metadata={
                    "group_ids": ordered_group_ids,
                    "b_g": self.current_b_g,
                    "dropout_counts": ordered_dropout_counts,
                },
            )

        localization_t0 = time.perf_counter()
        suspicious_set, p_hat, degenerate, client_scores, s3_aux = localize_s3_clients(
            A_matrix=self.A_matrix,
            b_g_list=self.b_g_history,
            s=int(self.config["suspicious_topk"]),
            method=str(self.config["s3_loc"].get("method", "counting")),
            group_signal=list(self.s3_loc._group_scores),
            dropout_counts=self._collect_dropout_history(),
            lasso_alpha=float(self.config["s3_loc"].get("lasso_alpha", 0.05)),
            degenerate_threshold=float(self.config["s3_loc"].get("degenerate_threshold", 0.8)),
            contrastive_negative_weight=float(self.config["s3_loc"].get("contrastive_negative_weight", 0.25)),
            contrastive_negative_quantile=float(self.config["s3_loc"].get("contrastive_negative_quantile", 0.9)),
            contrastive_positive_scale=float(self.config["s3_loc"].get("contrastive_positive_scale", 1.0)),
            pos_neg_contrast_enabled=bool(self.config["s3_loc"].get("pos_neg_contrast_enabled", True)),
            pos_neg_contrast_lambda=float(self.config["s3_loc"].get("pos_neg_contrast_lambda", 0.9)),
            pos_neg_contrast_blend=float(self.config["s3_loc"].get("pos_neg_contrast_blend", 0.20)),
            pos_neg_contrast_min_positive_groups=int(
                self.config["s3_loc"].get("pos_neg_contrast_min_positive_groups", 3)
            ),
            pos_neg_contrast_min_negative_groups=int(
                self.config["s3_loc"].get("pos_neg_contrast_min_negative_groups", 3)
            ),
            return_details=True,
        )
        localization_t1 = time.perf_counter()
        current_suspicious_set, current_p_hat, current_degenerate, current_client_scores, current_s3_aux = localize_s3_clients(
            A_matrix=current_A,
            b_g_list=self.current_b_g,
            s=int(self.config["suspicious_topk"]),
            method=str(self.config["s3_loc"].get("method", "counting")),
            group_signal=list(self.current_localization_signal_scores),
            dropout_counts=ordered_dropout_counts,
            lasso_alpha=float(self.config["s3_loc"].get("lasso_alpha", 0.05)),
            degenerate_threshold=float(self.config["s3_loc"].get("degenerate_threshold", 0.8)),
            contrastive_negative_weight=float(self.config["s3_loc"].get("contrastive_negative_weight", 0.25)),
            contrastive_negative_quantile=float(self.config["s3_loc"].get("contrastive_negative_quantile", 0.9)),
            contrastive_positive_scale=float(self.config["s3_loc"].get("contrastive_positive_scale", 1.0)),
            pos_neg_contrast_enabled=bool(self.config["s3_loc"].get("pos_neg_contrast_enabled", True)),
            pos_neg_contrast_lambda=float(self.config["s3_loc"].get("pos_neg_contrast_lambda", 0.9)),
            pos_neg_contrast_blend=float(self.config["s3_loc"].get("pos_neg_contrast_blend", 0.20)),
            pos_neg_contrast_min_positive_groups=int(
                self.config["s3_loc"].get("pos_neg_contrast_min_positive_groups", 3)
            ),
            pos_neg_contrast_min_negative_groups=int(
                self.config["s3_loc"].get("pos_neg_contrast_min_negative_groups", 3)
            ),
            return_details=True,
        )
        localization_t2 = time.perf_counter()
        stabilized_current_scores, stability_stats = self._stabilize_client_scores(current_client_scores)
        stabilized_order = sorted(
            (
                (int(client_id), float(score))
                for client_id, score in stabilized_current_scores.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        raw_order = sorted(
            (
                (int(client_id), max(float(score), 0.0))
                for client_id, score in current_client_scores.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        current_suspicious_set = [
            client_id
            for client_id, _ in stabilized_order[: int(self.config["suspicious_topk"])]
        ]

        localization = self.s3_loc.localize(
            round_idx=round_idx,
            metadata={
                "s": int(self.config["suspicious_topk"]),
                "method": str(self.config["s3_loc"].get("method", "counting")),
                "lasso_alpha": float(self.config["s3_loc"].get("lasso_alpha", 0.05)),
                "degenerate_threshold": float(self.config["s3_loc"].get("degenerate_threshold", 0.8)),
                "contrastive_negative_weight": float(self.config["s3_loc"].get("contrastive_negative_weight", 0.25)),
                "contrastive_negative_quantile": float(self.config["s3_loc"].get("contrastive_negative_quantile", 0.9)),
                "contrastive_positive_scale": float(self.config["s3_loc"].get("contrastive_positive_scale", 1.0)),
                "pos_neg_contrast_enabled": bool(self.config["s3_loc"].get("pos_neg_contrast_enabled", True)),
                "pos_neg_contrast_lambda": float(self.config["s3_loc"].get("pos_neg_contrast_lambda", 0.9)),
                "pos_neg_contrast_blend": float(self.config["s3_loc"].get("pos_neg_contrast_blend", 0.20)),
                "pos_neg_contrast_min_positive_groups": int(
                    self.config["s3_loc"].get("pos_neg_contrast_min_positive_groups", 3)
                ),
                "pos_neg_contrast_min_negative_groups": int(
                    self.config["s3_loc"].get("pos_neg_contrast_min_negative_groups", 3)
                ),
            },
        )
        localization_t3 = time.perf_counter()
        localization_runtime_ms_historical = float((localization_t1 - localization_t0) * 1000.0)
        localization_runtime_ms_current = float((localization_t2 - localization_t1) * 1000.0)
        localization_runtime_ms_wrapper = float((localization_t3 - localization_t2) * 1000.0)
        localization_runtime_ms_total = float((localization_t3 - localization_t0) * 1000.0)
        localization.client_scores = stabilized_current_scores
        localization.flagged_clients = current_suspicious_set
        localization.aux_stats["p_hat"] = p_hat
        localization.aux_stats["degenerate_mode"] = degenerate
        localization.aux_stats["selected_scores"] = {
            int(client_id): float(stabilized_current_scores.get(client_id, 0.0))
            for client_id in current_suspicious_set
        }
        localization.aux_stats["raw_selected_scores"] = current_s3_aux.get("selected_scores", {})
        localization.aux_stats["historical_client_scores"] = client_scores
        localization.aux_stats["historical_selected_scores"] = s3_aux.get("selected_scores", {})
        localization.aux_stats["current_p_hat"] = current_p_hat
        localization.aux_stats["current_degenerate_mode"] = current_degenerate
        localization.aux_stats["num_groups_total"] = int(self.A_matrix.shape[0])
        localization.aux_stats["num_groups_current"] = int(current_A.shape[0])
        localization.aux_stats["current_group_sizes"] = [int(len(group)) for group in ordered_group_assignments]
        localization.aux_stats["stability_stats"] = stability_stats
        localization.aux_stats["runtime_ms_historical"] = localization_runtime_ms_historical
        localization.aux_stats["runtime_ms_current"] = localization_runtime_ms_current
        localization.aux_stats["runtime_ms_wrapper"] = localization_runtime_ms_wrapper
        localization.aux_stats["runtime_ms_total"] = localization_runtime_ms_total

        self.current_suspicious_set = list(current_suspicious_set)
        p_hat_raw = float(p_hat)
        current_p_hat_raw = float(current_p_hat)
        p_hat_corrected, p_hat_correction_factor_used = self._correct_p_hat(
            p_hat_raw,
            selected_clients_count=len(self.selected_clients),
            group_sizes=[int(len(group)) for group in ordered_group_assignments],
        )
        current_p_hat_corrected, current_p_hat_correction_factor_used = self._correct_p_hat(
            current_p_hat_raw,
            selected_clients_count=len(self.selected_clients),
            group_sizes=[int(len(group)) for group in ordered_group_assignments],
        )
        p_hat = float(p_hat_corrected)
        current_p_hat = float(current_p_hat_corrected)
        p_hat_predicted = float(p_hat)
        p_hat_oracle_value = metadata.get("oracle_p_hat")
        oracle_override_active = bool(p_hat_oracle_value is not None)

        top_client_scores = sorted(
            ((int(client_id), float(score)) for client_id, score in stabilized_current_scores.items()),
            key=lambda item: (-item[1], item[0]),
        )[: max(10, int(self.config["suspicious_topk"]))]
        consistency_map = detection.aux_stats.get("consistency", {})
        anomalous_groups = [
            int(group_id) for group_id in ordered_group_ids if bool(_map_get_bool_local(b_g_map, int(group_id), False))
        ]
        positive_ratio = float(sum(self.current_b_g) / max(len(self.current_b_g), 1))
        cts_scores = [float(score) for score in self.current_cts_scores]
        cts_diff_map_raw = detection.aux_stats.get("diff_group_scores", {})
        oriented_diff_scores: List[float] = []
        dcbd_reliability_use_absolute_score = bool(
            self.config.get("dcbd_reliability_use_absolute_score", True)
        )
        for group_id in ordered_group_ids:
            gid = int(group_id)
            if dcbd_absolute_signal_priority and dcbd_reliability_use_absolute_score:
                oriented_diff_scores.append(float(max(0.0, _group_dcbd_score(gid))))
            else:
                diff_val = float(
                    cts_diff_map_raw.get(gid, cts_diff_map_raw.get(str(gid), 0.0))
                )
                oriented_diff_scores.append(float(_group_oriented_diff(gid, diff_val)))
        max_cts_score = max(oriented_diff_scores) if len(oriented_diff_scores) > 0 else 0.0
        max_cts_score = max(max_cts_score, 0.0)
        max_abs_cts_score = max((abs(float(score)) for score in cts_scores), default=0.0)
        mean_consistency = float(np.mean(list(consistency_map.values()))) if len(consistency_map) > 0 else 0.0
        mad_median = float(detection.aux_stats.get("mad_median", 0.0))
        mad_sigma = float(detection.aux_stats.get("mad_sigma", 0.0))
        mad_sigma_floor = float(max(self.config.get("cleaning_mad_sigma_floor", 0.01), 1e-6))
        mad_sigma_effective = float(max(mad_sigma, mad_sigma_floor))
        cts_z_clip = float(max(self.config.get("cleaning_cts_z_clip", 12.0), 0.0))
        cts_z_raw = float(max_cts_score / mad_sigma_effective)
        cts_z_score = float(min(cts_z_raw, cts_z_clip)) if cts_z_clip > 0.0 else float(cts_z_raw)
        group_cts_z_values = [
            float(
                min(
                    max(0.0, float(score)) / mad_sigma_effective,
                    cts_z_clip if cts_z_clip > 0.0 else max(0.0, float(score)) / mad_sigma_effective,
                )
            )
            for score in oriented_diff_scores
        ]
        cts_z_mean = float(np.mean(group_cts_z_values)) if len(group_cts_z_values) > 0 else 0.0
        cts_z_std = float(np.std(group_cts_z_values)) if len(group_cts_z_values) > 0 else 0.0
        cts_z_min = float(np.min(group_cts_z_values)) if len(group_cts_z_values) > 0 else 0.0
        cts_z_max = float(np.max(group_cts_z_values)) if len(group_cts_z_values) > 0 else 0.0
        cts_distribution_collapsed = False
        cts_distribution_collapse_reason = "none"
        if mad_sigma <= 1e-6:
            cts_distribution_collapsed = True
            cts_distribution_collapse_reason = "mad_sigma_near_zero"
        elif len(cts_scores) > 0 and float(np.std(np.asarray(cts_scores, dtype=np.float32))) <= 1e-4:
            cts_distribution_collapsed = True
            cts_distribution_collapse_reason = "cts_raw_std_near_zero"
        elif abs(float(detection.aux_stats.get("mad_threshold", 0.0))) <= 1e-6:
            cts_distribution_collapsed = True
            cts_distribution_collapse_reason = "mad_threshold_near_zero"
        dense_mode_active, dense_mode_p_hat, dense_mode_stats = self._estimate_dense_regime_p_hat(
            cts_scores=cts_scores,
            positive_ratio=positive_ratio,
            cts_z_score=cts_z_score,
            mean_consistency=mean_consistency,
            consensus_count=len(self.current_consensus_candidates),
            group_sizes=[int(len(group)) for group in ordered_group_assignments],
        )
        if dense_mode_active:
            p_hat = max(float(p_hat), float(dense_mode_p_hat))
            current_p_hat = max(float(current_p_hat), float(dense_mode_p_hat))
            if float(p_hat) >= float(self.config.get("degenerate_p_hat_floor", self.degenerate_p_hat_floor)):
                degenerate = True
            if float(current_p_hat) >= float(self.config.get("degenerate_p_hat_floor", self.degenerate_p_hat_floor)):
                current_degenerate = True

        if p_hat_oracle_value is not None:
            oracle_p_hat = float(min(1.0, max(0.0, p_hat_oracle_value)))
            p_hat = oracle_p_hat
            current_p_hat = oracle_p_hat
            degenerate = bool(oracle_p_hat > float(self.config["s3_loc"].get("degenerate_threshold", 0.8)))
            current_degenerate = bool(oracle_p_hat > float(self.config["s3_loc"].get("degenerate_threshold", 0.8)))
        else:
            oracle_p_hat = None

        self.current_p_hat = float(p_hat)
        self.current_degenerate = bool(degenerate)
        self.current_dense_mode_active = bool(dense_mode_active)
        self.current_dense_p_hat = float(dense_mode_p_hat if dense_mode_active else 0.0)
        cts_consistency_thr_cfg = float(self.config.get("cleaning_consistency_threshold", 0.90))
        cts_score_thr_cfg = float(self.config.get("cleaning_cts_score_threshold", 0.75))
        cts_z_thr_cfg = float(self.config.get("cleaning_cts_z_threshold", 2.5))
        cts_consistency_thr_floor = float(
            np.clip(self.config.get("cleaning_consistency_threshold_floor", 0.52), 0.0, 1.0)
        )
        cts_score_thr_floor = float(max(self.config.get("cleaning_cts_score_threshold_floor", 0.06), 0.0))
        cts_z_thr_floor = float(max(self.config.get("cleaning_cts_z_threshold_floor", 0.80), 0.0))
        cts_consistency_thr_quantile = float(
            np.clip(self.config.get("cleaning_consistency_threshold_quantile", 0.65), 0.0, 1.0)
        )
        cts_score_thr_quantile = float(
            np.clip(self.config.get("cleaning_cts_score_threshold_quantile", 0.85), 0.0, 1.0)
        )
        cts_z_thr_quantile = float(
            np.clip(self.config.get("cleaning_cts_z_threshold_quantile", 0.85), 0.0, 1.0)
        )
        consistency_values = [
            float(value) for value in consistency_map.values()
            if np.isfinite(float(value))
        ]
        cts_oriented_scores = [
            max(0.0, float(score))
            for score in oriented_diff_scores
            if np.isfinite(float(score))
        ]
        cts_z_values_for_thr = [float(abs(z)) for z in group_cts_z_values if np.isfinite(float(z))]
        cts_consistency_thr_adaptive = float(
            np.quantile(np.asarray(consistency_values, dtype=np.float32), cts_consistency_thr_quantile)
        ) if len(consistency_values) > 0 else cts_consistency_thr_cfg
        cts_score_thr_adaptive = float(
            np.quantile(np.asarray(cts_oriented_scores, dtype=np.float32), cts_score_thr_quantile)
        ) if len(cts_oriented_scores) > 0 else cts_score_thr_cfg
        cts_z_thr_adaptive = float(
            np.quantile(np.asarray(cts_z_values_for_thr, dtype=np.float32), cts_z_thr_quantile)
        ) if len(cts_z_values_for_thr) > 0 else cts_z_thr_cfg
        cts_consistency_thr = float(
            max(
                cts_consistency_thr_floor,
                min(cts_consistency_thr_cfg, cts_consistency_thr_adaptive),
            )
        )
        cts_score_thr = float(
            max(
                cts_score_thr_floor,
                min(cts_score_thr_cfg, cts_score_thr_adaptive),
            )
        )
        cts_z_thr = float(
            max(
                cts_z_thr_floor,
                min(cts_z_thr_cfg, cts_z_thr_adaptive),
            )
        )
        reliability_relaxed_cons_floor = float(
            np.clip(
                self.config.get("cleaning_reliability_relaxed_consistency_floor", 0.32),
                0.0,
                1.0,
            )
        )
        require_two_signals = bool(self.config.get("cleaning_reliability_require_two_signals", True))
        cts_score_norm_threshold = float(max(self.config.get("cleaning_cts_score_norm_threshold", 1.2), 0.0))
        cts_score_norm = float(max_cts_score / max(mad_sigma_effective, 1e-6))
        score_ok = bool(
            max_cts_score >= cts_score_thr
            or cts_score_norm >= cts_score_norm_threshold
        )
        z_ok = bool(cts_z_score >= cts_z_thr)
        consistency_ok = False
        cts_signal_reliable_rule_path = "none"
        cts_signal_reliable = False
        cts_signal_unreliable_reason = "pending_candidate_consistency"
        cleaning_p_hat = float(p_hat)
        self.current_cleaning_confidence = float(min(1.0, max(0.0, cts_z_score / max(float(self.config.get("cleaning_cts_z_threshold", 3.0)), 1e-6))))
        fallback_triggered = bool(detection.aux_stats.get("fallback_triggered", False))
        fallback_selected_groups_raw = detection.aux_stats.get("fallback_selected_groups", [])
        fallback_selected_groups = [
            int(group_id)
            for group_id in fallback_selected_groups_raw
            if int(group_id) in set(int(gid) for gid in ordered_group_ids)
        ]
        group_signal_map = {
            int(group_id): float(signal)
            for group_id, signal in zip(ordered_group_ids, self.current_localization_signal_scores)
        }
        group_cts_raw_map = {
            int(group_id): float(score)
            for group_id, score in zip(ordered_group_ids, self.current_cts_scores)
        }
        candidate_min_consistency = float(
            np.clip(self.config.get("candidate_min_consistency", 0.45), 0.0, 1.0)
        )
        candidate_min_consistency_abs = float(
            np.clip(self.config.get("candidate_min_consistency_abs", 0.40), 0.0, 1.0)
        )
        candidate_min_oriented_diff = float(
            max(self.config.get("candidate_min_oriented_diff", 0.003), 0.0)
        )
        candidate_min_oriented_signal = float(
            max(self.config.get("candidate_min_oriented_signal", 0.05), 0.0)
        )
        candidate_use_two_sided_signal = bool(
            self.config.get("candidate_use_two_sided_signal", True)
        )
        candidate_use_two_sided_signal_effective = bool(
            candidate_use_two_sided_signal and (not tail_alignment_required)
        )
        candidate_allow_abs_fallback = bool(
            self.config.get("candidate_allow_abs_fallback", True)
        )
        candidate_min_abs_diff = float(
            max(self.config.get("candidate_min_abs_diff", 0.002), 0.0)
        )
        candidate_min_abs_signal = float(
            max(self.config.get("candidate_min_abs_signal", 0.20), 0.0)
        )
        dcbd_candidate_min_tsc = float(self.config.get("dcbd_candidate_min_tsc", 0.0))
        dcbd_candidate_min_consistency = float(
            np.clip(self.config.get("dcbd_candidate_min_consistency", 0.34), 0.0, 1.0)
        )
        dcbd_candidate_min_dcbd_score = float(
            self.config.get("dcbd_candidate_min_dcbd_score", 0.0)
        )
        dcbd_candidate_min_dcbd_z = float(
            max(self.config.get("dcbd_candidate_min_dcbd_z", 1.0), 0.0)
        )
        dcbd_candidate_rank_tsc_weight = float(
            max(self.config.get("dcbd_candidate_rank_tsc_weight", 1.0), 0.0)
        )
        dcbd_candidate_rank_dcbd_z_weight = float(
            max(self.config.get("dcbd_candidate_rank_dcbd_z_weight", 0.25), 0.0)
        )
        candidate_allow_current_when_history_blocked = bool(
            self.config.get("candidate_allow_current_when_history_blocked", True)
        )
        candidate_history_block_min_cts_z = float(
            max(self.config.get("candidate_history_block_min_cts_z", 0.80), 0.0)
        )
        candidate_history_block_min_anomalous_groups = int(
            max(self.config.get("candidate_history_block_min_anomalous_groups", 1), 0)
        )
        candidate_consistency_topk_for_reliability = int(
            max(self.config.get("candidate_consistency_topk_for_reliability", 3), 1)
        )
        reliability_consistency_source_pref = str(
            self.config.get("cleaning_reliability_consistency_source", "candidate_selected")
        ).strip().lower()

        def _candidate_effective_signal(gid: int) -> float:
            if tail_alignment_required:
                directional_signal = float(sign * group_signal_map.get(int(gid), 0.0))
            else:
                direction = _group_direction_sign(int(gid))
                directional_signal = float(direction * group_signal_map.get(int(gid), 0.0))
            if candidate_use_two_sided_signal_effective:
                two_sided_signal = float(_group_two_sided_strength(int(gid)))
                return float(max(0.0, max(directional_signal, two_sided_signal)))
            return float(max(0.0, directional_signal))

        def _is_directional_candidate(gid: int) -> bool:
            diff_val = float(
                cts_diff_map_raw.get(gid, cts_diff_map_raw.get(str(gid), 0.0))
            )
            oriented_diff = float(_group_oriented_diff(int(gid), diff_val))
            abs_diff = float(abs(diff_val))
            oriented_signal = float(_candidate_effective_signal(int(gid)))
            cons_val = float(consistency_map.get(gid, consistency_map.get(str(gid), 0.0)))
            if dcbd_absolute_signal_priority:
                tsc_val = float(_group_tsc_target(int(gid)))
                dcbd_score_val = float(_group_dcbd_score(int(gid)))
                dcbd_z_val = float(_group_dcbd_anomaly_z(int(gid)))
                tsc_gate_ok = bool(
                    (not dcbd_tsc_hard_gate_apply_candidate) or _group_tsc_gate_pass(int(gid))
                )
                primary_ok = bool(
                    tsc_val >= dcbd_candidate_min_tsc
                    and oriented_signal >= candidate_min_oriented_signal
                    and cons_val >= max(candidate_min_consistency, dcbd_candidate_min_consistency)
                )
                support_ok = bool(
                    (
                        _group_dcbd_anomaly(int(gid))
                        or (
                            dcbd_score_val >= dcbd_candidate_min_dcbd_score
                            and dcbd_z_val >= dcbd_candidate_min_dcbd_z
                        )
                    )
                    and oriented_signal >= candidate_min_abs_signal
                    and cons_val >= min(candidate_min_consistency_abs, dcbd_candidate_min_consistency)
                )
                return bool(tsc_gate_ok and (primary_ok or support_ok))
            primary_ok = bool(
                oriented_diff >= candidate_min_oriented_diff
                and oriented_signal >= candidate_min_oriented_signal
                and cons_val >= candidate_min_consistency
            )
            abs_bridge_ok = bool(
                candidate_allow_abs_fallback
                and (not tail_alignment_required)
                and abs_diff >= candidate_min_abs_diff
                and oriented_signal >= candidate_min_abs_signal
                and cons_val >= candidate_min_consistency_abs
            )
            return bool(primary_ok or abs_bridge_ok)

        def _candidate_oriented_signal(gid: int) -> float:
            return float(_candidate_effective_signal(int(gid)))

        def _candidate_consistency(gid: int) -> float:
            return float(consistency_map.get(int(gid), consistency_map.get(str(int(gid)), 0.0)))

        def _candidate_rank_score(gid: int) -> float:
            oriented_signal = _candidate_oriented_signal(int(gid))
            rank_cons_floor = (
                min(candidate_min_consistency, candidate_min_consistency_abs)
                if candidate_allow_abs_fallback
                else candidate_min_consistency
            )
            cons_margin = float(max(0.0, _candidate_consistency(int(gid)) - rank_cons_floor))
            if dcbd_absolute_signal_priority:
                tsc_pos = float(max(0.0, _group_tsc_target(int(gid))))
                dcbd_z_pos = float(max(0.0, _group_dcbd_anomaly_z(int(gid))))
                enriched_signal = float(
                    oriented_signal
                    + dcbd_candidate_rank_tsc_weight * tsc_pos
                    + dcbd_candidate_rank_dcbd_z_weight * dcbd_z_pos
                )
                return float(enriched_signal * cons_margin)
            return float(oriented_signal * cons_margin)

        candidate_source = "none"
        candidate_from_hist_keep_bridge = False
        candidate_history_soft_override_applied = False
        hist_keep_set = {int(group_id) for group_id in hist_kept_group_ids}
        directional_anomalous_groups = [
            int(group_id)
            for group_id in anomalous_groups
            if _is_directional_candidate(int(group_id))
        ]
        require_hist_keep_for_candidate = bool(self.config.get("candidate_require_hist_keep", True))
        history_gate_allows_candidate = bool(history_accumulated or (not require_hist_keep_for_candidate))
        if not history_gate_allows_candidate:
            history_block_soft_override = bool(
                candidate_allow_current_when_history_blocked
                and len(directional_anomalous_groups) > 0
                and len(anomalous_groups) >= candidate_history_block_min_anomalous_groups
                and cts_z_score >= candidate_history_block_min_cts_z
            )
            if history_block_soft_override:
                candidate_group_ids = [int(group_id) for group_id in directional_anomalous_groups]
                candidate_source = "history_gate_soft_override"
                candidate_history_soft_override_applied = True
            else:
                candidate_group_ids = []
                candidate_source = "history_gate_blocked"
        elif len(anomalous_groups) > 0:
            if len(hist_keep_set) > 0:
                candidate_group_ids = [
                    int(group_id)
                    for group_id in directional_anomalous_groups
                    if int(group_id) in hist_keep_set
                ]
                candidate_source = "anomalous_groups_hist_keep_filtered"
            else:
                candidate_group_ids = [int(group_id) for group_id in directional_anomalous_groups]
                candidate_source = "anomalous_groups_directional"
        elif len(fallback_selected_groups) > 0:
            candidate_group_ids = [
                int(group_id)
                for group_id in fallback_selected_groups
                if _is_directional_candidate(int(group_id))
            ]
            candidate_source = "fallback_selected_groups"
        else:
            candidate_group_ids = []
        candidate_group_scores = {}
        candidate_group_rank_scores = {}
        for group_id in candidate_group_ids:
            gid = int(group_id)
            candidate_group_scores[gid] = float(_candidate_oriented_signal(gid))
            candidate_group_rank_scores[gid] = float(_candidate_rank_score(gid))
        if len(candidate_group_ids) == 0 and len(hist_kept_group_ids) > 0:
            # Bridge history-kept groups into candidate space when CTS anomaly flags
            # are sparse but history-gate accepted this audit.
            ranked_hist_groups = sorted(
                [int(group_id) for group_id in hist_kept_group_ids],
                key=lambda group_id: (
                    float(_candidate_rank_score(int(group_id))),
                    float(_candidate_oriented_signal(int(group_id))),
                    -int(group_id),
                ),
                reverse=True,
            )
            directional_hist_groups = [
                int(group_id)
                for group_id in ranked_hist_groups
                if _is_directional_candidate(int(group_id))
            ]
            if len(directional_hist_groups) > 0:
                candidate_group_ids = directional_hist_groups[: max(1, min(3, len(directional_hist_groups)))]
            candidate_group_scores = {}
            candidate_group_rank_scores = {}
            for group_id in candidate_group_ids:
                gid = int(group_id)
                candidate_group_scores[gid] = float(_candidate_oriented_signal(gid))
                candidate_group_rank_scores[gid] = float(_candidate_rank_score(gid))
            if len(candidate_group_ids) > 0:
                candidate_source = "history_kept_bridge"
                candidate_from_hist_keep_bridge = True
        candidate_selected_group_id = -1
        candidate_selected_score = 0.0
        candidate_selected_rank_score = 0.0
        if len(candidate_group_scores) > 0:
            selected_group_id = max(
                candidate_group_scores.keys(),
                key=lambda gid: (
                    float(candidate_group_rank_scores.get(int(gid), 0.0)),
                    float(candidate_group_scores.get(int(gid), 0.0)),
                    -int(gid),
                ),
            )
            candidate_selected_group_id = int(selected_group_id)
            candidate_selected_score = float(candidate_group_scores.get(candidate_selected_group_id, 0.0))
            candidate_selected_rank_score = float(
                candidate_group_rank_scores.get(candidate_selected_group_id, 0.0)
            )
        cts_diff_value = 0.0
        cts_raw_value = 0.0
        cts_tsc_value = 0.0
        cts_dcbd_score_value = 0.0
        cts_dcbd_z_value = 0.0
        if candidate_selected_group_id >= 0:
            cts_diff_value = float(
                cts_diff_map_raw.get(
                    int(candidate_selected_group_id),
                    cts_diff_map_raw.get(str(int(candidate_selected_group_id)), 0.0),
                )
            )
            cts_raw_value = float(group_cts_raw_map.get(int(candidate_selected_group_id), 0.0))
            cts_tsc_value = float(_group_tsc_target(int(candidate_selected_group_id)))
            cts_dcbd_score_value = float(_group_dcbd_score(int(candidate_selected_group_id)))
            cts_dcbd_z_value = float(_group_dcbd_anomaly_z(int(candidate_selected_group_id)))
        candidate_selected_consistency = (
            float(_candidate_consistency(candidate_selected_group_id))
            if candidate_selected_group_id >= 0
            else None
        )
        ranked_candidate_ids = sorted(
            [int(group_id) for group_id in candidate_group_ids],
            key=lambda gid: (
                float(candidate_group_rank_scores.get(int(gid), 0.0)),
                float(candidate_group_scores.get(int(gid), 0.0)),
                -int(gid),
            ),
            reverse=True,
        )
        candidate_topk_ids_for_reliability = ranked_candidate_ids[:candidate_consistency_topk_for_reliability]
        candidate_topk_consistency_values = [
            float(_candidate_consistency(int(group_id)))
            for group_id in candidate_topk_ids_for_reliability
        ]
        candidate_topk_consistency_mean = (
            float(np.mean(np.asarray(candidate_topk_consistency_values, dtype=np.float32)))
            if len(candidate_topk_consistency_values) > 0
            else None
        )
        reliability_consistency_value = float(mean_consistency)
        reliability_consistency_source = "global_mean"
        if reliability_consistency_source_pref in {"candidate_selected", "selected"}:
            if candidate_selected_consistency is not None:
                reliability_consistency_value = float(candidate_selected_consistency)
                reliability_consistency_source = "candidate_selected"
            elif candidate_topk_consistency_mean is not None:
                reliability_consistency_value = float(candidate_topk_consistency_mean)
                reliability_consistency_source = "candidate_topk"
        elif reliability_consistency_source_pref in {"candidate_topk", "topk"}:
            if candidate_topk_consistency_mean is not None:
                reliability_consistency_value = float(candidate_topk_consistency_mean)
                reliability_consistency_source = "candidate_topk"
            elif candidate_selected_consistency is not None:
                reliability_consistency_value = float(candidate_selected_consistency)
                reliability_consistency_source = "candidate_selected"
        elif reliability_consistency_source_pref in {"global_mean", "mean"}:
            reliability_consistency_value = float(mean_consistency)
            reliability_consistency_source = "global_mean"
        elif candidate_selected_consistency is not None:
            reliability_consistency_value = float(candidate_selected_consistency)
            reliability_consistency_source = "candidate_selected"
        elif candidate_topk_consistency_mean is not None:
            reliability_consistency_value = float(candidate_topk_consistency_mean)
            reliability_consistency_source = "candidate_topk"

        consistency_ok = bool(reliability_consistency_value >= cts_consistency_thr)
        cts_signal_reliable_rule_path = "none"
        cts_signal_reliable = False
        if consistency_ok and score_ok and z_ok:
            cts_signal_reliable = True
            cts_signal_reliable_rule_path = "strict_all"
        elif consistency_ok and (score_ok or z_ok):
            cts_signal_reliable = True
            cts_signal_reliable_rule_path = "consistency_plus_one_signal"
        elif (
            require_two_signals
            and score_ok
            and z_ok
            and reliability_consistency_value >= reliability_relaxed_cons_floor
        ):
            cts_signal_reliable = True
            cts_signal_reliable_rule_path = "two_signal_relaxed_consistency"
        if cts_distribution_collapsed:
            cts_signal_reliable = False
            cts_signal_reliable_rule_path = "blocked_distribution_collapsed"
        cts_signal_bridge_enable = bool(self.config.get("cts_signal_bridge_enable", True))
        cts_signal_bridge_min_candidate_count = int(
            max(self.config.get("cts_signal_bridge_min_candidate_count", 1), 1)
        )
        cts_signal_bridge_min_consistency = float(
            np.clip(self.config.get("cts_signal_bridge_min_consistency", 0.42), 0.0, 1.0)
        )
        cts_signal_bridge_min_score = float(
            max(self.config.get("cts_signal_bridge_min_score", 0.20), 0.0)
        )
        cts_signal_bridge_min_cts_z = float(
            max(self.config.get("cts_signal_bridge_min_cts_z", 0.80), 0.0)
        )
        cts_signal_bridge_min_rank_score = float(
            max(self.config.get("cts_signal_bridge_min_rank_score", 0.02), 0.0)
        )
        if (
            (not cts_signal_reliable)
            and (not cts_distribution_collapsed)
            and cts_signal_bridge_enable
        ):
            bridge_consistency = (
                float(candidate_selected_consistency)
                if candidate_selected_consistency is not None
                else float(
                    candidate_topk_consistency_mean
                    if candidate_topk_consistency_mean is not None
                    else 0.0
                )
            )
            bridge_signal_ok = bool(
                candidate_selected_score >= cts_signal_bridge_min_score
                and candidate_selected_rank_score >= cts_signal_bridge_min_rank_score
            )
            bridge_consistency_ok = bool(
                bridge_consistency >= cts_signal_bridge_min_consistency
            )
            bridge_cts_ok = bool(
                cts_z_score >= cts_signal_bridge_min_cts_z
                or (score_ok and z_ok)
            )
            if (
                len(candidate_group_ids) >= cts_signal_bridge_min_candidate_count
                and bridge_signal_ok
                and bridge_consistency_ok
                and bridge_cts_ok
            ):
                cts_signal_reliable = True
                cts_signal_reliable_rule_path = "candidate_signal_bridge"
        cts_signal_unreliable_reasons: List[str] = []
        if not consistency_ok:
            cts_signal_unreliable_reasons.append("consistency_below_threshold")
        if not score_ok:
            cts_signal_unreliable_reasons.append("max_cts_score_below_threshold")
        if not z_ok:
            cts_signal_unreliable_reasons.append("cts_z_below_threshold")
        if cts_distribution_collapsed:
            cts_signal_unreliable_reasons.append("cts_distribution_collapsed")
        cts_signal_unreliable_reason = (
            "none"
            if cts_signal_reliable
            else ("|".join(cts_signal_unreliable_reasons) if len(cts_signal_unreliable_reasons) > 0 else "unknown")
        )

        cleaning_p_hat = float(p_hat)
        if cts_signal_reliable and not bool(detection.aux_stats.get("fallback_triggered", False)):
            cleaning_p_hat = max(cleaning_p_hat, float(self.config.get("cleaning_min_p_hat", 0.06)))
            if len(anomalous_groups) > 0:
                cleaning_p_hat = max(cleaning_p_hat, float(self.config.get("cleaning_min_p_hat_when_anomalous", 0.10)))
        elif not cts_signal_reliable:
            cleaning_p_hat = min(cleaning_p_hat, float(self.config.get("cleaning_max_p_hat_with_weak_signal", 0.05)))
        if dense_mode_active and bool(self.config.get("dense_mode_cleaning_relax", True)):
            cleaning_p_hat = max(
                cleaning_p_hat,
                float(self.config.get("dense_mode_cleaning_min_p_hat", 0.03)),
            )
            self.current_cleaning_confidence = max(
                self.current_cleaning_confidence,
                float(self.config.get("dense_mode_cleaning_min_confidence", 0.25)),
            )
        hist_keep_candidate_drop_reason = "none"
        if len(hist_kept_group_ids) > 0 and len(candidate_group_ids) <= 0:
            hist_keep_candidate_drop_reason = "no_candidate_after_hist_filter"

        raw_score_map = {
            int(client_id): max(float(score), 0.0)
            for client_id, score in current_client_scores.items()
        }
        raw_top_score = float(raw_order[0][1]) if len(raw_order) > 0 else 0.0
        raw_second_score = float(raw_order[1][1]) if len(raw_order) > 1 else 0.0
        raw_score_gap = raw_top_score - raw_second_score
        raw_score_ratio = raw_top_score / max(raw_second_score, 1e-6) if raw_top_score > 0.0 else 0.0

        stability_gate_enabled = bool(self.config.get("stability_enable_dynamic_gate", True))
        stability_gate_weak_scale = min(max(float(self.config.get("stability_gate_weak_scale", 0.10)), 0.0), 1.0)
        stability_gate_disable_when_no_anomaly = bool(
            self.config.get("stability_gate_disable_when_no_anomaly", True)
        )
        stability_gate_min_anomalous_groups = int(self.config.get("stability_gate_min_anomalous_groups", 1))
        stability_gate_min_positive_ratio = float(self.config.get("stability_gate_min_positive_ratio", 0.02))
        stability_gate_min_current_max = float(self.config.get("stability_gate_min_current_max", 0.20))
        stability_gate_min_score_gap = float(self.config.get("stability_gate_min_score_gap", 0.05))
        stability_gate_min_score_ratio = float(self.config.get("stability_gate_min_score_ratio", 1.06))
        stability_gate_min_cts_z = float(self.config.get("stability_gate_min_cts_z", 2.20))
        stability_gate_min_consistency = float(self.config.get("stability_gate_min_consistency", 0.55))

        stability_evidence_by_groups = bool(
            len(anomalous_groups) >= stability_gate_min_anomalous_groups
            and positive_ratio >= stability_gate_min_positive_ratio
            and raw_top_score >= stability_gate_min_current_max
            and (
                raw_score_gap >= stability_gate_min_score_gap
                or raw_score_ratio >= stability_gate_min_score_ratio
            )
        )
        stability_evidence_by_cts = bool(
            cts_z_score >= stability_gate_min_cts_z
            and mean_consistency >= stability_gate_min_consistency
            and not fallback_triggered
        )
        stability_evidence_ok = bool(stability_evidence_by_groups or stability_evidence_by_cts)
        if (
            stability_gate_disable_when_no_anomaly
            and len(anomalous_groups) <= 0
            and not stability_evidence_by_cts
        ):
            stability_evidence_ok = False

        stability_gate_applied = bool(
            stability_gate_enabled
            and not stability_evidence_ok
            and stability_gate_weak_scale < 1.0
        )
        if stability_gate_applied:
            for client_id, raw_score in raw_score_map.items():
                stabilized_score = float(stabilized_current_scores.get(client_id, raw_score))
                stabilized_current_scores[client_id] = float(
                    raw_score + (stabilized_score - raw_score) * stability_gate_weak_scale
                )
            stabilized_order = sorted(
                (
                    (int(client_id), float(score))
                    for client_id, score in stabilized_current_scores.items()
                ),
                key=lambda item: (-item[1], item[0]),
            )
            current_suspicious_set = [
                client_id
                for client_id, _ in stabilized_order[: int(self.config["suspicious_topk"])]
            ]
            localization.client_scores = stabilized_current_scores
            localization.flagged_clients = current_suspicious_set
            localization.aux_stats["selected_scores"] = {
                int(client_id): float(stabilized_current_scores.get(client_id, 0.0))
                for client_id in current_suspicious_set
            }
            self.current_suspicious_set = list(current_suspicious_set)
            if len(stabilized_current_scores) > 0:
                top_client_id, top_client_score = max(
                    stabilized_current_scores.items(),
                    key=lambda item: (item[1], -item[0]),
                )
                total_weight = float(self._historical_score_weight.get(int(top_client_id), 0.0))
                stability_stats.update({
                    "current_max": float(max(float(top_client_score), 0.0)),
                    "top_frequency": int(self._historical_top_count.get(int(top_client_id), 0)),
                    "top_persistence": (
                        float(self._historical_top_count.get(int(top_client_id), 0))
                        / max(float(self._historical_anomalous_audits), 1.0)
                        if self._historical_anomalous_audits > 0 else 0.0
                    ),
                    "top_historical_mean": (
                        float(self._historical_score_sum.get(int(top_client_id), 0.0)) / total_weight
                        if total_weight > 0.0 else 0.0
                    ),
                })
        stability_stats.update({
            "stability_gate_enabled": bool(stability_gate_enabled),
            "stability_gate_applied": bool(stability_gate_applied),
            "stability_gate_weak_scale": float(stability_gate_weak_scale),
            "stability_evidence_by_groups": bool(stability_evidence_by_groups),
            "stability_evidence_by_cts": bool(stability_evidence_by_cts),
            "stability_evidence_ok": bool(stability_evidence_ok),
            "raw_top_score": float(raw_top_score),
            "raw_score_gap": float(raw_score_gap),
            "raw_score_ratio": float(raw_score_ratio),
            "raw_positive_ratio": float(positive_ratio),
            "raw_anomalous_group_count": int(len(raw_anomalous_groups)),
        })

        top_client_scores = sorted(
            ((int(client_id), float(score)) for client_id, score in stabilized_current_scores.items()),
            key=lambda item: (-item[1], item[0]),
        )[: max(10, int(self.config["suspicious_topk"]))]
        top_score = float(top_client_scores[0][1]) if len(top_client_scores) > 0 else 0.0
        second_score = float(top_client_scores[1][1]) if len(top_client_scores) > 1 else 0.0
        score_gap = top_score - second_score
        score_ratio = top_score / max(second_score, 1e-6) if top_score > 0.0 else 0.0
        consensus_candidates, consensus_scores = self._get_consensus_candidates(
            current_client_scores=current_client_scores,
        )
        self.current_consensus_candidates = list(consensus_candidates)
        consensus_candidate_count = int(len(consensus_candidates))
        history_quality_gate_enabled = bool(
            self.config.get("consensus_history_quality_gate_enabled", True)
        )

        history_quality_ok = bool(
            (not history_quality_gate_enabled)
            or (
               
                len(candidate_group_ids)
                >= int(self.config.get("consensus_history_quality_min_candidate_count", 2))
                
                and float(candidate_selected_consistency or 0.0)
                >= float(self.config.get("consensus_history_quality_min_selected_consistency", 0.64))
                
                and float(candidate_selected_rank_score)
                >= float(self.config.get("consensus_history_quality_min_rank_score", 0.50))
                
                and consensus_candidate_count
                >= int(self.config.get("consensus_history_quality_min_consensus_count", 0))
                
            
            )
        )
        reweight_quality_gate_enabled = bool(
            self.config.get("reweight_quality_gate_enabled", True)
        )
        
        reweight_quality_ok = bool(
            (not reweight_quality_gate_enabled)
            or (
                
                len(candidate_group_ids)
                >= int(self.config.get("reweight_quality_min_candidate_count", 2))
                
                and float(candidate_selected_consistency or 0.0)
                >= float(self.config.get("reweight_quality_min_selected_consistency", 0.64))
               
                and float(candidate_selected_rank_score)
                >= float(self.config.get("reweight_quality_min_rank_score", 0.50))
                
                and consensus_candidate_count
                >= int(self.config.get("reweight_quality_min_consensus_count", 0))
                
            
            )
        )
        refined_reliable_scores = self._refine_reliable_scores(
            current_client_scores=current_client_scores,
            consensus_scores=consensus_scores,
        )
        reliable_pool_candidate_topk = max(1, int(self.config.get("reliable_pool_candidate_topk", 5)))
        reliable_pool_consensus_topk = max(1, int(self.config.get("reliable_pool_consensus_topk", 3)))
        reweight_topk = max(1, int(self.config.get("reweight_topk", self.config.get("suspicious_topk", 1))))
        current_top_for_pool = {
            int(client_id)
            for client_id in current_suspicious_set[:reliable_pool_candidate_topk]
        }
        consensus_top_for_pool = {
            int(client_id)
            for client_id in consensus_candidates[:reliable_pool_consensus_topk]
        }
        reliable_consensus_overlap = int(len(current_top_for_pool & consensus_top_for_pool))
        top_frequency = int(stability_stats.get("top_frequency", 0))
        top_persistence = float(stability_stats.get("top_persistence", 0.0))
        history_bootstrap_signal = bool(
            not degenerate
            and len(anomalous_groups) >= int(self.config.get("consensus_history_min_anomalous_groups", 1))
            and positive_ratio <= (float(self.config.get("consensus_history_bootstrap_max_positive_ratio", 0.12)) + 1e-6)
            and top_score >= float(self.config.get("consensus_history_bootstrap_min_top_score", 2.80))
            and (
                score_gap >= float(self.config.get("consensus_history_bootstrap_min_score_gap", 0.18))
                or score_ratio >= float(self.config.get("consensus_history_bootstrap_min_score_ratio", 1.05))
            )
        )
        history_fallback_allowed = bool(
            not fallback_triggered
            or (
                cts_z_score >= float(self.config.get("consensus_history_fallback_min_cts_z", 1.70))
                and mean_consistency >= float(self.config.get("consensus_history_fallback_min_consistency", 0.70))
            )
        )
        projected_history_strict_eligible = bool(
            history_bootstrap_signal
            and len(anomalous_groups) >= int(self.config.get("consensus_history_min_anomalous_groups", 1))
            and positive_ratio <= (float(self.config.get("consensus_history_max_positive_ratio", 0.10)) + 1e-6)
            and top_score >= float(self.config.get("consensus_history_min_top_score", 3.0))
            and (
                bool(self.config.get("consensus_history_allow_fallback", False))
                or history_fallback_allowed
            )
        )
        projected_history_relaxed_eligible = bool(
            bool(self.config.get("consensus_history_relaxed_enabled", True))
            and len(anomalous_groups)
            >= int(self.config.get("consensus_history_relaxed_min_anomalous_groups", 1))
            and positive_ratio
            <= (float(self.config.get("consensus_history_relaxed_max_positive_ratio", 0.20)) + 1e-6)
            and top_score >= float(self.config.get("consensus_history_relaxed_min_top_score", 1.20))
            and (
                (not bool(self.config.get("consensus_history_relaxed_require_reliable", True)))
                or cts_signal_reliable
                or history_bootstrap_signal
            )
            and (
                bool(self.config.get("consensus_history_relaxed_allow_fallback", True))
                or history_fallback_allowed
            )
        )
        projected_history_eligible = bool(
            (projected_history_strict_eligible or projected_history_relaxed_eligible)
            and history_quality_ok
        )
        projected_historical_anomalous_audits = int(self._historical_anomalous_audits)
        projected_top_frequency = int(top_frequency)
        projected_top_persistence = float(top_persistence)
        projected_top_historical_mean = float(stability_stats.get("top_historical_mean", 0.0))
        if projected_history_eligible and len(top_client_scores) > 0:
            top_client_id = int(top_client_scores[0][0])
            current_max_score = max((float(score) for score in current_client_scores.values()), default=0.0)
            if current_max_score > 0.0:
                audit_weight = max(len(anomalous_groups), 1)
                projected_historical_anomalous_audits += 1
                normalized_top_score = (
                    max(float(current_client_scores.get(top_client_id, 0.0)), 0.0)
                    / max(current_max_score, 1e-6)
                )
                prior_weight = float(self._historical_score_weight.get(top_client_id, 0.0))
                projected_weight = prior_weight + float(audit_weight)
                if projected_weight > 0.0:
                    projected_top_historical_mean = (
                        float(self._historical_score_sum.get(top_client_id, 0.0))
                        + normalized_top_score * float(audit_weight)
                    ) / projected_weight
                projected_top_frequency += int(top_client_id in current_suspicious_set[:reweight_topk])
                projected_top_persistence = (
                    float(projected_top_frequency) / max(float(projected_historical_anomalous_audits), 1.0)
                )
        positive_ratio_limit = float(self.config.get("reweight_max_positive_ratio", 0.12))
        positive_ratio_within_limit = positive_ratio <= (positive_ratio_limit + 1e-6)
        stability_reliable = bool(
            len(anomalous_groups) >= int(self.config.get("reweight_min_anomalous_groups", 2))
            and top_score >= float(self.config.get("stability_reliable_min_score", 2.2))
            and top_frequency >= int(self.config.get("stability_min_top_frequency", 2))
            and top_persistence >= float(self.config.get("stability_min_persistence", 0.15))
            and not fallback_triggered
        )
        localization_reliable_strict = bool(
            self.config.get("enable_client_reweighting", True)
            and not degenerate
            and len(anomalous_groups) >= int(self.config.get("reweight_min_anomalous_groups", 2))
            and positive_ratio_within_limit
            and top_score >= float(self.config.get("reweight_min_top_score", 2.0))
            and (
                score_gap >= float(self.config.get("reweight_min_score_gap", 0.20))
                or score_ratio >= float(self.config.get("reweight_min_score_ratio", 1.10))
            )
            and (
                not fallback_triggered
                or (
                    len(anomalous_groups) >= 2
                    and mean_consistency <= 0.5
                    and cts_z_score >= 3.0
                )
            )
        )
        ranking_positive_ratio_limit = float(self.config.get("ranking_max_positive_ratio", 0.16))
        ranking_reliable = bool(
            not degenerate
            and len(anomalous_groups) >= int(self.config.get("ranking_min_anomalous_groups", 1))
            and positive_ratio <= (ranking_positive_ratio_limit + 1e-6)
            and top_score >= float(self.config.get("ranking_min_top_score", 2.30))
            and (
                score_gap >= float(self.config.get("ranking_min_score_gap", 0.50))
                or score_ratio >= float(self.config.get("ranking_min_score_ratio", 1.18))
            )
            and (
                not fallback_triggered
                or (
                    cts_z_score >= float(self.config.get("ranking_fallback_min_cts_z", 1.70))
                    and mean_consistency >= float(self.config.get("ranking_fallback_min_consistency", 0.70))
                )
            )
        )
        consensus_signal_reliable = bool(
            len(consensus_candidates) > 0
            and self._historical_anomalous_audits
            >= int(self.config.get("consensus_signal_min_historical_audits", 3))
            and positive_ratio <= (float(self.config.get("consensus_signal_max_positive_ratio", 0.12)) + 1e-6)
            and reliable_consensus_overlap
            >= int(self.config.get("consensus_signal_min_overlap", 1))
        )
        localization_confidence_score = float(len(anomalous_groups) / max(len(ordered_group_ids), 1))
        localization_reliable_min_anomalous_groups = int(
            self.config.get("localization_reliable_min_anomalous_groups", 2)
        )
        localization_reliable_max_positive_ratio = float(
            self.config.get("localization_reliable_max_positive_ratio", 0.12)
        )
        localization_reliable_min_cts_z = float(
            self.config.get("localization_reliable_min_cts_z", 1.4)
        )
        localization_reliable_min_consistency = float(
            self.config.get("localization_reliable_min_consistency", 0.50)
        )
        localization_reliable_min_rank_score = float(
            self.config.get("localization_reliable_min_rank_score", 0.30)
        )
        localization_reliable_min_selected_consistency = float(
            self.config.get("localization_reliable_min_selected_consistency", 0.55)
        )
        raw_localization_reliable = bool(
            len(anomalous_groups) >= localization_reliable_min_anomalous_groups
            and positive_ratio <= (localization_reliable_max_positive_ratio + 1e-6)
            and cts_z_score >= localization_reliable_min_cts_z
            and float(mean_consistency) >= localization_reliable_min_consistency
            and float(candidate_selected_rank_score) >= localization_reliable_min_rank_score
            and float(candidate_selected_consistency or 0.0)
            >= localization_reliable_min_selected_consistency
            and (
                (not fallback_triggered)
                or (
                    cts_z_score >= localization_reliable_min_cts_z + 0.20
                    and float(mean_consistency) >= localization_reliable_min_consistency + 0.03
                )
            )
        )
        
            
        reliable_fallback_allowed = bool(
            not fallback_triggered
            or (
                cts_z_score >= float(self.config.get("reliable_pool_fallback_min_cts_z", 1.70))
                and mean_consistency >= float(self.config.get("reliable_pool_fallback_min_consistency", 0.70))
            )
        )
        strong_reliable_override = bool(
            raw_localization_reliable
            and len(anomalous_groups) >= int(self.config.get("reliable_pool_strong_min_anomalous_groups", 2))
            and positive_ratio <= (float(self.config.get("reliable_pool_strong_max_positive_ratio", 0.10)) + 1e-6)
            and top_score >= float(self.config.get("reliable_pool_strong_min_top_score", 4.0))
            and (
                score_gap >= float(self.config.get("reliable_pool_strong_min_score_gap", 0.90))
                or score_ratio >= float(self.config.get("reliable_pool_strong_min_score_ratio", 1.24))
            )
            and self._historical_anomalous_audits
            >= int(self.config.get("reliable_pool_strong_min_historical_audits", 3))
            and float(stability_stats.get("top_historical_mean", 0.0))
            >= float(self.config.get("reliable_pool_strong_min_historical_mean", 0.80))
            and not fallback_triggered
        )
        weak_reliable_override = bool(
            (raw_localization_reliable or projected_history_eligible)
            and projected_historical_anomalous_audits
            >= int(self.config.get("reliable_pool_weak_min_historical_audits", 2))
            and positive_ratio <= (float(self.config.get("ranking_max_positive_ratio", 0.16)) + 1e-6)
            and top_score >= float(self.config.get("reliable_pool_weak_min_top_score", 2.8))
            and (
                score_gap >= float(self.config.get("reliable_pool_weak_min_score_gap", 0.18))
                or score_ratio >= float(self.config.get("reliable_pool_weak_min_score_ratio", 1.05))
            )
            and projected_top_historical_mean
            >= float(self.config.get("reliable_pool_weak_min_historical_mean", 0.75))
            and (
                reliable_consensus_overlap
                >= int(self.config.get("reliable_pool_weak_min_consensus_overlap", 1))
                or len(consensus_candidates) > 0
                or projected_history_eligible
            )
            and reliable_fallback_allowed
        )
        ranking_reliable_override = bool(
            ranking_reliable
            and self._historical_anomalous_audits >= int(self.config.get("reliable_pool_ranking_min_historical_audits", 3))
            and top_score >= float(self.config.get("reliable_pool_ranking_min_top_score", 3.0))
            and (
                score_gap >= float(self.config.get("reliable_pool_ranking_min_score_gap", 0.75))
                or score_ratio >= float(self.config.get("reliable_pool_ranking_min_score_ratio", 1.25))
            )
            and float(stability_stats.get("top_historical_mean", 0.0))
            >= float(self.config.get("reliable_pool_ranking_min_historical_mean", 0.55))
            and reliable_fallback_allowed
        )
        reliable_signal_eligible = bool(
            raw_localization_reliable
            or strong_reliable_override
            or weak_reliable_override
            or ranking_reliable_override
        )
        reliable_score_eligible = bool(
            reliable_signal_eligible
            and len(anomalous_groups) >= int(self.config.get("reliable_pool_min_anomalous_groups", 1))
            and top_score >= float(self.config.get("reliable_pool_min_top_score", 2.30))
            and (
                score_gap >= float(self.config.get("reliable_pool_min_score_gap", 0.35))
                or score_ratio >= float(self.config.get("reliable_pool_min_score_ratio", 1.12))
            )
            and (
                reliable_consensus_overlap >= int(self.config.get("reliable_pool_min_consensus_overlap", 2))
                or strong_reliable_override
                or weak_reliable_override
                or ranking_reliable_override
            )
            and (
                bool(self.config.get("reliable_pool_allow_fallback", False))
                or reliable_fallback_allowed
            )
        )
        if reliable_score_eligible:
            self._historical_reliable_audits += 1
            reliable_ranked_scores = sorted(
                ((int(client_id), float(score)) for client_id, score in refined_reliable_scores.items()),
                key=lambda item: (-item[1], item[0]),
            )
            current_suspicious_set = [
                int(client_id)
                for client_id, _ in reliable_ranked_scores[: int(self.config["suspicious_topk"])]
            ]
            localization.client_scores = refined_reliable_scores
            localization.flagged_clients = current_suspicious_set
            localization.aux_stats["selected_scores"] = {
                int(client_id): float(refined_reliable_scores.get(client_id, 0.0))
                for client_id in current_suspicious_set
            }
            self.current_suspicious_set = list(current_suspicious_set)
            top_client_scores = reliable_ranked_scores[: max(10, int(self.config["suspicious_topk"]))]

        initial_suspicious_clients = [int(client_id) for client_id in current_suspicious_set]
        fct_metadata = dict(metadata)
        fct_metadata["fct_initial_scores"] = {
            int(client_id): float(score)
            for client_id, score in current_client_scores.items()
        }
        confirmed_suspects, fct_stats_by_client, fct_fallback_reason, fct_summary = self._run_fct_confirmation(
            round_idx=int(round_idx),
            global_model=global_model,
            clean_dataloader=clean_dataloader,
            metadata=fct_metadata,
            initial_suspicious_ranking=initial_suspicious_clients,
        )
        confirmed_suspects = [int(client_id) for client_id in confirmed_suspects]
        confirmation_rejected_suspects = [
            int(client_id) for client_id in fct_summary.get("fct_rejected_suspects", [])
        ]
        risk_confirmed_suspects = [int(client_id) for client_id in confirmed_suspects]
        risk_source_cfg = str(self.dgc_repair.config.get("risk_aware_source", "raw")).strip().lower()
        risk_highconf_enabled = bool(self.config.get("risk_confirmed_highconf_enable", True))
        risk_highconf_min_fct_z = float(self.config.get("risk_confirmed_highconf_min_fct_z", 1.5))
        risk_highconf_min_pairs = int(max(self.config.get("risk_confirmed_highconf_min_pairs", 4), 0))
        risk_highconf_require_reliable = bool(
            self.config.get("risk_confirmed_highconf_require_localization_reliable", True)
        )
        risk_confirmed_filter_reason = "not_required"
        if risk_source_cfg == "confirmed" and risk_highconf_enabled:
            if risk_highconf_require_reliable and (not bool(raw_localization_reliable)):
                risk_confirmed_suspects = []
                risk_confirmed_filter_reason = "localization_not_reliable"
            else:
                filtered_confirmed: List[int] = []
                for client_id in confirmed_suspects:
                    stats = fct_stats_by_client.get(int(client_id), {})
                    fct_z_value = float(stats.get("fct_z", 0.0))
                    num_pairs_valid = int(stats.get("num_pairs_valid", 0))
                    if fct_z_value >= risk_highconf_min_fct_z and num_pairs_valid >= risk_highconf_min_pairs:
                        filtered_confirmed.append(int(client_id))
                risk_confirmed_suspects = filtered_confirmed
                risk_confirmed_filter_reason = "filtered_by_fct_confidence"
        self.current_confirmed_suspects = list(confirmed_suspects)
        self.current_fct_stats = {int(client_id): dict(stats) for client_id, stats in fct_stats_by_client.items()}
        self.current_fct_fallback_reason = str(fct_fallback_reason)

        previous_suspicious_clients = set(self.suspicious_clients)
        current_reweight_candidates = {
            int(client_id) for client_id in current_suspicious_set[:reweight_topk]
        }
        current_consensus_overlap = int(len(current_reweight_candidates & consensus_top_for_pool))
        consensus_backed_reweight_supported = bool(
            len(consensus_candidates) > 0
            and current_consensus_overlap
            >= int(self.config.get("reweight_min_consensus_overlap", 1))
        )
        strong_signal_without_consensus = bool(
            bool(self.config.get("reweight_allow_no_consensus_with_strong_signal", False))
            and len(consensus_candidates) == 0
            and raw_localization_reliable
            and top_score >= float(self.config.get("reweight_no_consensus_min_top_score", 2.4))
            and (
                score_gap >= float(self.config.get("reweight_no_consensus_min_score_gap", 0.30))
                or score_ratio >= float(self.config.get("reweight_no_consensus_min_score_ratio", 1.10))
            )
            and projected_historical_anomalous_audits
            >= int(self.config.get("reweight_no_consensus_min_historical_audits", 2))
        )
        current_reweight_supported = bool(
            consensus_backed_reweight_supported or strong_signal_without_consensus
        )
        retained_consensus_clients = {
            int(client_id) for client_id in previous_suspicious_clients & set(consensus_candidates)
        }
        retained_consensus_overlap = int(len(retained_consensus_clients))
        retained_reweight_supported = bool(
            len(consensus_candidates) > 0
            and retained_consensus_overlap
            >= int(self.config.get("retain_reweight_min_consensus_overlap", 1))
        )
        reliable_reweight_min_audits = int(self.config.get("reweight_min_reliable_audits", 3))
        reliable_reweighting_ready = bool(
            reliable_score_eligible
            and len(anomalous_groups) >= int(self.config.get("reweight_min_anomalous_groups", 2))
            and self._historical_reliable_audits >= reliable_reweight_min_audits
            and current_reweight_supported
            and reweight_quality_ok
        )
        provisional_reweighting_ready = bool(
            bool(self.config.get("enable_provisional_reweighting", False))
            and not reliable_reweighting_ready
            and reliable_score_eligible
            and len(anomalous_groups)
            >= int(self.config.get("provisional_reweight_min_anomalous_groups", 2))
            and positive_ratio
            <= (float(self.config.get("provisional_reweight_max_positive_ratio", 0.10)) + 1e-6)
            and top_score >= float(self.config.get("provisional_reweight_min_top_score", 2.0))
            and (
                score_gap >= float(self.config.get("provisional_reweight_min_score_gap", 0.25))
                or score_ratio >= float(self.config.get("provisional_reweight_min_score_ratio", 1.10))
            )
            and (
                not bool(self.config.get("provisional_reweight_require_nonfallback", True))
                or not fallback_triggered
            )
            and reweight_quality_ok
        )
        consensus_reweight_min_anomalous_groups = int(
            self.config.get(
                "consensus_reweight_min_anomalous_groups",
                self.config.get("reweight_min_anomalous_groups", 2),
            )
        )
        consensus_reweight_max_positive_ratio = float(
            self.config.get("consensus_reweight_max_positive_ratio", positive_ratio_limit)
        )
        consensus_reweight_min_historical_audits = int(
            self.config.get(
                "consensus_reweight_min_historical_audits",
                max(2, int(self.config.get("consensus_min_anomalous_audits", 2))),
            )
        )
        consensus_reweight_min_candidates = max(
            1, int(self.config.get("consensus_reweight_min_consensus_candidates", 1))
        )
        consensus_reweight_require_overlap = bool(
            self.config.get("consensus_reweight_require_overlap", True)
        )
        consensus_reweight_require_signal = bool(
            self.config.get("consensus_reweight_require_signal", True)
        )
        consensus_reweight_require_current_overlap = bool(
            self.config.get("consensus_reweight_require_current_overlap", True)
        )
        consensus_reweight_intersection_only = bool(
            self.config.get("consensus_reweight_intersection_only", True)
        )
        consensus_reweight_allow_top_consensus_fallback = bool(
            self.config.get("consensus_reweight_allow_top_consensus_fallback", True)
        )
        consensus_overlap_ready = bool(
            not consensus_reweight_require_overlap
            or current_reweight_supported
            or retained_reweight_supported
        )
        if consensus_reweight_require_current_overlap:
            consensus_overlap_ready = bool(consensus_overlap_ready and current_reweight_supported)
        consensus_signal_ready = bool(
            not consensus_reweight_require_signal
            or (raw_localization_reliable and consensus_signal_reliable)
        )
        consensus_reweighting_ready = bool(
            bool(self.config.get("enable_consensus_reweighting", True))
            and len(anomalous_groups) >= consensus_reweight_min_anomalous_groups
            and positive_ratio <= (consensus_reweight_max_positive_ratio + 1e-6)
            and self._historical_anomalous_audits >= consensus_reweight_min_historical_audits
            and len(consensus_candidates) >= consensus_reweight_min_candidates
            and consensus_overlap_ready
            and consensus_signal_ready
            and reweight_quality_ok
        )
        retain_reweight_min_history_audits = int(
            self.config.get("retain_reweight_min_history_audits", reliable_reweight_min_audits)
        )
        fallback_history_supported = bool(projected_history_eligible or history_bootstrap_signal)
        fallback_min_historical_audits = int(self.config.get("fallback_reweight_min_historical_audits", 2))
        fallback_min_top_frequency = int(self.config.get("fallback_reweight_min_top_frequency", 1))
        fallback_reweighting_ready = bool(
            bool(self.config.get("enable_fallback_reweighting", False))
            and not reliable_reweighting_ready
            and not provisional_reweighting_ready
            and not consensus_reweighting_ready
            and self._historical_anomalous_audits >= fallback_min_historical_audits
            and projected_top_frequency >= fallback_min_top_frequency
            and len(current_suspicious_set) > 0
            and len(anomalous_groups)
            >= int(self.config.get("fallback_reweight_min_anomalous_groups", 1))
            and positive_ratio
            <= (float(self.config.get("fallback_reweight_max_positive_ratio", 0.12)) + 1e-6)
            and top_score >= float(self.config.get("fallback_reweight_min_top_score", 2.0))
            and (
                score_gap >= float(self.config.get("fallback_reweight_min_score_gap", 0.20))
                or score_ratio >= float(self.config.get("fallback_reweight_min_score_ratio", 1.10))
            )
            and (
                not bool(self.config.get("fallback_reweight_require_history_support", True))
                or fallback_history_supported
            )
            and (
                not bool(self.config.get("fallback_reweight_require_consensus_overlap", True))
                or current_reweight_supported
                or retained_reweight_supported
            )
            and (
                not bool(self.config.get("fallback_reweight_require_nonfallback", False))
                or not fallback_triggered
            )
            and reweight_quality_ok
        )
        consensus_reweighting_applied = False
        retained_reweighting_applied = False
        provisional_reweighting_applied = False
        ranking_reweighting_applied = False
        fallback_reweighting_applied = False
        ranking_reweight_min_consensus_overlap = int(
            self.config.get("ranking_reweight_min_consensus_overlap", 1)
        )
        ranking_reweight_consensus_supported = bool(
            current_consensus_overlap >= ranking_reweight_min_consensus_overlap
            or retained_reweight_supported
        )
        ranking_reweight_strong_signal_override = bool(
            bool(self.config.get("ranking_reweight_enable_strong_override", True))
            and projected_history_eligible
            and projected_historical_anomalous_audits
            >= int(self.config.get("ranking_reweight_override_min_historical_audits", 4))
            and top_score >= float(self.config.get("ranking_reweight_override_min_top_score", 3.5))
            and (
                score_gap >= float(self.config.get("ranking_reweight_override_min_score_gap", 0.50))
                or score_ratio >= float(self.config.get("ranking_reweight_override_min_score_ratio", 1.20))
            )
        )
        ranking_reweight_local_signal_override = bool(
            bool(self.config.get("ranking_reweight_enable_local_override", True))
            and ranking_reliable
            and projected_historical_anomalous_audits
            >= int(self.config.get("ranking_reweight_local_override_min_historical_audits", 2))
            and top_score >= float(self.config.get("ranking_reweight_local_override_min_top_score", 1.90))
            and (
                score_gap >= float(self.config.get("ranking_reweight_local_override_min_score_gap", 0.18))
                or score_ratio >= float(self.config.get("ranking_reweight_local_override_min_score_ratio", 1.08))
            )
            and (
                not bool(self.config.get("ranking_reweight_local_override_require_nonfallback", False))
                or not fallback_triggered
            )
        )
        ranking_reweight_support_ok = bool(
            ranking_reweight_consensus_supported
            or ranking_reweight_strong_signal_override
            or ranking_reweight_local_signal_override
        )
        ranking_reweighting_ready = bool(
            bool(self.config.get("enable_ranking_reweighting", False))
            and not reliable_reweighting_ready
            and not provisional_reweighting_ready
            and ranking_reliable
            and raw_localization_reliable
            and len(current_suspicious_set) > 0
            and len(anomalous_groups)
            >= int(self.config.get("ranking_reweight_min_anomalous_groups", 1))
            and positive_ratio
            <= (float(self.config.get("ranking_reweight_max_positive_ratio", 0.12)) + 1e-6)
            and top_score >= float(self.config.get("ranking_reweight_min_top_score", 2.0))
            and projected_historical_anomalous_audits
            >= int(self.config.get("ranking_reweight_min_historical_audits", 3))
            and projected_top_persistence
            >= float(self.config.get("ranking_reweight_min_top_persistence", 0.10))
            and projected_top_historical_mean
            >= float(self.config.get("ranking_reweight_min_historical_mean", 0.20))
            and ranking_reweight_support_ok
            and (
                not bool(self.config.get("ranking_reweight_require_nonfallback", False))
                or not fallback_triggered
            )
            and reweight_quality_ok
        )
        reweight_require_confirmed_suspects = bool(
            self.config.get("reweight_require_confirmed_suspects", True)
        )
        reweight_require_confirmed_min_count = int(
            max(self.config.get("reweight_require_confirmed_min_count", 1), 0)
        )
        reweight_confirmed_gate_active = bool(
            reweight_require_confirmed_suspects and bool(self.config.get("fct_enable", False))
        )
        reweight_confirmed_ready = bool(
            (not reweight_confirmed_gate_active)
            or (
                len(confirmed_suspects)
                >= max(int(reweight_require_confirmed_min_count), 1)
            )
        )
        reweight_fct_quality_gate_active = bool(
            reweight_confirmed_gate_active
            and bool(self.config.get("reweight_require_fct_quality_gate", True))
        )
        reweight_fct_min_confirmation_rate = float(
            max(self.config.get("reweight_fct_min_confirmation_rate", 0.35), 0.0)
        )
        reweight_fct_min_mean_z = float(
            max(self.config.get("reweight_fct_min_mean_z", 1.20), 0.0)
        )
        reweight_fct_min_mean_diff = float(
            self.config.get("reweight_fct_min_mean_diff", 0.00)
        )
        reweight_fct_max_clean_pool_contamination_risk = float(
            np.clip(
                self.config.get("reweight_fct_max_clean_pool_contamination_risk", 0.40),
                0.0,
                1.0,
            )
        )
        fct_confirmation_rate = float(
            max(fct_summary.get("fct_confirmation_rate", 0.0), 0.0)
        )
        fct_contamination_risk = float(
            np.clip(
                fct_summary.get("fct_clean_pool_contamination_risk_estimate", 0.0),
                0.0,
                1.0,
            )
        )
        confirmed_z_values: List[float] = []
        confirmed_mean_values: List[float] = []
        for client_id in confirmed_suspects:
            stats = fct_stats_by_client.get(int(client_id), {})
            confirmed_z_values.append(float(stats.get("fct_z", 0.0)))
            confirmed_mean_values.append(float(stats.get("fct_mean", 0.0)))
        fct_confirmed_mean_z = float(np.mean(np.asarray(confirmed_z_values, dtype=np.float32))) if len(confirmed_z_values) > 0 else 0.0
        fct_confirmed_mean_diff = float(np.mean(np.asarray(confirmed_mean_values, dtype=np.float32))) if len(confirmed_mean_values) > 0 else 0.0
        reweight_fct_quality_ready = bool(
            (not reweight_fct_quality_gate_active)
            or (
                fct_confirmation_rate >= reweight_fct_min_confirmation_rate
                and fct_confirmed_mean_z >= reweight_fct_min_mean_z
                and fct_confirmed_mean_diff >= reweight_fct_min_mean_diff
                and fct_contamination_risk <= reweight_fct_max_clean_pool_contamination_risk
            )
        )
        if not reweight_confirmed_ready:
            reliable_reweighting_ready = False
            provisional_reweighting_ready = False
            ranking_reweighting_ready = False
            consensus_reweighting_ready = False
            fallback_reweighting_ready = False
        if not reweight_fct_quality_ready:
            reliable_reweighting_ready = False
            provisional_reweighting_ready = False
            ranking_reweighting_ready = False
            consensus_reweighting_ready = False
            fallback_reweighting_ready = False
        if reliable_reweighting_ready:
            self.suspicious_clients = self._select_reweight_clients(current_suspicious_set, reweight_topk)
        else:
            if provisional_reweighting_ready:
                provisional_topk = max(
                    1,
                    int(self.config.get("provisional_reweight_topk", reweight_topk)),
                )
                self.suspicious_clients = self._select_reweight_clients(current_suspicious_set, provisional_topk)
                provisional_reweighting_applied = len(self.suspicious_clients) > 0
            elif ranking_reweighting_ready:
                ranking_topk = max(
                    1,
                    int(self.config.get("ranking_reweight_topk", reweight_topk)),
                )
                self.suspicious_clients = self._select_reweight_clients(current_suspicious_set, ranking_topk)
                ranking_reweighting_applied = len(self.suspicious_clients) > 0
            elif consensus_reweighting_ready:
                if consensus_reweight_intersection_only:
                    current_reweight_slice = set(
                        int(client_id)
                        for client_id in current_suspicious_set[: max(reweight_topk, reliable_pool_candidate_topk)]
                    )
                    selected_consensus_clients = [
                        int(client_id)
                        for client_id in consensus_candidates
                        if int(client_id) in current_reweight_slice
                    ]
                    if (
                        len(selected_consensus_clients) == 0
                        and consensus_reweight_allow_top_consensus_fallback
                    ):
                        selected_consensus_clients = [int(client_id) for client_id in consensus_candidates]
                else:
                    selected_consensus_clients = [int(client_id) for client_id in consensus_candidates]
                self.suspicious_clients = self._select_reweight_clients(selected_consensus_clients, reweight_topk)
                consensus_reweighting_applied = len(self.suspicious_clients) > 0
            elif (
                bool(self.config.get("retain_reweight_on_unreliable", True))
                and len(previous_suspicious_clients) > 0
                and max(
                    int(self._historical_reliable_audits),
                    int(self._historical_anomalous_audits),
                ) >= retain_reweight_min_history_audits
                and len(anomalous_groups) >= int(self.config.get("retain_reweight_min_anomalous_groups", 1))
                and positive_ratio_within_limit
                and retained_reweight_supported
            ):
                if bool(self.config.get("retain_reweight_intersection_only", True)):
                    self.suspicious_clients = set(int(client_id) for client_id in retained_consensus_clients)
                else:
                    self.suspicious_clients = set(int(client_id) for client_id in previous_suspicious_clients)
                retained_reweighting_applied = True
            elif fallback_reweighting_ready:
                fallback_topk = max(
                    1,
                    int(self.config.get("fallback_reweight_topk", 1)),
                )
                self.suspicious_clients = self._select_reweight_clients(current_suspicious_set, fallback_topk)
                fallback_reweighting_applied = len(self.suspicious_clients) > 0
            else:
                self.suspicious_clients = set()
            if not (
                provisional_reweighting_applied
                or ranking_reweighting_applied
                or consensus_reweighting_applied
                or retained_reweighting_applied
                or fallback_reweighting_applied
            ):
                if not (
                    dense_mode_active
                    and bool(self.config.get("dense_mode_allow_without_reweight", True))
                ):
                    cleaning_p_hat = min(
                        cleaning_p_hat,
                        float(self.config.get("cleaning_max_p_hat_when_unreliable", 0.08)),
                    )
        consensus_history_strict_eligible = bool(
            (raw_localization_reliable or history_bootstrap_signal)
            and len(anomalous_groups) >= int(self.config.get("consensus_history_min_anomalous_groups", 1))
            and positive_ratio <= (float(self.config.get("consensus_history_max_positive_ratio", 0.10)) + 1e-6)
            and top_score >= float(self.config.get("consensus_history_min_top_score", 3.0))
            and (
                score_gap >= float(self.config.get("consensus_history_min_score_gap", 0.75))
                or score_ratio >= float(self.config.get("consensus_history_min_score_ratio", 1.20))
                or history_bootstrap_signal
            )
            and (
                bool(self.config.get("consensus_history_allow_fallback", False))
                or history_fallback_allowed
            )
        )
        consensus_history_relaxed_reliable_ok = bool(
            raw_localization_reliable
            or cts_signal_reliable
            or history_bootstrap_signal
        )
        consensus_history_relaxed_eligible = bool(
            bool(self.config.get("consensus_history_relaxed_enabled", True))
            and len(anomalous_groups)
            >= int(self.config.get("consensus_history_relaxed_min_anomalous_groups", 1))
            and positive_ratio
            <= (float(self.config.get("consensus_history_relaxed_max_positive_ratio", 0.20)) + 1e-6)
            and top_score >= float(self.config.get("consensus_history_relaxed_min_top_score", 1.20))
            and (
                (not bool(self.config.get("consensus_history_relaxed_require_reliable", True)))
                or consensus_history_relaxed_reliable_ok
            )
            and (
                bool(self.config.get("consensus_history_relaxed_allow_fallback", True))
                or history_fallback_allowed
            )
        )
        consensus_history_eligible = bool(
            (consensus_history_strict_eligible or consensus_history_relaxed_eligible)
            and history_quality_ok
        )
        if consensus_history_eligible:
            self._update_historical_consensus(
                current_client_scores=current_client_scores,
                ranked_clients=[int(client_id) for client_id, _ in raw_order],
                audit_weight=max(len(anomalous_groups), 1),
            )
        self.current_localization_reliable = bool(raw_localization_reliable)
        self.current_cleaning_p_hat = float(cleaning_p_hat)
        selected_client_set = set(int(client_id) for client_id in self.selected_clients)
        reweighting_enabled = bool(self.config.get("enable_client_reweighting", True))
        effective_applied_clients = {
            int(client_id)
            for client_id in self.suspicious_clients
            if int(client_id) in selected_client_set
        }
        reweighting_applied_for_logging = bool(
            reweighting_enabled and len(effective_applied_clients) > 0
        )
        if not reweighting_enabled:
            reweighting_gate_reason = "reweighting_disabled"
        elif reweight_confirmed_gate_active and (not reweight_confirmed_ready):
            reweighting_gate_reason = "blocked_no_confirmed_suspects"
        elif reweight_fct_quality_gate_active and (not reweight_fct_quality_ready):
            reweighting_gate_reason = "blocked_low_fct_quality"
        elif reliable_reweighting_ready and len(self.suspicious_clients) > 0:
            reweighting_gate_reason = "applied_reliable_gate"
        elif provisional_reweighting_applied and len(self.suspicious_clients) > 0:
            reweighting_gate_reason = "applied_provisional_gate"
        elif ranking_reweighting_applied and len(self.suspicious_clients) > 0:
            reweighting_gate_reason = "applied_ranking_gate"
        elif consensus_reweighting_applied and len(self.suspicious_clients) > 0:
            reweighting_gate_reason = "applied_consensus_gate"
        elif retained_reweighting_applied and len(self.suspicious_clients) > 0:
            reweighting_gate_reason = "applied_retained_gate"
        elif fallback_reweighting_applied and len(self.suspicious_clients) > 0:
            reweighting_gate_reason = "applied_fallback_gate"
        else:
            reweighting_gate_reason = "skipped_gate_not_satisfied"

        dgc_repair_enabled = bool(self.config.get("dgc_repair_enable", True)) and bool(self.dgc_repair.enabled)
        gate_stats: Dict[str, Any] = {}
        risk_stats: Dict[str, Any] = {}
        risk_activation_reason = "disabled"
        risk_aware_cfg_enabled = bool(self.config.get("risk_aware_enable", False))
        risk_aware_require_fct = bool(self.config.get("risk_aware_require_fct", False))
        risk_source_effective = str(self.dgc_repair.config.get("risk_aware_source", "raw")).strip().lower()
        if risk_source_effective not in {"raw", "confirmed"}:
            risk_source_effective = "raw"
        risk_activation_passed = False
        recent_raw_candidate_precision_mean: Optional[float] = None
        recent_raw_proxy_cv: Optional[float] = None
        raw_precision_current: Optional[float] = None
        group_members_map = {
            int(group_id): [int(client_id) for client_id in self.current_group_clients.get(int(group_id), [])]
            for group_id in ordered_group_ids
        }
        malicious_ids_raw = metadata.get("malicious_client_ids")
        malicious_set_for_eval: Set[int] = set()
        if isinstance(malicious_ids_raw, Sequence) and not isinstance(malicious_ids_raw, (str, bytes)):
            malicious_set_for_eval = {int(client_id) for client_id in malicious_ids_raw}
        candidate_ids_for_precision = [
            int(client_id) for client_id in fct_summary.get("fct_initial_suspects", initial_suspicious_clients)
        ]
        if len(malicious_set_for_eval) > 0 and len(candidate_ids_for_precision) > 0:
            raw_precision_current = float(
                len(set(candidate_ids_for_precision).intersection(malicious_set_for_eval))
                / max(len(candidate_ids_for_precision), 1)
            )
            self._risk_candidate_precision_history.append(float(raw_precision_current))
            if len(self._risk_candidate_precision_history) > 200:
                self._risk_candidate_precision_history = self._risk_candidate_precision_history[-200:]
        if risk_aware_cfg_enabled:
            if risk_aware_require_fct and (not bool(self.config.get("fct_enable", False))):
                risk_activation_reason = "require_fct_disabled"
            elif risk_source_effective == "confirmed":
                if len(risk_confirmed_suspects) > 0:
                    risk_activation_passed = True
                    risk_activation_reason = "confirmed_available"
                else:
                    risk_activation_reason = "no_confirmed_suspects"
            else:
                precision_min = float(max(self.config.get("risk_aware_raw_precision_min", 0.30), 0.0))
                if len(self._risk_candidate_precision_history) >= 3:
                    recent = self._risk_candidate_precision_history[-3:]
                    recent_raw_candidate_precision_mean = float(np.mean(recent))
                    if recent_raw_candidate_precision_mean >= precision_min:
                        risk_activation_passed = True
                        risk_activation_reason = "raw_precision_gate_passed"
                    else:
                        risk_activation_reason = "raw_precision_gate_failed"
                else:
                    ratio_history = [
                        float(entry.get("abnormal_group_ratio", 0.0))
                        for entry in self.audit_history[-2:]
                    ]
                    ratio_history.append(float(positive_ratio))
                    if len(ratio_history) >= 3:
                        ratio_arr = np.asarray(ratio_history[-3:], dtype=np.float64)
                        ratio_mean = float(np.mean(ratio_arr))
                        ratio_std = float(np.std(ratio_arr))
                        recent_raw_proxy_cv = float(ratio_std / max(abs(ratio_mean), 1e-8))
                        proxy_cv_max = float(max(self.config.get("risk_aware_proxy_cv_max", 0.50), 1e-6))
                        if recent_raw_proxy_cv <= proxy_cv_max:
                            risk_activation_passed = True
                            risk_activation_reason = "raw_proxy_stable"
                        else:
                            risk_activation_reason = "raw_proxy_unstable"
                    else:
                        risk_activation_reason = "insufficient_history"
        self.current_risk_sampling_enabled = bool(risk_aware_cfg_enabled and risk_activation_passed)
        self.current_risk_activation_reason = str(risk_activation_reason)
        if dgc_repair_enabled:
            diff_score_map = {
                int(group_id): float(detection.aux_stats.get("diff_group_scores", {}).get(int(group_id), 0.0))
                for group_id in ordered_group_ids
            }
            gate_stats = self.dgc_repair.compute_audit_gate(
                group_ids=ordered_group_ids,
                group_updates=group_update_mapping_raw,
                group_members=group_members_map,
                cts_diff_map=diff_score_map,
                b_g_map={int(gid): bool(b_g_map.get(gid, False)) for gid in ordered_group_ids},
                confirmed_suspects=risk_confirmed_suspects,
            )
            self.current_gate_update = self.dgc_repair.last_gate_update
            if self.current_risk_sampling_enabled:
                risk_stats = self.dgc_repair.update_risk_scores(
                    group_ids=ordered_group_ids,
                    group_members=group_members_map,
                    b_g_map={int(gid): bool(b_g_map.get(gid, False)) for gid in ordered_group_ids},
                    selected_clients=self.selected_clients,
                    confirmed_suspects=risk_confirmed_suspects,
                )
            else:
                risk_stats = {
                    "risk_score_mean": float(self.dgc_repair.risk_scores.mean().item()),
                    "risk_score_max": float(self.dgc_repair.risk_scores.max().item()),
                    "risk_aware_source": str(risk_source_effective),
                    "risk_confirmed_count": int(len(risk_confirmed_suspects)),
                    "risk_signal_mean": 0.0,
                    "risk_signal_min": 0.0,
                    "risk_signal_max": 0.0,
                    "risk_update_applied": False,
                    "risk_update_skip_reason": str(risk_activation_reason),
                }
        else:
            self.current_gate_update = None
        risk_scores_selected = {
            int(client_id): float(self.dgc_repair.risk_scores[int(client_id)].item())
            for client_id in self.selected_clients
            if 0 <= int(client_id) < int(self.num_clients)
        }
        risk_scores_all = {
            int(client_id): float(self.dgc_repair.risk_scores[int(client_id)].item())
            for client_id in range(int(self.num_clients))
        }
        risk_scores_top20 = dict(
            sorted(risk_scores_all.items(), key=lambda item: (-item[1], item[0]))[:20]
        )
        sampling_probability_selected: Dict[int, float] = {}
        sampling_probability_top20: Dict[int, float] = {}
        try:
            sampling_probs = self.dgc_repair.get_sampling_probabilities()
            sampling_probability_selected = {
                int(client_id): float(sampling_probs[int(client_id)])
                for client_id in self.selected_clients
                if 0 <= int(client_id) < len(sampling_probs)
            }
            sampled_pairs = [(int(client_id), float(sampling_probs[int(client_id)])) for client_id in range(len(sampling_probs))]
            sampling_probability_top20 = dict(
                sorted(sampled_pairs, key=lambda item: (-item[1], item[0]))[:20]
            )
        except Exception:
            sampling_probability_selected = {}
            sampling_probability_top20 = {}
        confirmed_set = {int(client_id) for client_id in confirmed_suspects}
        confirmed_risk_supported_groups = [
            int(group_id)
            for group_id in ordered_group_ids
            if len(confirmed_set.intersection(set(group_members_map.get(int(group_id), [])))) > 0
        ]

        audit_log = {
            "round": int(round_idx),
            "y_star": int(self.y_star_estimate),
            "y_star_source": detection.aux_stats.get("y_star_source", "estimated"),
            "cts_group_update_scale": str(self.config.get("cts_group_update_scale", "mean")),
            "cts_scores": list(self.current_cts_scores),
            "localization_signal_scores": list(self.current_localization_signal_scores),
            "localization_signal_source_requested": str(localization_signal_source_requested),
            "localization_signal_source_effective": str(localization_signal_source_effective),
            "group_ids": list(ordered_group_ids),
            "group_members": {
                int(group_id): [int(client_id) for client_id in ordered_group_assignments[index]]
                for index, group_id in enumerate(ordered_group_ids)
            },
            "group_sizes": [int(len(group)) for group in ordered_group_assignments],
            "anomalous_groups": anomalous_groups,
            "anomalous_group_count": int(len(anomalous_groups)),
            "raw_anomalous_groups": [int(group_id) for group_id in raw_anomalous_groups],
            "raw_anomalous_group_count": int(len(raw_anomalous_groups)),
            "tsc_hard_gate_filtered_groups": [int(group_id) for group_id in tsc_hard_gate_filtered_groups],
            "tsc_hard_gate_filtered_count": int(len(tsc_hard_gate_filtered_groups)),
            "dcbd_tsc_hard_gate_enable": bool(dcbd_tsc_hard_gate_enable),
            "dcbd_tsc_hard_gate_min": float(dcbd_tsc_hard_gate_min),
            "dcbd_tsc_hard_gate_use_abs": bool(dcbd_tsc_hard_gate_use_abs),
            "dcbd_tsc_hard_gate_apply_anomalous": bool(dcbd_tsc_hard_gate_apply_anomalous),
            "dcbd_tsc_hard_gate_apply_hist_keep": bool(dcbd_tsc_hard_gate_apply_hist_keep),
            "dcbd_tsc_hard_gate_apply_candidate": bool(dcbd_tsc_hard_gate_apply_candidate),
            "positive_ratio_current": positive_ratio,
            "num_abnormal_groups": int(len(anomalous_groups)),
            "abnormal_group_ratio": float(positive_ratio),
            "selected_clients": [int(client_id) for client_id in self.selected_clients],
            "initial_suspicious_clients": [int(client_id) for client_id in initial_suspicious_clients],
            "suspicious_clients": list(self.current_suspicious_set),
            "confirmed_suspects": [int(client_id) for client_id in confirmed_suspects],
            "risk_confirmed_suspects": [int(client_id) for client_id in risk_confirmed_suspects],
            "risk_confirmed_filter_reason": str(risk_confirmed_filter_reason),
            "risk_confirmed_highconf_enabled": bool(risk_highconf_enabled),
            "risk_confirmed_highconf_min_fct_z": float(risk_highconf_min_fct_z),
            "risk_confirmed_highconf_min_pairs": int(risk_highconf_min_pairs),
            "risk_confirmed_highconf_require_localization_reliable": bool(risk_highconf_require_reliable),
            "confirmation_rejected_suspects": [int(client_id) for client_id in confirmation_rejected_suspects],
            "fct_stats_by_client": {
                int(client_id): {
                    "fct_mean": float(stats.get("fct_mean", 0.0)),
                    "fct_std": float(stats.get("fct_std", 0.0)),
                    "fct_z": float(stats.get("fct_z", 0.0)),
                    "fct_mean_raw": float(stats.get("fct_mean_raw", 0.0)),
                    "fct_std_raw": float(stats.get("fct_std_raw", 0.0)),
                    "fct_z_std": float(stats.get("fct_z_std", 0.0)),
                    "fct_z_mad": float(stats.get("fct_z_mad", 0.0)),
                    "fct_winsor_applied": bool(stats.get("fct_winsor_applied", False)),
                    "fct_winsor_q": float(stats.get("fct_winsor_q", 0.0)),
                    "fct_confirmed_raw": bool(stats.get("fct_confirmed_raw", False)),
                    "fct_stability_pass_count": int(stats.get("fct_stability_pass_count", 0)),
                    "num_pairs_requested": int(stats.get("num_pairs_requested", 0)),
                    "num_pairs_valid": int(stats.get("num_pairs_valid", 0)),
                    "confirmed": bool(stats.get("confirmed", False)),
                    "fallback_below_k_min": bool(stats.get("fallback_below_k_min", False)),
                    "fct_pool_filter_reason": str(stats.get("fct_pool_filter_reason", "none")),
                    "fct_pool_base_size": int(stats.get("fct_pool_base_size", 0)),
                    "skip_reason": str(stats.get("skip_reason", "none")),
                    "fct_diffs": [float(v) for v in stats.get("fct_diffs", stats.get("pair_diffs", []))],
                    "pair_diffs": [float(v) for v in stats.get("pair_diffs", [])],
                }
                for client_id, stats in fct_stats_by_client.items()
            },
            "fct_enabled": bool(fct_summary.get("fct_enabled", self.config.get("fct_enable", False))),
            "fct_skipped": bool(fct_summary.get("fct_skipped", False)),
            "fct_skipped_reason": str(fct_summary.get("fct_skipped_reason", "none")),
            "fct_topn": int(max(self.config.get("fct_topn", 6), 1)),
            "fct_num_pairs": int(max(self.config.get("fct_num_pairs", 6), 1)),
            "fct_z_threshold": float(self.config.get("fct_z_threshold", 1.0)),
            "fct_use_matched_controls": bool(self.config.get("fct_use_matched_controls", True)),
            "fct_fallback_reason": str(fct_fallback_reason),
            "fct_clean_pool_size": int(fct_summary.get("fct_clean_pool_size", 0)),
            "fct_clean_pool_resampled": bool(fct_summary.get("fct_clean_pool_resampled", False)),
            "fct_clean_pool_strategy": str(fct_summary.get("fct_clean_pool_strategy", "exclude_topn")),
            "fct_clean_pool_contamination_risk_estimate": float(
                fct_summary.get("fct_clean_pool_contamination_risk_estimate", 0.0)
            ),
            "fct_aggregate_mode": str(fct_summary.get("fct_aggregate_mode", metadata.get("fct_aggregate_mode", "unknown"))),
            "fct_initial_suspects": [int(client_id) for client_id in fct_summary.get("fct_initial_suspects", [])],
            "fct_confirmed_suspects": [int(client_id) for client_id in fct_summary.get("fct_confirmed_suspects", confirmed_suspects)],
            "fct_confirmed_suspects_raw": [int(client_id) for client_id in fct_summary.get("fct_confirmed_suspects_raw", [])],
            "fct_rejected_suspects": [int(client_id) for client_id in fct_summary.get("fct_rejected_suspects", confirmation_rejected_suspects)],
            "fct_confirmation_rate_raw": float(fct_summary.get("fct_confirmation_rate_raw", 0.0)),
            "fct_confirmation_rate": float(fct_summary.get("fct_confirmation_rate", 0.0)),
            "fct_confirmed_overlap_with_malicious": [
                int(client_id) for client_id in fct_summary.get("fct_confirmed_overlap_with_malicious", [])
            ],
            "fct_confirmed_precision": (
                None
                if fct_summary.get("fct_confirmed_precision") is None
                else float(fct_summary.get("fct_confirmed_precision"))
            ),
            "fct_confirmed_recall": (
                None
                if fct_summary.get("fct_confirmed_recall") is None
                else float(fct_summary.get("fct_confirmed_recall"))
            ),
            "normal_clients": [
                int(client_id)
                for client_id in self.selected_clients
                if int(client_id) not in set(int(cid) for cid in self.current_suspicious_set)
            ],
            "applied_suspicious_clients": sorted(int(client_id) for client_id in effective_applied_clients),
            "consensus_suspicious_clients": list(self.current_consensus_candidates),
            "reweighting_gate_reason": str(reweighting_gate_reason),
            "reweight_require_confirmed_suspects": bool(reweight_require_confirmed_suspects),
            "reweight_require_confirmed_min_count": int(reweight_require_confirmed_min_count),
            "reweight_confirmed_gate_active": bool(reweight_confirmed_gate_active),
            "reweight_confirmed_ready": bool(reweight_confirmed_ready),
            "reweight_fct_quality_gate_active": bool(reweight_fct_quality_gate_active),
            "reweight_fct_quality_ready": bool(reweight_fct_quality_ready),
            "reweight_fct_min_confirmation_rate": float(reweight_fct_min_confirmation_rate),
            "reweight_fct_min_mean_z": float(reweight_fct_min_mean_z),
            "reweight_fct_min_mean_diff": float(reweight_fct_min_mean_diff),
            "reweight_fct_max_clean_pool_contamination_risk": float(
                reweight_fct_max_clean_pool_contamination_risk
            ),
            "reweight_fct_confirmation_rate": float(fct_confirmation_rate),
            "reweight_fct_confirmed_mean_z": float(fct_confirmed_mean_z),
            "reweight_fct_confirmed_mean_diff": float(fct_confirmed_mean_diff),
            "reweight_fct_clean_pool_contamination_risk": float(fct_contamination_risk),
            "reweighting_applied": bool(reweighting_applied_for_logging),
            "risk_scores_selected_clients": risk_scores_selected,
            "sampling_probability_selected_clients": sampling_probability_selected,
            "confirmed_risk_supported_groups": [int(group_id) for group_id in confirmed_risk_supported_groups],
            "risk_aware_enabled": bool(self.current_risk_sampling_enabled),
            "risk_activation_reason": str(self.current_risk_activation_reason),
            "risk_aware_source": str(risk_stats.get("risk_aware_source", self.dgc_repair.config.get("risk_aware_source", "raw"))),
            "risk_scores_by_client": {int(client_id): float(value) for client_id, value in risk_scores_top20.items()},
            "sampling_probabilities_by_client": {
                int(client_id): float(value) for client_id, value in sampling_probability_top20.items()
            },
            "risk_confirmed_count": int(risk_stats.get("risk_confirmed_count", len(risk_confirmed_suspects))),
            "risk_confirmed_input_count": int(len(risk_confirmed_suspects)),
            "risk_signal_mean": float(risk_stats.get("risk_signal_mean", 0.0)),
            "risk_signal_min": float(risk_stats.get("risk_signal_min", 0.0)),
            "risk_signal_max": float(risk_stats.get("risk_signal_max", 0.0)),
            "risk_update_applied": bool(risk_stats.get("risk_update_applied", False)),
            "risk_update_skip_reason": str(risk_stats.get("risk_update_skip_reason", "none")),
            "risk_recent_raw_candidate_precision_mean": (
                None if recent_raw_candidate_precision_mean is None else float(recent_raw_candidate_precision_mean)
            ),
            "risk_recent_raw_proxy_cv": (
                None if recent_raw_proxy_cv is None else float(recent_raw_proxy_cv)
            ),
            "risk_raw_candidate_precision_current": (
                None if raw_precision_current is None else float(raw_precision_current)
            ),
            "gate_by_group": {
                int(group_id): float(value)
                for group_id, value in gate_stats.get("gate_values", {}).items()
            },
            "confirmed_gate_by_group": {
                int(group_id): float(value)
                for group_id, value in gate_stats.get("confirmed_gate_values", {}).items()
            },
            "confirmed_gate_enabled": bool(
                gate_stats.get("confirmed_gate_enabled", self.dgc_repair.config.get("confirmed_gate_enable", False))
            ),
            "confirmed_gate_mean": float(gate_stats.get("confirmed_gate_mean", 1.0)),
            "confirmed_gate_min": float(gate_stats.get("confirmed_gate_min", 1.0)),
            "confirmed_gate_threshold": float(
                gate_stats.get("confirmed_gate_threshold", self.dgc_repair.config.get("confirmed_gate_threshold", 0.20))
            ),
            "confirmed_gate_beta": float(
                gate_stats.get("confirmed_gate_beta", self.dgc_repair.config.get("confirmed_gate_beta", 8.0))
            ),
            "confirmed_gate_reason": str(gate_stats.get("confirmed_gate_reason", "disabled")),
            "confirmed_gate_reject_count": int(
                sum(
                    1
                    for _, value in gate_stats.get("confirmed_gate_values", {}).items()
                    if float(value) <= 0.5
                )
            ),
            "confirmed_abnormal_ratio": float(gate_stats.get("confirmed_abnormal_ratio", 0.0)),
            "p_hat": float(p_hat),
            "p_hat_raw": float(p_hat_raw),
            "p_hat_corrected": float(p_hat_corrected),
            "p_hat_predicted": float(p_hat_predicted),
            "p_hat_predicted_raw": float(p_hat_raw),
            "p_hat_predicted_corrected": float(p_hat_corrected),
            "p_hat_oracle": float(oracle_p_hat) if oracle_p_hat is not None else None,
            "p_hat_source": (
                "oracle"
                if oracle_override_active
                else ("dense" if dense_mode_active else ("corrected" if p_hat_correction_factor_used > 1.0 else "predicted"))
            ),
            "cleaning_p_hat": float(self.current_cleaning_p_hat),
            "cleaning_confidence": float(self.current_cleaning_confidence),
            "cleaning_policy_version": str(self.config.get("cleaning_policy_version", "v2_continuous_tiered")),
            "cleaning_accept_score": 0.0,
            "cleaning_tier": "none",
            "cleaning_accept_threshold_weak": float(self.config.get("cleaning_accept_threshold_weak", 0.35)),
            "cleaning_accept_threshold_medium": float(self.config.get("cleaning_accept_threshold_medium", 0.55)),
            "cleaning_accept_margin_to_weak": 0.0,
            "cleaning_accept_margin_to_medium": 0.0,
            "cleaning_reject_reason": "none",
            "clean_candidate_available": bool(len(candidate_group_ids) > 0),
            "clean_candidate_count": int(len(candidate_group_ids)),
            "clean_candidate_score_best": (
                float(max(candidate_group_scores.values()))
                if len(candidate_group_scores) > 0
                else 0.0
            ),
            "clean_accept_raw_score": 0.0,
            "clean_accept_block_reason": "not_evaluated",
            "clean_accept_block_flags": self._default_clean_accept_block_flags(),
            "clean_accept_final": 0.0,
            "clean_accept_path": "not_evaluated",
            "repair_candidate_allowed": True,
            "repair_candidate_block_reason": "allowed",
            "repair_candidate_require_confirmation": bool(
                self.config.get("repair_candidate_require_confirmation", True)
            ),
            "repair_candidate_min_safe_acc": float(
                self.config.get("repair_candidate_min_safe_acc", 0.45)
            ),
            "repair_candidate_shadow_only": bool(
                self.config.get("repair_candidate_shadow_only", False)
            ),
            "cleaned_candidate_allowed": True,
            "cleaned_candidate_block_reason": "none",
            "cleaned_candidate_block_no_confirmed": False,
            "cleaned_candidate_block_safe_model_weak": False,
            "cleaned_candidate_block_repair_confidence_low": False,
            "candidate_source": str(candidate_source),
            "candidate_from_hist_keep_bridge": bool(candidate_from_hist_keep_bridge),
            "candidate_history_soft_override": bool(candidate_history_soft_override_applied),
            "cts_z_value": float(cts_z_score),
            "cts_z_threshold": float(cts_z_thr),
            "cts_z_threshold_effective": float(cts_z_thr),
            "cts_score_threshold_effective": float(cts_score_thr),
            "cts_consistency_threshold_effective": float(cts_consistency_thr),
            "cts_z_mean": float(cts_z_mean),
            "cts_z_std": float(cts_z_std),
            "cts_z_min": float(cts_z_min),
            "cts_z_max": float(cts_z_max),
            "cts_diff_value": float(cts_diff_value),
            "cts_tsc_value": float(cts_tsc_value),
            "cts_dcbd_score_value": float(cts_dcbd_score_value),
            "cts_dcbd_z_value": float(cts_dcbd_z_value),
            "cts_reliability_consistency": float(reliability_consistency_value),
            "cts_reliability_consistency_source": str(reliability_consistency_source),
            "cts_signal_reliable": bool(cts_signal_reliable),
            "cts_signal_unreliable_reason": str(cts_signal_unreliable_reason),
            "cts_signal_reliable_rule_path": str(cts_signal_reliable_rule_path),
            "cts_hist_keep_rule_path": str(cts_hist_keep_rule_path),
            "dcbd_absolute_signal_priority": bool(dcbd_absolute_signal_priority),
            "cts_distribution_collapsed": bool(cts_distribution_collapsed),
            "cts_distribution_collapse_reason": str(cts_distribution_collapse_reason),
            "effective_rho_planned": 0.0,
            "effective_rho": 0.0,
            "effective_projection_planned": 0.0,
            "effective_projection_applied": 0.0,
            "effective_reweighting_applied": 0.0,
            "effective_distillation_steps": 0,
            "force_weak_cleaning": False,
            "selection_forced_cleaning": False,
            "hist_kept_group_ids": [int(group_id) for group_id in hist_kept_group_ids],
            "candidate_group_ids": [int(group_id) for group_id in candidate_group_ids],
            "candidate_group_scores": {
                int(group_id): float(score)
                for group_id, score in candidate_group_scores.items()
            },
            "candidate_group_rank_scores": {
                int(group_id): float(score)
                for group_id, score in candidate_group_rank_scores.items()
            },
            "candidate_selected_group_id": int(candidate_selected_group_id),
            "candidate_selected_score": float(candidate_selected_score),
            "candidate_selected_rank_score": float(candidate_selected_rank_score),
            "candidate_selected_consistency": (
                float(candidate_selected_consistency)
                if candidate_selected_consistency is not None
                else None
            ),
            "candidate_topk_consistency_mean": (
                float(candidate_topk_consistency_mean)
                if candidate_topk_consistency_mean is not None
                else None
            ),
            "hist_keep_candidate_drop_reason": str(hist_keep_candidate_drop_reason),
            "hist_keep_count": int(len(hist_kept_group_ids)),
            "hist_drop_count_by_z": int(
                1 if ((not history_accumulated) and ("cts_z_too_low" in str(cts_hist_keep_rule_path))) else 0
            ),
            "hist_drop_count_by_consistency": int(
                1 if ((not history_accumulated) and ("consistency" in str(cts_signal_unreliable_reason))) else 0
            ),
            "hist_drop_count_by_reliability": int(
                1 if ((not history_accumulated) and (not cts_signal_reliable)) else 0
            ),
            "cts_bi_tail_right_count": int(right_anomalous_count),
            "cts_bi_tail_left_count": int(left_anomalous_count),
            "cts_raw_value": float(cts_raw_value),
            "cts_scale": float(mad_sigma_effective),
            "cleaning_projection_norm_before": 0.0,
            "cleaning_projection_norm_after": 0.0,
            "clean_model_delta_norm": 0.0,
            "cleaning_damage_proxy": 0.0,
            "cleaning_benefit_proxy": 0.0,
            "b_tilde_update": dict(btilde_update_event),
            "b_tilde_diagnostics": dict(self.B_tilde_diagnostics),
            "b_tilde_updated_at": int(self.B_tilde_updated_at),
            "degenerate": bool(degenerate),
            "dense_mode_active": bool(dense_mode_active),
            "dense_mode_p_hat": float(dense_mode_p_hat if dense_mode_active else 0.0),
            "clean_acc_before": metadata.get("clean_acc_before"),
            "asr_before": metadata.get("asr_before"),
            "cts_debug": {
                "cts_mode_requested": str(detection.aux_stats.get("cts_mode_requested", "raw")),
                "cts_mode_effective": str(detection.aux_stats.get("cts_mode_effective", "raw")),
                "dcbd_tsc_hard_gate_enable": bool(dcbd_tsc_hard_gate_enable),
                "dcbd_tsc_hard_gate_min": float(dcbd_tsc_hard_gate_min),
                "dcbd_tsc_hard_gate_use_abs": bool(dcbd_tsc_hard_gate_use_abs),
                "dcbd_tsc_hard_gate_apply_anomalous": bool(dcbd_tsc_hard_gate_apply_anomalous),
                "dcbd_tsc_hard_gate_apply_hist_keep": bool(dcbd_tsc_hard_gate_apply_hist_keep),
                "dcbd_tsc_hard_gate_apply_candidate": bool(dcbd_tsc_hard_gate_apply_candidate),
                "raw_anomalous_groups": [int(group_id) for group_id in raw_anomalous_groups],
                "tsc_hard_gate_filtered_groups": [int(group_id) for group_id in tsc_hard_gate_filtered_groups],
                "mdbf_fallback_reason": str(detection.aux_stats.get("mdbf_fallback_reason", "none")),
                "mdbf_alpha": float(detection.aux_stats.get("mdbf_alpha", 0.7)),
                "mdbf_percentile": float(detection.aux_stats.get("mdbf_percentile", 75.0)),
                "candidate_classes": detection.aux_stats.get("candidate_classes", []),
                "candidate_score_median": detection.aux_stats.get("candidate_score_median", []),
                "raw_cts_target": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("raw_cts_target", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "cts_per_class": {
                    int(group_id): [
                        float(value)
                        for value in (
                            detection.aux_stats.get("cts_per_class", {}).get(
                                int(group_id),
                                detection.aux_stats.get("cts_per_class", {}).get(str(int(group_id)), []),
                            )
                            if isinstance(detection.aux_stats.get("cts_per_class", {}), Mapping)
                            else []
                        )
                    ]
                    for group_id in ordered_group_ids
                },
                "cts_trig_per_class": {
                    int(group_id): [
                        float(value)
                        for value in (
                            detection.aux_stats.get("cts_trig_per_class", {}).get(
                                int(group_id),
                                detection.aux_stats.get("cts_trig_per_class", {}).get(str(int(group_id)), []),
                            )
                            if isinstance(detection.aux_stats.get("cts_trig_per_class", {}), Mapping)
                            else []
                        )
                    ]
                    for group_id in ordered_group_ids
                },
                "cts_clean_per_class": {
                    int(group_id): [
                        float(value)
                        for value in (
                            detection.aux_stats.get("cts_clean_per_class", {}).get(
                                int(group_id),
                                detection.aux_stats.get("cts_clean_per_class", {}).get(str(int(group_id)), []),
                            )
                            if isinstance(detection.aux_stats.get("cts_clean_per_class", {}), Mapping)
                            else []
                        )
                    ]
                    for group_id in ordered_group_ids
                },
                "tsc_target": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("tsc_target", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "cse_target": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("cse_target", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "mdbf_target": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("mdbf_target", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "group_behavior_score": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("group_behavior_score", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "dcbd_score": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("dcbd_score", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "anomaly_decision": {
                    int(group_id): bool(
                        detection.aux_stats.get("anomaly_decision", {}).get(
                            int(group_id),
                            detection.aux_stats.get("anomaly_decision", {}).get(str(int(group_id)), False),
                        )
                        if isinstance(detection.aux_stats.get("anomaly_decision", {}), Mapping)
                        else False
                    )
                    for group_id in ordered_group_ids
                },
                "dcbd_anomaly_decision": {
                    int(group_id): bool(
                        detection.aux_stats.get("dcbd_anomaly_decision", {}).get(
                            int(group_id),
                            detection.aux_stats.get("dcbd_anomaly_decision", {}).get(str(int(group_id)), False),
                        )
                        if isinstance(detection.aux_stats.get("dcbd_anomaly_decision", {}), Mapping)
                        else False
                    )
                    for group_id in ordered_group_ids
                },
                "dcbd_anomaly_z_score": {
                    int(group_id): float(
                        self._map_get_float(detection.aux_stats.get("dcbd_anomaly_z_score", {}), int(group_id), 0.0)
                    )
                    for group_id in ordered_group_ids
                },
                "consistency_by_group": {
                    int(group_id): float(consistency_map.get(group_id, 0.0))
                    for group_id in ordered_group_ids
                },
                "diff_group_scores": {
                    int(group_id): float(
                        detection.aux_stats.get("diff_group_scores", {}).get(
                            group_id,
                            detection.aux_stats.get("diff_group_scores", {}).get(str(group_id), float("nan")),
                        )
                    )
                    for group_id in ordered_group_ids
                },
                "cts_round_median": float(detection.aux_stats.get("cts_round_median", 0.0)),
                "anomaly_tail": str(detection.aux_stats.get("anomaly_tail", "right")),
                "mad_median": float(detection.aux_stats.get("mad_median", 0.0)),
                "mad_sigma": float(detection.aux_stats.get("mad_sigma", 0.0)),
                "mad_sigma_effective": float(mad_sigma_effective),
                "mad_sigma_floor": float(mad_sigma_floor),
                "mad_threshold": float(detection.aux_stats.get("mad_threshold", 0.0)),
                "dcbd_scores_all_groups": [
                    float(value) for value in detection.aux_stats.get("dcbd_scores_all_groups", [])
                ],
                "dcbd_mad_median": float(detection.aux_stats.get("dcbd_mad_median", 0.0)),
                "dcbd_mad_sigma": float(detection.aux_stats.get("dcbd_mad_sigma", 0.0)),
                "dcbd_threshold": float(detection.aux_stats.get("dcbd_threshold", 0.0)),
                "max_cts_score": float(max_cts_score),
                "max_abs_cts_score": float(max_abs_cts_score),
                "cts_score_norm": float(cts_score_norm),
                "cts_score_norm_threshold": float(cts_score_norm_threshold),
                "mean_consistency": float(mean_consistency),
                "reliability_consistency_value": float(reliability_consistency_value),
                "reliability_consistency_source": str(reliability_consistency_source),
                "cts_z_raw": float(cts_z_raw),
                "cts_z_score": float(cts_z_score),
                "cts_z_value": float(cts_z_score),
                "cts_z_clip": float(cts_z_clip),
                "cts_z_threshold": float(cts_z_thr),
                "cts_z_threshold_effective": float(cts_z_thr),
                "cts_score_threshold_effective": float(cts_score_thr),
                "cts_consistency_threshold_effective": float(cts_consistency_thr),
                "cts_z_mean": float(cts_z_mean),
                "cts_z_std": float(cts_z_std),
                "cts_z_min": float(cts_z_min),
                "cts_z_max": float(cts_z_max),
                "cts_diff_value": float(cts_diff_value),
                "cts_tsc_value": float(cts_tsc_value),
                "cts_dcbd_score_value": float(cts_dcbd_score_value),
                "cts_dcbd_z_value": float(cts_dcbd_z_value),
                "cts_signal_reliable": bool(cts_signal_reliable),
                "cts_signal_unreliable_reason": str(cts_signal_unreliable_reason),
                "cts_signal_reliable_rule_path": str(cts_signal_reliable_rule_path),
                "cts_reliability_consistency_ok": bool(consistency_ok),
                "cts_reliability_score_ok": bool(score_ok),
                "cts_reliability_z_ok": bool(z_ok),
                "candidate_selected_consistency": (
                    float(candidate_selected_consistency)
                    if candidate_selected_consistency is not None
                    else None
                ),
                "candidate_topk_consistency_mean": (
                    float(candidate_topk_consistency_mean)
                    if candidate_topk_consistency_mean is not None
                    else None
                ),
                "candidate_group_rank_scores": {
                    int(group_id): float(score)
                    for group_id, score in candidate_group_rank_scores.items()
                },
                "candidate_selected_rank_score": float(candidate_selected_rank_score),
                "cts_hist_keep_rule_path": str(cts_hist_keep_rule_path),
                "cts_distribution_collapsed": bool(cts_distribution_collapsed),
                "cts_distribution_collapse_reason": str(cts_distribution_collapse_reason),
                "cts_signal_bridge_enable": bool(cts_signal_bridge_enable),
                "cts_signal_bridge_min_candidate_count": int(cts_signal_bridge_min_candidate_count),
                "cts_signal_bridge_min_consistency": float(cts_signal_bridge_min_consistency),
                "cts_signal_bridge_min_score": float(cts_signal_bridge_min_score),
                "cts_signal_bridge_min_cts_z": float(cts_signal_bridge_min_cts_z),
                "cts_signal_bridge_min_rank_score": float(cts_signal_bridge_min_rank_score),
                "dcbd_absolute_signal_priority": bool(dcbd_absolute_signal_priority),
                "candidate_history_soft_override": bool(candidate_history_soft_override_applied),
                "right_anomalous_count": int(right_anomalous_count),
                "left_anomalous_count": int(left_anomalous_count),
                "dense_mode_stats": {key: float(value) for key, value in dense_mode_stats.items()},
                "p_hat_correction_mode": str(self.config.get("p_hat_correction_mode", "group_ratio")),
                "p_hat_correction_factor_used": float(p_hat_correction_factor_used),
                "current_p_hat_correction_factor_used": float(current_p_hat_correction_factor_used),
                "fallback_triggered": bool(detection.aux_stats.get("fallback_triggered", False)),
                "fallback_selected_groups": detection.aux_stats.get("fallback_selected_groups", []),
                "fallback_score_gap": float(detection.aux_stats.get("fallback_score_gap", 0.0)),
                "fallback_top_score": float(detection.aux_stats.get("fallback_top_score", 0.0)),
                "group_eval_scale_by_group": {
                    int(group_id): float(group_eval_scale_map.get(int(group_id), 1.0))
                    for group_id in ordered_group_ids
                },
                "group_eval_norm_by_group": {
                    int(group_id): float(group_eval_norm_map.get(int(group_id), 0.0))
                    for group_id in ordered_group_ids
                },
                "global_update_norm": float(global_update_norm),
                "history_accumulated": bool(history_accumulated),
                "history_skip_reason": str(history_skip_reason),
                "history_gate_enabled": bool(history_gate_enabled),
                "history_min_anomalous_groups": int(history_min_anomalous),
                "history_min_cts_z": float(history_min_cts_z),
                "history_min_signal_std": float(history_min_signal_std),
                "pre_anomalous_group_count": int(pre_anomalous_group_count),
                "pre_cts_z": float(pre_cts_z),
                "pre_signal_std": float(pre_signal_std),
                "reprobe_applied": bool(reprobe_applied),
                "reprobe_reason": str(reprobe_reason),
                "reprobe_boost_used": float(reprobe_boost_used),
                "reprobe_max_abs_before": float(reprobe_max_abs_before),
                "reprobe_std_before": float(reprobe_std_before),
                "reprobe_max_abs_after": float(reprobe_max_abs_after),
                "reprobe_std_after": float(reprobe_std_after),
            },
            "s3_debug": {
                "num_groups_total": int(self.A_matrix.shape[0]),
                "num_groups_current": int(current_A.shape[0]),
                "raw_positive_ratio": float(current_s3_aux.get("raw_positive_ratio", 0.0)),
                "p_hat_raw": float(p_hat_raw),
                "p_hat_corrected": float(p_hat_corrected),
                "current_p_hat_raw": float(current_p_hat_raw),
                "current_p_hat_corrected": float(current_p_hat_corrected),
                "initial_suspicious_clients": [int(client_id) for client_id in initial_suspicious_clients],
                "confirmed_suspects": [int(client_id) for client_id in confirmed_suspects],
                "risk_confirmed_suspects": [int(client_id) for client_id in risk_confirmed_suspects],
                "risk_confirmed_filter_reason": str(risk_confirmed_filter_reason),
                "confirmation_rejected_suspects": [int(client_id) for client_id in confirmation_rejected_suspects],
                "fct_fallback_reason": str(fct_fallback_reason),
                "fct_stats_by_client": {
                    int(client_id): {
                        "fct_mean": float(stats.get("fct_mean", 0.0)),
                        "fct_std": float(stats.get("fct_std", 0.0)),
                        "fct_z": float(stats.get("fct_z", 0.0)),
                        "fct_mean_raw": float(stats.get("fct_mean_raw", 0.0)),
                        "fct_std_raw": float(stats.get("fct_std_raw", 0.0)),
                        "fct_z_std": float(stats.get("fct_z_std", 0.0)),
                        "fct_z_mad": float(stats.get("fct_z_mad", 0.0)),
                        "fct_winsor_applied": bool(stats.get("fct_winsor_applied", False)),
                        "fct_confirmed_raw": bool(stats.get("fct_confirmed_raw", False)),
                        "fct_stability_pass_count": int(stats.get("fct_stability_pass_count", 0)),
                        "num_pairs_requested": int(stats.get("num_pairs_requested", 0)),
                        "num_pairs_valid": int(stats.get("num_pairs_valid", 0)),
                        "confirmed": bool(stats.get("confirmed", False)),
                        "fallback_below_k_min": bool(stats.get("fallback_below_k_min", False)),
                    }
                    for client_id, stats in fct_stats_by_client.items()
                },
                "selected_scores": {
                    int(client_id): float(score)
                    for client_id, score in current_s3_aux.get("selected_scores", {}).items()
                },
                "selected_pos_rate": {
                    int(client_id): float(score)
                    for client_id, score in current_s3_aux.get("selected_pos_rate", {}).items()
                },
                "selected_neg_rate": {
                    int(client_id): float(score)
                    for client_id, score in current_s3_aux.get("selected_neg_rate", {}).items()
                },
                "selected_contrast_scores": {
                    int(client_id): float(score)
                    for client_id, score in current_s3_aux.get("selected_contrast_scores", {}).items()
                },
                "pos_neg_contrast_enabled": bool(current_s3_aux.get("pos_neg_contrast_enabled", False)),
                "pos_neg_contrast_reason": str(current_s3_aux.get("pos_neg_contrast_reason", "none")),
                "pos_neg_contrast_lambda": float(current_s3_aux.get("pos_neg_contrast_lambda", 0.0)),
                "pos_neg_contrast_blend": float(current_s3_aux.get("pos_neg_contrast_blend", 0.0)),
                "stabilized_selected_scores": {
                    int(client_id): float(stabilized_current_scores.get(client_id, 0.0))
                    for client_id in current_suspicious_set
                },
                "refined_reliable_scores": {
                    int(client_id): float(refined_reliable_scores.get(client_id, 0.0))
                    for client_id in current_suspicious_set
                },
                "weighted_y": current_s3_aux.get("weighted_y", []),
                "signal_center": float(current_s3_aux.get("signal_center", 0.0)),
                "signal_scale": float(current_s3_aux.get("signal_scale", 1.0)),
                "negative_cutoff": float(current_s3_aux.get("negative_cutoff", 0.0)),
                "positive_mass": float(current_s3_aux.get("positive_mass", 0.0)),
                "negative_mass": float(current_s3_aux.get("negative_mass", 0.0)),
                "positive_group_count": int(current_s3_aux.get("positive_group_count", 0)),
                "negative_group_count": int(current_s3_aux.get("negative_group_count", 0)),
                "positive_mean": float(current_s3_aux.get("positive_mean", 0.0)),
                "negative_mean": float(current_s3_aux.get("negative_mean", 0.0)),
                "historical_selected_scores": {
                    int(client_id): float(score)
                    for client_id, score in s3_aux.get("selected_scores", {}).items()
                },
                "top_client_scores": [
                    {"client_id": int(client_id), "score": float(score)}
                    for client_id, score in top_client_scores
                ],
                "top_score": float(top_score),
                "second_score": float(second_score),
                "score_gap": float(score_gap),
                "score_ratio": float(score_ratio),
                "top_frequency": int(top_frequency),
                "top_persistence": float(top_persistence),
                "top_historical_mean": float(stability_stats.get("top_historical_mean", 0.0)),
                "stability_gate_enabled": bool(stability_stats.get("stability_gate_enabled", False)),
                "stability_gate_applied": bool(stability_stats.get("stability_gate_applied", False)),
                "stability_gate_weak_scale": float(stability_stats.get("stability_gate_weak_scale", 1.0)),
                "stability_evidence_by_groups": bool(stability_stats.get("stability_evidence_by_groups", False)),
                "stability_evidence_by_cts": bool(stability_stats.get("stability_evidence_by_cts", False)),
                "stability_evidence_ok": bool(stability_stats.get("stability_evidence_ok", False)),
                "raw_top_score": float(stability_stats.get("raw_top_score", 0.0)),
                "raw_score_gap": float(stability_stats.get("raw_score_gap", 0.0)),
                "raw_score_ratio": float(stability_stats.get("raw_score_ratio", 0.0)),
                "raw_anomalous_group_count": int(stability_stats.get("raw_anomalous_group_count", 0)),
                "projected_history_strict_eligible": bool(projected_history_strict_eligible),
                "projected_history_relaxed_eligible": bool(projected_history_relaxed_eligible),
                "projected_history_eligible": bool(projected_history_eligible),
                "projected_historical_anomalous_audits": int(projected_historical_anomalous_audits),
                "projected_top_frequency": int(projected_top_frequency),
                "projected_top_persistence": float(projected_top_persistence),
                "projected_top_historical_mean": float(projected_top_historical_mean),
                "stability_reliable": bool(stability_reliable),
                "localization_reliable_strict": bool(localization_reliable_strict),
                "ranking_reliable": bool(ranking_reliable),
                "consensus_signal_reliable": bool(consensus_signal_reliable),
                "raw_localization_reliable": bool(raw_localization_reliable),
                "localization_confidence_score": float(localization_confidence_score),
                "reliable_consensus_overlap": int(reliable_consensus_overlap),
                "current_consensus_overlap": int(current_consensus_overlap),
                "consensus_backed_reweight_supported": bool(consensus_backed_reweight_supported),
                "strong_signal_without_consensus": bool(strong_signal_without_consensus),
                "current_reweight_supported": bool(current_reweight_supported),
                "retained_consensus_overlap": int(retained_consensus_overlap),
                "retained_reweight_supported": bool(retained_reweight_supported),
                "strong_reliable_override": bool(strong_reliable_override),
                "weak_reliable_override": bool(weak_reliable_override),
                "ranking_reliable_override": bool(ranking_reliable_override),
                "reliable_score_eligible": bool(reliable_score_eligible),
                "localization_reliable": bool(raw_localization_reliable),
    
                "reweighting_ready": bool(reliable_reweighting_ready),
                "provisional_reweighting_ready": bool(provisional_reweighting_ready),
                "provisional_reweighting_applied": bool(
                    reweighting_enabled and provisional_reweighting_applied
                ),
                "ranking_reweighting_ready": bool(ranking_reweighting_ready),
                "ranking_reweight_consensus_supported": bool(ranking_reweight_consensus_supported),
                "ranking_reweight_strong_signal_override": bool(ranking_reweight_strong_signal_override),
                "ranking_reweight_local_signal_override": bool(ranking_reweight_local_signal_override),
                "ranking_reweight_support_ok": bool(ranking_reweight_support_ok),
                "ranking_reweighting_applied": bool(
                    reweighting_enabled and ranking_reweighting_applied
                ),
                "consensus_reweighting_ready": bool(consensus_reweighting_ready),
                "consensus_reweighting_applied": bool(
                    reweighting_enabled and consensus_reweighting_applied
                ),
                "retained_reweighting_applied": bool(
                    reweighting_enabled and retained_reweighting_applied
                ),
                "fallback_reweighting_ready": bool(fallback_reweighting_ready),
                "fallback_reweighting_applied": bool(
                    reweighting_enabled and fallback_reweighting_applied
                ),
                "history_quality_ok": bool(history_quality_ok),
                "reweight_quality_ok": bool(reweight_quality_ok),
                "consensus_candidate_count": int(consensus_candidate_count),
                "history_quality_gate_enabled": bool(history_quality_gate_enabled),
                "reweight_quality_gate_enabled": bool(reweight_quality_gate_enabled),
                "history_bootstrap_signal": bool(history_bootstrap_signal),
                "consensus_history_strict_eligible": bool(consensus_history_strict_eligible),
                "consensus_history_relaxed_eligible": bool(consensus_history_relaxed_eligible),
                "consensus_history_eligible": bool(consensus_history_eligible),
                "historical_anomalous_audits": int(self._historical_anomalous_audits),
                "historical_reliable_audits": int(self._historical_reliable_audits),
                "consensus_candidates": [
                    {"client_id": int(client_id), "score": float(consensus_scores.get(client_id, 0.0))}
                    for client_id in self.current_consensus_candidates
                ],
                "reweighting_enabled": bool(reweighting_enabled),
            "reweighting_applied": bool(reweighting_applied_for_logging),
            "reweighting_gate_reason": str(reweighting_gate_reason),
            "gate_mean": float(gate_stats.get("gate_mean", 1.0)),
            "gate_min": float(gate_stats.get("gate_min", 1.0)),
            "gate_reject_count": int(gate_stats.get("gate_reject_count", 0)),
            "gate_floor_activated": bool(gate_stats.get("gate_floor_activated", False)),
            "effective_ratio": float(gate_stats.get("effective_ratio", 1.0)),
            "gate_by_group": {
                int(group_id): float(value)
                for group_id, value in gate_stats.get("gate_values", {}).items()
            },
            "confirmed_gate_by_group": {
                int(group_id): float(value)
                for group_id, value in gate_stats.get("confirmed_gate_values", {}).items()
            },
            "confirmed_gate_enabled": bool(
                gate_stats.get("confirmed_gate_enabled", self.dgc_repair.config.get("confirmed_gate_enable", False))
            ),
            "confirmed_gate_mean": float(gate_stats.get("confirmed_gate_mean", 1.0)),
            "confirmed_gate_min": float(gate_stats.get("confirmed_gate_min", 1.0)),
            "confirmed_gate_threshold": float(
                gate_stats.get("confirmed_gate_threshold", self.dgc_repair.config.get("confirmed_gate_threshold", 0.20))
            ),
            "confirmed_gate_beta": float(
                gate_stats.get("confirmed_gate_beta", self.dgc_repair.config.get("confirmed_gate_beta", 8.0))
            ),
            "confirmed_gate_reason": str(gate_stats.get("confirmed_gate_reason", "disabled")),
            "confirmed_abnormal_ratio": float(gate_stats.get("confirmed_abnormal_ratio", 0.0)),
            "risk_score_mean": float(risk_stats.get("risk_score_mean", float(self.dgc_repair.risk_scores.mean().item()))),
            "risk_score_max": float(risk_stats.get("risk_score_max", float(self.dgc_repair.risk_scores.max().item()))),
            "risk_aware_source": str(risk_stats.get("risk_aware_source", self.dgc_repair.config.get("risk_aware_source", "raw"))),
            "risk_confirmed_count": int(risk_stats.get("risk_confirmed_count", len(confirmed_set))),
            "risk_signal_mean": float(risk_stats.get("risk_signal_mean", 0.0)),
            "risk_signal_min": float(risk_stats.get("risk_signal_min", 0.0)),
            "risk_signal_max": float(risk_stats.get("risk_signal_max", 0.0)),
            "risk_update_applied": bool(risk_stats.get("risk_update_applied", False)),
            "risk_update_skip_reason": str(risk_stats.get("risk_update_skip_reason", "none")),
            "risk_scores_selected_clients": {
                int(client_id): float(score) for client_id, score in risk_scores_selected.items()
            },
            "sampling_probability_selected_clients": {
                int(client_id): float(prob) for client_id, prob in sampling_probability_selected.items()
            },
            "confirmed_risk_supported_groups": [int(group_id) for group_id in confirmed_risk_supported_groups],
            "runtime_ms_historical": localization_runtime_ms_historical,
                "runtime_ms_current": localization_runtime_ms_current,
                "runtime_ms_wrapper": localization_runtime_ms_wrapper,
                "runtime_ms_total": localization_runtime_ms_total,
                "localization_signal_source_requested": str(localization_signal_source_requested),
                "localization_signal_source_effective": str(localization_signal_source_effective),
                "reweight_require_confirmed_suspects": bool(reweight_require_confirmed_suspects),
                "reweight_require_confirmed_min_count": int(reweight_require_confirmed_min_count),
                "reweight_confirmed_gate_active": bool(reweight_confirmed_gate_active),
                "reweight_confirmed_ready": bool(reweight_confirmed_ready),
                "reweight_fct_quality_gate_active": bool(reweight_fct_quality_gate_active),
                "reweight_fct_quality_ready": bool(reweight_fct_quality_ready),
                "reweight_fct_min_confirmation_rate": float(reweight_fct_min_confirmation_rate),
                "reweight_fct_min_mean_z": float(reweight_fct_min_mean_z),
                "reweight_fct_min_mean_diff": float(reweight_fct_min_mean_diff),
                "reweight_fct_max_clean_pool_contamination_risk": float(
                    reweight_fct_max_clean_pool_contamination_risk
                ),
                "reweight_fct_confirmation_rate": float(fct_confirmation_rate),
                "reweight_fct_confirmed_mean_z": float(fct_confirmed_mean_z),
                "reweight_fct_confirmed_mean_diff": float(fct_confirmed_mean_diff),
                "reweight_fct_clean_pool_contamination_risk": float(fct_contamination_risk),
            },
        }
        self.audit_history.append(audit_log)

        decision = C3SGuardDecision(
            round_idx=round_idx,
            audit_triggered=True,
            aggregation_action="inspect",
            detection=detection,
            localization=localization,
            cleaning=None,
            aux_stats={
                "audit_log": audit_log,
                "backup_members": deepcopy(self._backup_members),
            },
        )
        # Release heavy per-group update caches as soon as audit outputs are materialized.
        self.current_group_updates = OrderedDict()
        self.current_group_clients = {}
        self.current_dropout_counts = []
        return decision

    def _get_probe_set(self, clean_dataloader: Optional[DataLoader] = None) -> torch.Tensor:
        if self._probe_set is not None:
            return self._probe_set.to(self.device)
        probe_set = self.cts_intent._resolve_probe_set(
            clean_dataloader=clean_dataloader or self.clean_dataloader,
            metadata={},
        )
        self._probe_set = probe_set.detach().cpu()
        return probe_set.to(self.device)

    def _collect_dropout_history(self) -> List[int]:
        dropout_history = []
        for _, group_assignments in enumerate(self.s3_loc._group_assignments):
            dropout_history.append(len(group_assignments))
        return dropout_history

    @staticmethod
    def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _clip01(value: float) -> float:
        return float(min(1.0, max(0.0, float(value))))

    @staticmethod
    def _default_clean_accept_block_flags() -> Dict[str, bool]:
        return {
            "no_candidate_after_hist_filter": False,
            "candidate_exists_but_score_too_low": False,
            "blocked_by_margin_rule": False,
            "blocked_by_policy_rule": False,
            "blocked_by_safe_teacher_unready": False,
            "blocked_by_reject_override": False,
            "blocked_by_consistency_rule": False,
            "blocked_by_cts_signal_unreliable": False,
            "blocked_by_fallback_rule": False,
        }

    def _normalize_clean_accept_block_reason(
        self,
        *,
        base_reason: str,
        candidate_available: bool,
        candidate_score_best: float,
        cts_signal_reliable: bool,
        safe_teacher_ready: bool,
        reject_override: bool,
    ) -> Tuple[str, Dict[str, bool]]:
        reason = str(base_reason or "none").strip().lower()
        score_floor = float(self.config.get("clean_candidate_score_floor", 1e-6))
        allowed_reasons = set(self._default_clean_accept_block_flags().keys())

        normalized_reason = reason
        if not candidate_available:
            normalized_reason = "no_candidate_after_hist_filter"
        elif reject_override or reason.startswith("blocked_by_reject") or reason in {"reject", "rollback"}:
            normalized_reason = "blocked_by_reject_override"
        elif (not safe_teacher_ready) and (
            ("safe" in reason) or reason in {"cleaned_update_unavailable", "cleaning_result_unavailable"}
        ):
            normalized_reason = "blocked_by_safe_teacher_unready"
        elif reason in {"consensus_too_weak", "blocked_by_consistency", "consistency_too_low"}:
            normalized_reason = "blocked_by_consistency_rule"
        elif ("fallback" in reason) or reason in {"fallback_waiting", "fallback_probe_too_low", "fallback_disabled"}:
            normalized_reason = "blocked_by_fallback_rule"
        elif ("margin" in reason) or reason in {"predicted_margin_too_low", "blocked_by_margin"}:
            normalized_reason = "blocked_by_margin_rule"
        elif (
            (not cts_signal_reliable and reason in {"none", "cleaning_blocked_unknown", "repair_blocked_unknown"})
            or ("unreliable" in reason)
            or reason in {"localization_unreliable", "cts_signal_unreliable"}
        ):
            normalized_reason = "blocked_by_cts_signal_unreliable"
        elif (
            candidate_score_best <= score_floor
            and reason in {"predicted_benefit_too_low", "none", "cleaning_blocked_unknown", "repair_blocked_unknown"}
        ):
            normalized_reason = "candidate_exists_but_score_too_low"
        elif reason in {
            "cleaning_disabled_by_config",
            "target_unavailable",
            "no_audit_triggered",
            "candidate_count_too_small",
            "cleaning_result_unavailable",
            "cleaned_update_unavailable",
            "repair_blocked_unknown",
            "cleaning_blocked_unknown",
            "blocked_by_policy",
            "blocked_by_raw",
            "blocked_by_raw_fallback",
            "blocked_by_noop",
        }:
            normalized_reason = "blocked_by_policy_rule"
        elif normalized_reason not in allowed_reasons:
            normalized_reason = "blocked_by_policy_rule"

        flags = self._default_clean_accept_block_flags()
        if normalized_reason in flags:
            flags[normalized_reason] = True
        return normalized_reason, flags

    def _correct_p_hat(
        self,
        p_hat_raw: float,
        *,
        selected_clients_count: int,
        group_sizes: Sequence[int],
    ) -> Tuple[float, float]:
        """Map group-level anomaly rate to a client-level p_hat with configurable correction."""
        raw = float(max(0.0, p_hat_raw))
        mode = str(self.config.get("p_hat_correction_mode", "group_ratio")).strip().lower()
        factor_extra = float(max(self.config.get("p_hat_correction_factor", 1.0), 0.0))
        corrected_max = float(max(self.config.get("p_hat_correction_max", 0.35), 0.0))
        if mode == "none":
            return float(min(raw, corrected_max)), 1.0

        valid_group_sizes = [float(size) for size in group_sizes if int(size) > 0]
        mean_group_size = float(np.mean(valid_group_sizes)) if valid_group_sizes else float(max(self.audit_group_size, 1))
        selected = float(max(int(selected_clients_count), 1))
        # Empirical correction: group-level anomaly prevalence tends to under-estimate
        # client-level malicious prevalence by approximately selected/group_size.
        auto_factor = max(1.0, selected / max(mean_group_size, 1e-6))
        auto_factor_cap = float(max(self.config.get("p_hat_correction_auto_factor_max", 3.0), 1.0))
        auto_factor = min(auto_factor, auto_factor_cap)
        total_factor = float(max(1.0, auto_factor * factor_extra))
        corrected = float(min(max(raw * total_factor, raw), corrected_max))
        return corrected, total_factor

    def _compute_cleaning_acceptance(
        self,
        *,
        p_hat: float,
        oracle_cleaning_rho: Optional[float],
        localization_reliable: bool,
        top_score: float,
        score_gap: float,
        score_ratio: float,
        consensus_count: int,
        suspicious_count: int,
    ) -> Dict[str, Any]:
        loc_flag = 1.0 if bool(localization_reliable) else 0.0
        top_ref = max(float(self.config.get("cleaning_accept_top_score_ref", 3.0)), 1e-6)
        gap_ref = max(float(self.config.get("cleaning_accept_score_gap_ref", 0.5)), 1e-6)
        ratio_ref = max(float(self.config.get("cleaning_accept_score_ratio_ref", 1.2)), 1.0 + 1e-6)
        suspicious_topk = max(1, int(self.config.get("suspicious_topk", 1)))

        score_strength = self._clip01(float(top_score) / top_ref)
        gap_strength = self._clip01(max(float(score_gap), 0.0) / gap_ref)
        ratio_strength = self._clip01(max(float(score_ratio) - 1.0, 0.0) / max(ratio_ref - 1.0, 1e-6))
        cand_support = self._clip01(float(consensus_count) / max(float(suspicious_count), 1.0))
        r_loc = self._clip01(0.70 * loc_flag + 0.30 * score_strength)
        r_cand = self._clip01(0.50 * score_strength + 0.25 * gap_strength + 0.15 * ratio_strength + 0.10 * cand_support)
        r_cons = self._clip01(float(consensus_count) / float(suspicious_topk))
        p_hat_norm = self._clip01(float(p_hat) / 0.10)
        min_suspicious_count = max(int(self.config.get("cleaning_accept_min_suspicious_count", 1)), 0)
        min_consensus_ratio = float(
            max(0.0, min(1.0, self.config.get("cleaning_accept_min_consensus_ratio", 0.10)))
        )
        require_consensus_gate = bool(self.config.get("cleaning_accept_require_consensus_gate", True))

        accept_score = (
            float(self.config.get("cleaning_accept_w_loc", 0.40)) * r_loc
            + float(self.config.get("cleaning_accept_w_cand", 0.30)) * r_cand
            + float(self.config.get("cleaning_accept_w_cons", 0.20)) * r_cons
            + float(self.config.get("cleaning_accept_w_phat", 0.10)) * p_hat_norm
        )
        accept_score = self._clip01(accept_score)

        weak_th = float(self.config.get("cleaning_accept_threshold_weak", 0.45))
        med_th = float(self.config.get("cleaning_accept_threshold_medium", 0.65))
        if float(accept_score) < weak_th:
            tier = "none"
        elif float(accept_score) < med_th:
            tier = "weak"
        else:
            tier = "medium"

        if oracle_cleaning_rho is not None:
            rho_val = float(max(0.0, min(1.0, oracle_cleaning_rho)))
            if rho_val >= float(self.config.get("cleaning_rho_min_strong", 0.15)):
                tier = "strong"
            elif rho_val >= float(self.config.get("cleaning_rho_min_medium", 0.08)):
                tier = "medium"
            elif rho_val > 0.0:
                tier = "weak"
            else:
                tier = "none"

        min_accept_p_hat = float(self.config.get("cleaning_min_p_hat_for_accept", 0.01))
        reject_reason = "none"
        if suspicious_count < min_suspicious_count:
            reject_reason = "candidate_count_too_small"
        elif tier == "none":
            if not localization_reliable:
                reject_reason = "localization_unreliable"
            elif require_consensus_gate and (r_cons < min_consensus_ratio):
                reject_reason = "consensus_too_weak"
            elif float(p_hat) < min_accept_p_hat:
                reject_reason = "p_hat_too_small"
            else:
                reject_reason = "predicted_benefit_too_low"

        return {
            "accept_score": float(accept_score),
            "tier": str(tier),
            "reject_reason": str(reject_reason),
            "weak_threshold": float(weak_th),
            "medium_threshold": float(med_th),
            "margin_to_weak": float(accept_score - weak_th),
            "margin_to_medium": float(accept_score - med_th),
            "r_loc": float(r_loc),
            "r_cand": float(r_cand),
            "r_cons": float(r_cons),
            "p_hat_norm": float(p_hat_norm),
            "score_strength": float(score_strength),
            "gap_strength": float(gap_strength),
            "ratio_strength": float(ratio_strength),
            "cand_support": float(cand_support),
            "require_consensus_gate": bool(require_consensus_gate),
            "min_consensus_ratio": float(min_consensus_ratio),
            "min_suspicious_count": int(min_suspicious_count),
        }

    def _estimate_dense_regime_p_hat(
        self,
        cts_scores: Sequence[float],
        positive_ratio: float,
        cts_z_score: float,
        mean_consistency: float,
        consensus_count: int,
        group_sizes: Sequence[int],
    ) -> Tuple[bool, float, Dict[str, float]]:
        dense_enabled = bool(self.config.get("dense_mode_enabled", True))
        if (not dense_enabled) or len(cts_scores) == 0:
            return False, 0.0, {
                "active": 0.0,
                "q_hard": float(positive_ratio),
                "q_soft": 0.0,
                "q_consensus": 0.0,
                "q_blend": float(positive_ratio),
                "group_size": float(self.audit_group_size),
            }

        positive_ratio_max = float(self.config.get("dense_mode_positive_ratio_max", 0.20))
        min_cts_z = float(self.config.get("dense_mode_min_cts_z", 2.0))
        min_consistency = float(self.config.get("dense_mode_min_consistency", 0.40))
        dense_active = bool(
            positive_ratio <= positive_ratio_max
            and (cts_z_score >= min_cts_z or mean_consistency >= min_consistency)
        )
        if not dense_active:
            return False, 0.0, {
                "active": 0.0,
                "q_hard": float(positive_ratio),
                "q_soft": 0.0,
                "q_consensus": 0.0,
                "q_blend": float(positive_ratio),
                "group_size": float(self.audit_group_size),
            }

        cts_array = np.asarray(cts_scores, dtype=np.float32)
        cts_array = np.nan_to_num(cts_array, nan=0.0, posinf=0.0, neginf=0.0)
        anchor_scale = float(self.config.get("dense_mode_anchor_scale", 1.25))
        robust_scale = max(float(np.std(cts_array)), anchor_scale, 1e-6)
        q_soft = float(np.mean(self._safe_sigmoid(cts_array / robust_scale)))

        suspicious_topk = max(1, int(self.config.get("suspicious_topk", 1)))
        q_consensus = float(
            min(1.0, max(0.0, float(consensus_count) / float(suspicious_topk)))
        )
        w_hard = float(self.config.get("dense_mode_hard_weight", 0.15))
        w_soft = float(self.config.get("dense_mode_soft_weight", 0.70))
        w_cons = float(self.config.get("dense_mode_consensus_weight", 0.15))
        w_sum = max(w_hard + w_soft + w_cons, 1e-6)
        q_blend = (w_hard * float(positive_ratio) + w_soft * q_soft + w_cons * q_consensus) / w_sum
        q_blend = float(min(0.999, max(0.0, q_blend)))

        if len(group_sizes) > 0:
            group_size = max(1.0, float(np.median(np.asarray(group_sizes, dtype=np.float32))))
        else:
            group_size = float(max(1, self.audit_group_size))
        dense_p_hat = 1.0 - pow(max(1e-6, 1.0 - q_blend), 1.0 / group_size)
        dense_p_hat = float(max(dense_p_hat, float(self.config.get("dense_mode_min_p_hat", 0.08))))
        dense_p_hat = float(min(dense_p_hat, float(self.config.get("dense_mode_max_p_hat", 0.95))))
        return True, dense_p_hat, {
            "active": 1.0,
            "q_hard": float(positive_ratio),
            "q_soft": float(q_soft),
            "q_consensus": float(q_consensus),
            "q_blend": float(q_blend),
            "group_size": float(group_size),
        }

    def _update_historical_consensus(
        self,
        current_client_scores: Mapping[int, float],
        ranked_clients: Sequence[int],
        audit_weight: int,
    ) -> None:
        max_score = max((float(score) for score in current_client_scores.values()), default=0.0)
        if max_score <= 0.0:
            return
        self._decay_historical_consensus()
        self._historical_anomalous_audits += 1
        normalizer = max(max_score, 1e-6)
        for client_id, score in current_client_scores.items():
            score_value = float(score)
            if score_value <= 0.0:
                continue
            normalized_score = score_value / normalizer
            client_key = int(client_id)
            self._historical_score_sum[client_key] = self._historical_score_sum.get(client_key, 0.0) + normalized_score * float(audit_weight)
            self._historical_score_weight[client_key] = self._historical_score_weight.get(client_key, 0.0) + float(audit_weight)

        history_topk = max(
            1,
            int(
                self.config.get(
                    "consensus_history_topk",
                    self.config.get("suspicious_topk", self.config.get("reweight_topk", 1)),
                )
            ),
        )
        for client_id in ranked_clients[:history_topk]:
            client_key = int(client_id)
            self._historical_top_count[client_key] = self._historical_top_count.get(client_key, 0.0) + 1.0

    def _decay_historical_consensus(self) -> None:
        score_decay = float(self.config.get("consensus_history_score_decay", 1.0))
        top_decay = float(self.config.get("consensus_history_top_decay", 1.0))
        warmup = max(0, int(self.config.get("consensus_history_decay_warmup", 0)))
        if self._historical_anomalous_audits < warmup:
            return

        score_decay = min(max(score_decay, 0.0), 1.0)
        top_decay = min(max(top_decay, 0.0), 1.0)
        if abs(score_decay - 1.0) <= 1e-8 and abs(top_decay - 1.0) <= 1e-8:
            return

        for client_id in list(self._historical_score_sum.keys()):
            self._historical_score_sum[client_id] = float(self._historical_score_sum.get(client_id, 0.0)) * score_decay
            self._historical_score_weight[client_id] = float(self._historical_score_weight.get(client_id, 0.0)) * score_decay
            if self._historical_score_weight[client_id] <= 1e-8:
                self._historical_score_sum.pop(client_id, None)
                self._historical_score_weight.pop(client_id, None)

        for client_id in list(self._historical_top_count.keys()):
            self._historical_top_count[client_id] = float(self._historical_top_count.get(client_id, 0.0)) * top_decay
            if self._historical_top_count[client_id] <= 1e-8:
                self._historical_top_count.pop(client_id, None)

    def _stabilize_client_scores(
        self,
        current_client_scores: Mapping[int, float],
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        positive_scores = [max(float(score), 0.0) for score in current_client_scores.values()]
        current_max = max(positive_scores, default=0.0)
        if current_max <= 0.0:
            return (
                {int(client_id): max(float(score), 0.0) for client_id, score in current_client_scores.items()},
                {
                    "current_max": 0.0,
                    "top_frequency": 0,
                    "top_persistence": 0.0,
                    "top_historical_mean": 0.0,
                },
            )

        blend_weight = float(self.config.get("stability_blend_weight", 0.35))
        persistence_boost = float(self.config.get("stability_persistence_boost", 0.20))
        stabilized_scores: Dict[int, float] = {}
        top_stats = {
            "current_max": float(current_max),
            "top_frequency": 0,
            "top_persistence": 0.0,
            "top_historical_mean": 0.0,
        }

        for client_id, raw_score in current_client_scores.items():
            client_key = int(client_id)
            current_score = max(float(raw_score), 0.0)
            total_weight = float(self._historical_score_weight.get(client_key, 0.0))
            historical_mean = (
                float(self._historical_score_sum.get(client_key, 0.0)) / total_weight
                if total_weight > 0.0 else 0.0
            )
            top_frequency = int(self._historical_top_count.get(client_key, 0))
            persistence = (
                float(top_frequency) / max(float(self._historical_anomalous_audits), 1.0)
                if self._historical_anomalous_audits > 0 else 0.0
            )
            stability_boost = current_max * (
                blend_weight * historical_mean + persistence_boost * persistence
            )
            stabilized = current_score + stability_boost
            stabilized_scores[client_key] = float(stabilized)

        if len(stabilized_scores) > 0:
            top_client_id, _ = max(
                stabilized_scores.items(),
                key=lambda item: (item[1], -item[0]),
            )
            total_weight = float(self._historical_score_weight.get(int(top_client_id), 0.0))
            top_stats = {
                "current_max": float(current_max),
                "top_frequency": int(self._historical_top_count.get(int(top_client_id), 0)),
                "top_persistence": (
                    float(self._historical_top_count.get(int(top_client_id), 0))
                    / max(float(self._historical_anomalous_audits), 1.0)
                    if self._historical_anomalous_audits > 0 else 0.0
                ),
                "top_historical_mean": (
                    float(self._historical_score_sum.get(int(top_client_id), 0.0)) / total_weight
                    if total_weight > 0.0 else 0.0
                ),
            }
        return stabilized_scores, top_stats

    def _get_consensus_candidates(
        self,
        current_client_scores: Mapping[int, float],
    ) -> Tuple[List[int], Dict[int, float]]:
        if self._historical_anomalous_audits < int(self.config.get("consensus_min_anomalous_audits", 2)):
            return [], {}

        min_frequency = max(1, int(self.config.get("consensus_min_top_frequency", 2)))
        candidate_topk = max(
            1,
            int(
                self.config.get(
                    "consensus_candidate_topk",
                    self.config.get("suspicious_topk", self.config.get("reweight_topk", 1)),
                )
            ),
        )
        score_boost = float(self.config.get("consensus_score_boost", 0.5))
        min_current_ratio = float(self.config.get("consensus_min_current_ratio", 0.50))
        min_current_ratio_hard = float(self.config.get("consensus_min_current_ratio_hard", 0.20))
        min_historical_mean = float(self.config.get("consensus_min_historical_mean", 0.12))
        persistence_override = float(self.config.get("consensus_min_persistence_override", 0.60))
        persistence_cap = float(self.config.get("consensus_persistence_cap", 0.85))
        persistence_cap = min(max(persistence_cap, 0.0), 1.0)
        relax_after_audits = int(self.config.get("consensus_relax_after_audits", 0))
        if relax_after_audits > 0 and self._historical_anomalous_audits >= relax_after_audits:
            min_frequency = max(1, min_frequency - 1)
            min_current_ratio = min(min_current_ratio, 0.35)
            min_current_ratio_hard = min(min_current_ratio_hard, 0.15)
            min_historical_mean = min(min_historical_mean, 0.10)
        consensus_scores: Dict[int, float] = {}
        current_ranked = sorted(
            ((int(client_id), float(score)) for client_id, score in current_client_scores.items() if float(score) > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        current_max_score = max((float(score) for _, score in current_ranked), default=0.0)

        for client_id, current_score in current_ranked:
            if (
                current_max_score > 0.0
                and float(current_score) < current_max_score * max(min_current_ratio_hard, 0.0)
            ):
                continue
            top_count = float(self._historical_top_count.get(client_id, 0.0))
            if top_count < float(min_frequency):
                continue
            persistence = float(top_count) / max(float(self._historical_anomalous_audits), 1.0)
            effective_persistence = min(persistence, persistence_cap)
            if (
                current_max_score > 0.0
                and float(current_score) < current_max_score * max(min_current_ratio, 0.0)
                and effective_persistence < max(persistence_override, 0.0)
            ):
                continue
            total_weight = float(self._historical_score_weight.get(client_id, 0.0))
            if total_weight <= 0.0:
                continue
            historical_mean = float(self._historical_score_sum.get(client_id, 0.0)) / total_weight
            if historical_mean < max(min_historical_mean, 0.0):
                continue
            consensus_scores[client_id] = float(current_score) * (
                historical_mean + score_boost * effective_persistence
            )

        ordered = sorted(consensus_scores.items(), key=lambda item: (-item[1], item[0]))
        return [int(client_id) for client_id, _ in ordered[:candidate_topk]], consensus_scores

    def _refine_reliable_scores(
        self,
        current_client_scores: Mapping[int, float],
        consensus_scores: Mapping[int, float],
    ) -> Dict[int, float]:
        positive_scores = [max(float(score), 0.0) for score in current_client_scores.values()]
        current_max = max(positive_scores, default=0.0)
        if current_max <= 0.0:
            return {int(client_id): max(float(score), 0.0) for client_id, score in current_client_scores.items()}

        consensus_max = max((float(score) for score in consensus_scores.values()), default=0.0)
        consensus_weight = float(self.config.get("reliable_consensus_weight", 0.35))
        history_weight = float(self.config.get("reliable_history_weight", 0.15))
        persistence_weight = float(self.config.get("reliable_persistence_weight", 0.10))
        min_support = float(self.config.get("reliable_min_support", 0.25))
        low_support_penalty = float(self.config.get("reliable_low_support_penalty", 0.75))
        min_current_ratio = float(self.config.get("reliable_min_current_ratio", 0.55))
        max_boost = float(self.config.get("reliable_max_boost", 0.40))

        client_stats: Dict[int, Dict[str, float]] = {}
        historical_values = []
        for client_id, raw_score in current_client_scores.items():
            client_key = int(client_id)
            current_score = max(float(raw_score), 0.0)
            if current_score <= 0.0:
                continue

            total_weight = float(self._historical_score_weight.get(client_key, 0.0))
            historical_mean = (
                float(self._historical_score_sum.get(client_key, 0.0)) / total_weight
                if total_weight > 0.0 else 0.0
            )
            top_frequency = float(self._historical_top_count.get(client_key, 0.0))
            persistence = (
                float(top_frequency) / max(float(self._historical_anomalous_audits), 1.0)
                if self._historical_anomalous_audits > 0 else 0.0
            )
            consensus_component = (
                float(consensus_scores.get(client_key, 0.0)) / max(consensus_max, 1e-6)
                if consensus_max > 0.0 else 0.0
            )
            client_stats[client_key] = {
                "current_score": current_score,
                "historical_mean": historical_mean,
                "persistence": persistence,
                "consensus_component": consensus_component,
            }
            historical_values.append(historical_mean)

        historical_max = max(historical_values, default=0.0)

        refined_scores: Dict[int, float] = {}
        for client_id, raw_score in current_client_scores.items():
            client_key = int(client_id)
            current_score = max(float(raw_score), 0.0)
            if current_score <= 0.0:
                refined_scores[client_key] = 0.0
                continue

            stats = client_stats.get(client_key)
            if stats is None:
                refined_scores[client_key] = current_score
                continue

            historical_component = (
                stats["historical_mean"] / max(historical_max, 1e-6)
                if historical_max > 0.0 else 0.0
            )
            support = max(historical_component, stats["persistence"], stats["consensus_component"])
            current_ratio = current_score / max(current_max, 1e-6)
            current_support_scale = 1.0
            if current_ratio < min_current_ratio:
                current_support_scale = max(current_ratio / max(min_current_ratio, 1e-6), 0.25)

            additive_boost = (
                consensus_weight * stats["consensus_component"]
                + history_weight * historical_component
                + persistence_weight * stats["persistence"]
            )
            if support < min_support:
                additive_boost *= low_support_penalty
            additive_boost *= current_support_scale
            additive_boost = min(additive_boost, max_boost)
            refined_scores[client_key] = float(current_score * (1.0 + additive_boost))

        return refined_scores

    @staticmethod
    def _map_get_float(mapping: Any, key: int, default: float = 0.0) -> float:
        if not isinstance(mapping, Mapping):
            return float(default)
        if int(key) in mapping:
            return float(mapping[int(key)])
        key_str = str(int(key))
        if key_str in mapping:
            return float(mapping[key_str])
        return float(default)

    def _evaluate_fct_group_score(
        self,
        *,
        group_update: ModelUpdate,
        round_idx: int,
        global_model: Optional[Module],
        clean_dataloader: Optional[DataLoader],
        detection_metadata_base: Mapping[str, Any],
    ) -> float:
        detect_result = self.cts_intent.detect(
            group_updates=OrderedDict([(0, group_update)]),
            global_model=global_model,
            clean_dataloader=clean_dataloader,
            round_idx=int(round_idx),
            metadata=dict(detection_metadata_base),
        )
        behavior_map = detect_result.aux_stats.get("dcbd_score", {})
        if not isinstance(behavior_map, Mapping) or len(behavior_map) == 0:
            behavior_map = detect_result.aux_stats.get("group_behavior_score", {})
        if isinstance(behavior_map, Mapping):
            if 0 in behavior_map:
                return float(behavior_map[0])
            if "0" in behavior_map:
                return float(behavior_map["0"])
        if 0 in detect_result.group_scores:
            return float(detect_result.group_scores[0])
        return 0.0

    def _run_fct_confirmation(
        self,
        *,
        round_idx: int,
        global_model: Optional[Module],
        clean_dataloader: Optional[DataLoader],
        metadata: Mapping[str, Any],
        initial_suspicious_ranking: Sequence[int],
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]], str, Dict[str, Any]]:
        fct_enabled = bool(self.config.get("fct_enable", False))
        summary: Dict[str, Any] = {
            "fct_enabled": bool(fct_enabled),
            "fct_skipped": True,
            "fct_skipped_reason": "disabled",
            "fct_clean_pool_size": 0,
            "fct_clean_pool_resampled": False,
            "fct_clean_pool_strategy": str(self.config.get("fct_clean_pool_strategy", "exclude_topn")),
            "fct_clean_pool_contamination_risk_estimate": 0.0,
            "fct_aggregate_mode": str(metadata.get("fct_aggregate_mode", "unknown")),
            "fct_initial_suspects": [],
            "fct_confirmed_suspects_raw": [],
            "fct_confirmed_suspects": [],
            "fct_rejected_suspects": [],
            "fct_confirmation_rate_raw": 0.0,
            "fct_confirmation_rate": 0.0,
            "fct_confirmed_overlap_with_malicious": [],
            "fct_confirmed_precision": None,
            "fct_confirmed_recall": None,
        }
        if not fct_enabled:
            return [], {}, "disabled", summary

        aggregate_fn = metadata.get("fct_group_aggregate_fn")
        if not callable(aggregate_fn):
            summary["fct_skipped_reason"] = "insufficient_clean_pool"
            return [], {}, "missing_group_aggregate_fn", summary

        topn = max(int(self.config.get("fct_topn", 6)), 1)
        num_pairs = max(int(self.config.get("fct_num_pairs", 6)), 1)
        z_threshold = float(self.config.get("fct_z_threshold", 1.0))
        use_matched_controls = bool(self.config.get("fct_use_matched_controls", True))
        configured_group_size = int(self.config.get("fct_group_size", 0))
        if configured_group_size <= 0:
            configured_group_size = int(self.audit_group_size)
        configured_group_size = max(configured_group_size, 2)
        clean_pool_strategy = str(self.config.get("fct_clean_pool_strategy", "exclude_topn")).strip().lower()
        if clean_pool_strategy not in {"bottom", "exclude_topn"}:
            clean_pool_strategy = "exclude_topn"
        clean_pool_bottom_ratio = float(np.clip(self.config.get("fct_clean_pool_bottom_ratio", 0.6), 0.05, 1.0))
        unique_ranking: List[int] = []
        seen: Set[int] = set()
        for client_id in initial_suspicious_ranking:
            cid = int(client_id)
            if cid in seen:
                continue
            seen.add(cid)
            unique_ranking.append(cid)
        suspect_candidates = [int(client_id) for client_id in unique_ranking[:topn]]
        summary["fct_initial_suspects"] = [int(client_id) for client_id in suspect_candidates]
        if len(suspect_candidates) == 0:
            summary["fct_skipped_reason"] = "insufficient_clean_pool"
            return [], {}, "no_suspect_candidates", summary

        pool_base_raw = metadata.get("fct_clean_pool_clients", self.selected_clients)
        if isinstance(pool_base_raw, Sequence) and not isinstance(pool_base_raw, (str, bytes)):
            pool_base_all = [int(client_id) for client_id in pool_base_raw]
        else:
            pool_base_all = [int(client_id) for client_id in self.selected_clients]
        score_by_client_raw = metadata.get("fct_initial_scores", {})
        score_by_client: Dict[int, float] = {}
        if isinstance(score_by_client_raw, Mapping):
            for key, value in score_by_client_raw.items():
                try:
                    score_by_client[int(key)] = float(value)
                except Exception:
                    continue
        suspicious_exclusion_topn = {int(client_id) for client_id in suspect_candidates}
        suspicious_exclusion_full = {int(client_id) for client_id in unique_ranking}
        if clean_pool_strategy == "bottom":
            filtered = [cid for cid in pool_base_all if cid not in suspicious_exclusion_topn]
            if len(filtered) > 0:
                ranked_bottom = sorted(
                    filtered,
                    key=lambda cid: (float(score_by_client.get(int(cid), 0.0)), int(cid)),
                )
                keep_count = max(1, int(np.ceil(len(ranked_bottom) * clean_pool_bottom_ratio)))
                pool_base = [int(cid) for cid in ranked_bottom[:keep_count]]
                fct_pool_filter_reason = "bottom_score_pool"
            else:
                pool_base = []
                fct_pool_filter_reason = "bottom_score_pool_empty"
        else:
            pool_base = [cid for cid in pool_base_all if cid not in suspicious_exclusion_topn]
            fct_pool_filter_reason = "exclude_topn_suspicious"
        if len(pool_base) < 1:
            pool_base = [cid for cid in pool_base_all if cid not in suspicious_exclusion_full]
            fct_pool_filter_reason = "exclude_all_suspicious_fallback"
        if len(pool_base) < 1:
            summary["fct_clean_pool_size"] = 0
            summary["fct_skipped_reason"] = "insufficient_clean_pool"
            return [], {}, "empty_clean_pool", summary
        summary["fct_clean_pool_size"] = int(len(pool_base))
        summary["fct_clean_pool_strategy"] = str(clean_pool_strategy)

        confirmed: List[int] = []
        stats_by_client: Dict[int, Dict[str, Any]] = {}
        clean_pool_resampled = False
        global_state = global_model.state_dict() if global_model is not None else self.model.state_dict()
        known_target_label = metadata.get("target_label", self.config.get("target_label"))
        force_known_target = bool(
            metadata.get("force_known_target", self.config.get("force_known_target", False))
        )
        reference_state_cache = self.cts_intent._resolve_reference_state(
            global_model=global_model,
            metadata={"w_t": global_state},
        )
        detection_metadata_base = {
            "w_t": global_state,
            "reference_state_cache": reference_state_cache,
            "x_syn": self._get_probe_set(clean_dataloader=clean_dataloader),
            "known_target_label": known_target_label,
            "force_known_target": force_known_target,
            "temporal_skip_update": True,
        }
        malicious_ids_raw = metadata.get("malicious_client_ids")
        contamination_risk = 0.0
        if isinstance(malicious_ids_raw, Sequence) and not isinstance(malicious_ids_raw, (str, bytes)):
            malicious_set = {int(client_id) for client_id in malicious_ids_raw}
            if len(pool_base) > 0:
                contamination_risk = float(
                    len(set(pool_base).intersection(malicious_set)) / max(len(pool_base), 1)
                )
        elif len(score_by_client) > 0:
            contamination_risk = float(
                np.clip(np.mean([max(score_by_client.get(int(cid), 0.0), 0.0) for cid in pool_base]), 0.0, 1.0)
            )
        summary["fct_clean_pool_contamination_risk_estimate"] = float(contamination_risk)
        skip_on_high_contamination = bool(
            self.config.get("fct_skip_on_high_contamination", True)
        )
        max_contamination_risk = float(
            np.clip(self.config.get("fct_skip_max_contamination_risk", 0.25), 0.0, 1.0)
        )
        summary["fct_skip_on_high_contamination"] = bool(skip_on_high_contamination)
        summary["fct_skip_max_contamination_risk"] = float(max_contamination_risk)
        if skip_on_high_contamination and contamination_risk > (max_contamination_risk + 1e-8):
            summary["fct_skipped"] = True
            summary["fct_skipped_reason"] = "high_contamination_risk"
            summary["fct_confirmation_rate"] = 0.0
            return [], {}, "high_contamination_risk", summary

        rank_by_client = {int(client_id): idx for idx, client_id in enumerate(initial_suspicious_ranking)}

        fallback_reasons: Set[str] = set()
        def _sample_from_pool(pool: Sequence[int], k: int, *, replace_if_needed: bool) -> Tuple[List[int], bool]:
            if k <= 0:
                return [], False
            if len(pool) <= 0:
                return [], False
            if (not replace_if_needed) and len(pool) >= k:
                perm = torch.randperm(len(pool), generator=self._fct_rng).tolist()
                return [int(pool[idx]) for idx in perm[:k]], False
            if len(pool) >= k:
                perm = torch.randperm(len(pool), generator=self._fct_rng).tolist()
                return [int(pool[idx]) for idx in perm[:k]], False
            draw_indices = torch.randint(0, len(pool), (k,), generator=self._fct_rng).tolist()
            return [int(pool[idx]) for idx in draw_indices], True

        for suspect_id in suspect_candidates:
            candidate_pool = [cid for cid in pool_base if cid != int(suspect_id)]
            if len(candidate_pool) < 1:
                stats_by_client[int(suspect_id)] = {
                    "fct_suspect_id": int(suspect_id),
                    "fct_suspect_initial_rank": int(rank_by_client.get(int(suspect_id), -1)),
                    "fct_suspect_initial_score": float(score_by_client.get(int(suspect_id), 0.0)),
                    "fct_mean": 0.0,
                    "fct_std": 0.0,
                    "fct_z": 0.0,
                    "num_pairs_requested": int(num_pairs),
                    "num_pairs_valid": 0,
                    "confirmed": False,
                    "fct_confirmed": False,
                    "fct_pool_filter_reason": str(fct_pool_filter_reason),
                    "fct_pool_base_size": int(len(pool_base)),
                    "skip_reason": "insufficient_clean_pool",
                    "fct_diffs": [],
                    "pair_diffs": [],
                }
                fallback_reasons.add("insufficient_clean_pool")
                continue

            effective_group_size = int(configured_group_size)
            fallback_below_k_min = bool(effective_group_size < int(self.k_min))
            if fallback_below_k_min:
                fallback_reasons.add("group_size_below_k_min_fallback")
            if effective_group_size <= 1:
                stats_by_client[int(suspect_id)] = {
                    "fct_suspect_id": int(suspect_id),
                    "fct_suspect_initial_rank": int(rank_by_client.get(int(suspect_id), -1)),
                    "fct_suspect_initial_score": float(score_by_client.get(int(suspect_id), 0.0)),
                    "fct_mean": 0.0,
                    "fct_std": 0.0,
                    "fct_z": 0.0,
                    "num_pairs_requested": int(num_pairs),
                    "num_pairs_valid": 0,
                    "confirmed": False,
                    "fct_confirmed": False,
                    "fct_pool_filter_reason": str(fct_pool_filter_reason),
                    "fct_pool_base_size": int(len(pool_base)),
                    "fallback_below_k_min": bool(fallback_below_k_min),
                    "skip_reason": "effective_group_size_too_small",
                    "fct_diffs": [],
                    "pair_diffs": [],
                }
                fallback_reasons.add("effective_group_size_too_small")
                continue

            pair_diffs: List[float] = []
            for _ in range(num_pairs):
                pos_others, pos_resampled = _sample_from_pool(
                    candidate_pool,
                    max(effective_group_size - 1, 0),
                    replace_if_needed=True,
                )
                clean_pool_resampled = bool(clean_pool_resampled or pos_resampled)
                if len(pos_others) < max(effective_group_size - 1, 0):
                    fallback_reasons.add("candidate_pool_shrunk")
                    continue
                pos_members = [int(suspect_id)] + pos_others

                neg_members: List[int] = []
                if use_matched_controls:
                    pos_set = set(pos_others)
                    control_pool = [cid for cid in candidate_pool if cid not in pos_set]
                    if len(control_pool) > 0:
                        control_pick, _ = _sample_from_pool(control_pool, 1, replace_if_needed=False)
                        neg_members = list(pos_others) + list(control_pick)
                    else:
                        fallback_reasons.add("matched_control_unavailable")
                if len(neg_members) == 0:
                    neg_members, neg_resampled = _sample_from_pool(
                        candidate_pool,
                        int(effective_group_size),
                        replace_if_needed=True,
                    )
                    clean_pool_resampled = bool(clean_pool_resampled or neg_resampled)
                elif len(neg_members) < int(effective_group_size):
                    extra_needed = int(effective_group_size) - len(neg_members)
                    extras, extra_resampled = _sample_from_pool(
                        candidate_pool,
                        int(extra_needed),
                        replace_if_needed=True,
                    )
                    neg_members.extend(extras)
                    clean_pool_resampled = bool(clean_pool_resampled or extra_resampled)

                pos_update = aggregate_fn(pos_members)
                neg_update = aggregate_fn(neg_members)
                if pos_update is None or neg_update is None:
                    fallback_reasons.add("aggregate_fn_returned_none")
                    continue
                try:
                    pos_score = self._evaluate_fct_group_score(
                        group_update=pos_update,
                        round_idx=round_idx,
                        global_model=global_model,
                        clean_dataloader=clean_dataloader,
                        detection_metadata_base=detection_metadata_base,
                    )
                    neg_score = self._evaluate_fct_group_score(
                        group_update=neg_update,
                        round_idx=round_idx,
                        global_model=global_model,
                        clean_dataloader=clean_dataloader,
                        detection_metadata_base=detection_metadata_base,
                    )
                except Exception:
                    fallback_reasons.add("fct_detect_failed")
                    continue
                pair_diffs.append(float(pos_score - neg_score))

            if len(pair_diffs) == 0:
                stats_by_client[int(suspect_id)] = {
                    "fct_suspect_id": int(suspect_id),
                    "fct_suspect_initial_rank": int(rank_by_client.get(int(suspect_id), -1)),
                    "fct_suspect_initial_score": float(score_by_client.get(int(suspect_id), 0.0)),
                    "fct_mean": 0.0,
                    "fct_std": 0.0,
                    "fct_z": 0.0,
                    "num_pairs_requested": int(num_pairs),
                    "num_pairs_valid": 0,
                    "confirmed": False,
                    "fct_confirmed": False,
                    "fct_pool_filter_reason": str(fct_pool_filter_reason),
                    "fct_pool_base_size": int(len(pool_base)),
                    "fallback_below_k_min": bool(fallback_below_k_min),
                    "skip_reason": "no_valid_pairs",
                    "fct_diffs": [],
                    "pair_diffs": [],
                }
                fallback_reasons.add("no_valid_pairs")
                continue

            diff_array_raw = np.asarray(pair_diffs, dtype=np.float64)
            fct_mean_raw = float(np.mean(diff_array_raw))
            fct_std_raw = float(np.std(diff_array_raw))
            winsor_q = float(
                np.clip(self.config.get("fct_diff_winsor_quantile", 0.10), 0.0, 0.49)
            )
            winsor_applied = bool(winsor_q > 0.0 and diff_array_raw.size >= 3)
            if winsor_applied:
                low_q = float(np.quantile(diff_array_raw, winsor_q))
                high_q = float(np.quantile(diff_array_raw, 1.0 - winsor_q))
                diff_array = np.clip(diff_array_raw, low_q, high_q)
            else:
                low_q = float(np.min(diff_array_raw))
                high_q = float(np.max(diff_array_raw))
                diff_array = diff_array_raw
            fct_mean = float(np.mean(diff_array))
            fct_std = float(np.std(diff_array))
            fct_z_std = float(fct_mean / max(fct_std, 1e-8))
            median = float(np.median(diff_array))
            mad = float(np.median(np.abs(diff_array - median)))
            mad_scale = float(max(1.4826 * mad, float(self.config.get("fct_mad_scale_eps", 1e-6))))
            fct_z_mad = float(fct_mean / max(mad_scale, 1e-8))
            use_mad_z = bool(self.config.get("fct_use_mad_z", True))
            fct_z = float(fct_z_mad if use_mad_z else fct_z_std)
            is_confirmed_raw = bool(fct_mean > 0.0 and fct_z > z_threshold)

            # Optional cross-audit stability vote: require repeated confirmations before final confirm.
            stable_enable = bool(self.config.get("fct_stability_enable", True))
            stable_window = int(max(self.config.get("fct_stability_window", 3), 1))
            stable_min_pass = int(max(self.config.get("fct_stability_min_pass", 1), 1))
            history = list(self._fct_confirm_history.get(int(suspect_id), []))
            history.append(1 if is_confirmed_raw else 0)
            if len(history) > stable_window:
                history = history[-stable_window:]
            self._fct_confirm_history[int(suspect_id)] = history
            stable_pass_count = int(sum(history))
            is_confirmed = bool(
                is_confirmed_raw
                and ((not stable_enable) or (stable_pass_count >= stable_min_pass))
            )
            if is_confirmed:
                confirmed.append(int(suspect_id))
            stats_by_client[int(suspect_id)] = {
                "fct_suspect_id": int(suspect_id),
                "fct_suspect_initial_rank": int(rank_by_client.get(int(suspect_id), -1)),
                "fct_suspect_initial_score": float(score_by_client.get(int(suspect_id), 0.0)),
                "fct_mean": float(fct_mean),
                "fct_std": float(fct_std),
                "fct_z": float(fct_z),
                "fct_mean_raw": float(fct_mean_raw),
                "fct_std_raw": float(fct_std_raw),
                "fct_z_std": float(fct_z_std),
                "fct_z_mad": float(fct_z_mad),
                "fct_median": float(median),
                "fct_mad": float(mad),
                "fct_mad_scale": float(mad_scale),
                "fct_winsor_q": float(winsor_q),
                "fct_winsor_low": float(low_q),
                "fct_winsor_high": float(high_q),
                "fct_winsor_applied": bool(winsor_applied),
                "fct_use_mad_z": bool(use_mad_z),
                "fct_confirmed_raw": bool(is_confirmed_raw),
                "fct_stability_enable": bool(stable_enable),
                "fct_stability_window": int(stable_window),
                "fct_stability_min_pass": int(stable_min_pass),
                "fct_stability_pass_count": int(stable_pass_count),
                "num_pairs_requested": int(num_pairs),
                "num_pairs_valid": int(len(pair_diffs)),
                "confirmed": bool(is_confirmed),
                "fct_confirmed": bool(is_confirmed),
                "fct_pool_filter_reason": str(fct_pool_filter_reason),
                "fct_pool_base_size": int(len(pool_base)),
                "fallback_below_k_min": bool(fallback_below_k_min),
                "skip_reason": "none",
                "fct_diffs": [float(value) for value in pair_diffs],
                "pair_diffs": [float(value) for value in pair_diffs],
            }

        fallback_reason = "none" if len(fallback_reasons) == 0 else "|".join(sorted(fallback_reasons))
        confirmed_unique = sorted({int(client_id) for client_id in confirmed})
        confirmed_raw_unique = sorted(
            int(client_id)
            for client_id, stats in stats_by_client.items()
            if bool(stats.get("fct_confirmed_raw", False))
        )
        rejected = [
            int(client_id)
            for client_id in suspect_candidates
            if int(client_id) not in set(confirmed_unique)
        ]
        summary.update(
            {
                "fct_skipped": bool(len(stats_by_client) == 0),
                "fct_skipped_reason": "none" if len(stats_by_client) > 0 else "insufficient_clean_pool",
                "fct_clean_pool_resampled": bool(clean_pool_resampled),
                "fct_confirmed_suspects_raw": [int(client_id) for client_id in confirmed_raw_unique],
                "fct_confirmed_suspects": [int(client_id) for client_id in confirmed_unique],
                "fct_rejected_suspects": [int(client_id) for client_id in rejected],
                "fct_confirmation_rate_raw": float(
                    len(confirmed_raw_unique) / max(len(suspect_candidates), 1)
                ),
                "fct_confirmation_rate": float(
                    len(confirmed_unique) / max(len(suspect_candidates), 1)
                ),
            }
        )
        malicious_ids_summary = metadata.get("malicious_client_ids")
        if isinstance(malicious_ids_summary, Sequence) and not isinstance(malicious_ids_summary, (str, bytes)):
            malicious_set_summary = {int(client_id) for client_id in malicious_ids_summary}
            overlap = sorted(
                int(client_id)
                for client_id in set(confirmed_unique).intersection(malicious_set_summary)
            )
            precision = float(len(overlap) / max(len(confirmed_unique), 1)) if len(confirmed_unique) > 0 else 0.0
            recall = float(len(overlap) / max(len(malicious_set_summary), 1)) if len(malicious_set_summary) > 0 else 0.0
            summary["fct_confirmed_overlap_with_malicious"] = [int(client_id) for client_id in overlap]
            summary["fct_confirmed_precision"] = float(precision)
            summary["fct_confirmed_recall"] = float(recall)
        else:
            summary["fct_confirmed_overlap_with_malicious"] = []
            summary["fct_confirmed_precision"] = None
            summary["fct_confirmed_recall"] = None
        return confirmed_unique, stats_by_client, str(fallback_reason), summary

    def _resolve_device(self, device: Optional[Union[str, torch.device]], model: Module) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _update_norm(self, update: ModelUpdate) -> float:
        if isinstance(update, torch.Tensor):
            return float(update.detach().float().norm(p=2).item())
        if isinstance(update, Mapping):
            tensors = [value.detach().reshape(-1).float() for value in update.values()]
            if len(tensors) == 0:
                return 0.0
            return float(torch.cat(tensors, dim=0).norm(p=2).item())
        if isinstance(update, (list, tuple)):
            tensors = [tensor.detach().reshape(-1).float() for tensor in update]
            if len(tensors) == 0:
                return 0.0
            return float(torch.cat(tensors, dim=0).norm(p=2).item())
        raise TypeError("Unsupported update type.")

    def _clone_update(self, update: ModelUpdate) -> ModelUpdate:
        if isinstance(update, torch.Tensor):
            return update.detach().clone()
        if isinstance(update, Mapping):
            return {key: value.detach().clone() for key, value in update.items()}
        if isinstance(update, (list, tuple)):
            return [tensor.detach().clone() for tensor in update]
        raise TypeError("Unsupported update type.")

    def _scale_update(self, update: ModelUpdate, scale: float) -> ModelUpdate:
        s = float(scale)
        if isinstance(update, torch.Tensor):
            return update.detach().clone() * s
        if isinstance(update, Mapping):
            return {key: value.detach().clone() * s for key, value in update.items()}
        if isinstance(update, (list, tuple)):
            return [tensor.detach().clone() * s for tensor in update]
        raise TypeError("Unsupported update type.")

    def apply_continuous_projection(
        self,
        aggregated_update: ModelUpdate,
        *,
        round_idx: Optional[int] = None,
    ) -> Tuple[ModelUpdate, Dict[str, Any]]:
        """Project every-round global update using the latest B_tilde (if available)."""

        if not bool(self.config.get("dgc_continuous_projection", True)):
            return self._clone_update(aggregated_update), {"projected": False, "reason": "disabled"}

        if self.current_B_tilde is None or self.current_B_tilde.numel() == 0:
            self.continuous_proj_stats["rounds_without_B_tilde"] = float(
                self.continuous_proj_stats.get("rounds_without_B_tilde", 0.0) + 1.0
            )
            return self._clone_update(aggregated_update), {"projected": False, "reason": "no_B_tilde"}

        strength = self._clip01(float(self.config.get("dgc_continuous_projection_strength", 1.0)))
        if strength <= 0.0:
            return self._clone_update(aggregated_update), {"projected": False, "reason": "zero_strength"}

        delta_flat = self.dgc_clean._flatten_update(aggregated_update).detach().to(self.device)
        basis = self.current_B_tilde.detach().to(self.device, dtype=delta_flat.dtype)
        if basis.ndim != 2 or basis.shape[0] != delta_flat.numel() or basis.shape[1] <= 0:
            return self._clone_update(aggregated_update), {
                "projected": False,
                "reason": "basis_shape_mismatch",
                "basis_shape": list(basis.shape),
                "update_dim": int(delta_flat.numel()),
            }

        proj_coeff = basis.t().matmul(delta_flat)
        proj_component = basis.matmul(proj_coeff)
        delta_clean_flat = delta_flat - strength * proj_component

        norm_before = float(delta_flat.norm(p=2).item())
        norm_after = float(delta_clean_flat.norm(p=2).item())
        proj_norm = float(proj_component.norm(p=2).item())
        reduction_ratio = float(proj_norm / max(norm_before, 1e-8))

        cleaned_update: ModelUpdate
        if isinstance(aggregated_update, Mapping):
            cleaned_mapping: "OrderedDict[str, torch.Tensor]" = OrderedDict(
                (key, value.detach().clone()) for key, value in aggregated_update.items()
            )
            restored = self.dgc_clean._unflatten_like_model(delta_clean_flat)
            named_parameters = OrderedDict(self.model.named_parameters())
            for name, tensor in zip(self.dgc_clean._param_names, restored):
                if name in cleaned_mapping:
                    ref = cleaned_mapping[name]
                    cleaned_mapping[name] = tensor.detach().to(device=ref.device, dtype=ref.dtype)
                elif name in named_parameters:
                    cleaned_mapping[name] = tensor.detach().to(dtype=named_parameters[name].dtype).cpu()
            cleaned_update = cleaned_mapping
        elif isinstance(aggregated_update, torch.Tensor):
            cleaned_update = delta_clean_flat.detach().clone().to(
                device=aggregated_update.device,
                dtype=aggregated_update.dtype,
            )
        elif isinstance(aggregated_update, tuple):
            restored = self.dgc_clean._unflatten_like_model(delta_clean_flat)
            cleaned_update = tuple(restored)
        elif isinstance(aggregated_update, list):
            restored = self.dgc_clean._unflatten_like_model(delta_clean_flat)
            cleaned_update = list(restored)
        else:
            cleaned_update = self._clone_update(aggregated_update)
            return cleaned_update, {"projected": False, "reason": "unsupported_update_type"}

        self.continuous_proj_stats["total_rounds_projected"] = float(
            self.continuous_proj_stats.get("total_rounds_projected", 0.0) + 1.0
        )
        self.continuous_proj_stats["total_norm_removed"] = float(
            self.continuous_proj_stats.get("total_norm_removed", 0.0) + proj_norm
        )
        self.continuous_proj_stats["total_reduction_ratio"] = float(
            self.continuous_proj_stats.get("total_reduction_ratio", 0.0) + reduction_ratio
        )

        if round_idx is not None and (int(round_idx) + 1) % 20 == 0:
            self.continuous_proj_history[f"round_{int(round_idx) + 1}"] = {
                "norm_before": norm_before,
                "norm_after": norm_after,
                "reduction_ratio": reduction_ratio,
            }

        return cleaned_update, {
            "projected": True,
            "norm_before": norm_before,
            "norm_after": norm_after,
            "proj_norm": proj_norm,
            "reduction_ratio": reduction_ratio,
            "projection_strength": strength,
        }

    def purify_model_weights(
        self,
        model: Module,
        *,
        strength: Optional[float] = None,
        round_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Purify accumulated model weights by removing B_tilde component from (w_t - w_init).

        Formula:
            w_clean = w_init + (w_t - w_init - s * B(B^T (w_t - w_init)))
        """

        if not bool(self.config.get("dgc_weight_purification", False)):
            return {"purified": False, "reason": "disabled"}
        if self.current_B_tilde is None or self.current_B_tilde.numel() == 0:
            return {"purified": False, "reason": "no_B_tilde"}

        use_strength = self._clip01(
            float(self.config.get("dgc_weight_purify_strength", 0.3) if strength is None else strength)
        )
        if use_strength <= 0.0:
            return {"purified": False, "reason": "zero_strength"}

        with torch.no_grad():
            current_flat = self.dgc_clean._flatten_model_parameters(model).detach().to(self.device)
            if self.w_init_flat.numel() != current_flat.numel():
                return {
                    "purified": False,
                    "reason": "w_init_shape_mismatch",
                    "w_init_dim": int(self.w_init_flat.numel()),
                    "w_current_dim": int(current_flat.numel()),
                }

            delta_accumulated = current_flat - self.w_init_flat
            basis = self.current_B_tilde.detach().to(self.device, dtype=delta_accumulated.dtype)
            if basis.ndim != 2 or basis.shape[0] != delta_accumulated.numel() or basis.shape[1] <= 0:
                return {
                    "purified": False,
                    "reason": "basis_shape_mismatch",
                    "basis_shape": list(basis.shape),
                    "delta_dim": int(delta_accumulated.numel()),
                }

            coeff = basis.t().matmul(delta_accumulated)
            backdoor_component = basis.matmul(coeff)
            delta_clean = delta_accumulated - use_strength * backdoor_component
            w_clean_flat = self.w_init_flat + delta_clean

            restored_params = self.dgc_clean._unflatten_like_model(w_clean_flat)
            for parameter, clean_param in zip(model.parameters(), restored_params):
                parameter.data.copy_(clean_param.to(device=parameter.device, dtype=parameter.dtype))

            total_norm = float(delta_accumulated.norm(p=2).item())
            backdoor_norm = float(backdoor_component.norm(p=2).item())
            energy_fraction = float(backdoor_norm / max(total_norm, 1e-8))
            rank = int(basis.shape[1])

            self.weight_purify_stats["total_purify_calls"] = float(
                self.weight_purify_stats.get("total_purify_calls", 0.0) + 1.0
            )
            self.weight_purify_stats["total_backdoor_energy_fraction"] = float(
                self.weight_purify_stats.get("total_backdoor_energy_fraction", 0.0) + energy_fraction
            )
            total_calls = max(float(self.weight_purify_stats.get("total_purify_calls", 0.0)), 1.0)
            self.weight_purify_stats["mean_backdoor_energy_fraction"] = float(
                self.weight_purify_stats.get("total_backdoor_energy_fraction", 0.0) / total_calls
            )
            self.weight_purify_stats["last_round"] = float(int(round_idx)) if round_idx is not None else -1.0
            self.weight_purify_stats["last_strength"] = float(use_strength)
            self.weight_purify_stats["last_b_tilde_rank"] = float(rank)
            hist_item = {
                "round": int(round_idx) if round_idx is not None else -1,
                "backdoor_energy_fraction": float(energy_fraction),
                "strength": float(use_strength),
                "b_tilde_rank": int(rank),
            }
            self.weight_purify_history.append(hist_item)

        return {
            "purified": True,
            "backdoor_energy_fraction": float(energy_fraction),
            "strength": float(use_strength),
            "B_tilde_rank": int(rank),
            "delta_accumulated_norm": float(total_norm),
            "backdoor_component_norm": float(backdoor_norm),
        }
