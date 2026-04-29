"""CTS-Intent: behavior-intent detection for secure-aggregation FL."""

from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch.func import functional_call
from torch.utils.data import DataLoader

from defense.c3s_guard.trigger_proxy import TriggerProxyBank
from defense.c3s_guard.utils import CTSIntentResult, ModelUpdate


class CTSIntent:
    """Score secure-aggregation group updates by behavior intent."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Module,
        trigger_proxy_bank: Optional[TriggerProxyBank] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the CTS-Intent detector.

        Args:
            config: Dictionary of CTS-Intent hyper-parameters.
            model: BackdoorBench-compatible classification model.
            trigger_proxy_bank: Optional proxy bank used for trigger probing.
            device: Device used for intent feature extraction.
        """

        self.model = model
        self.trigger_proxy_bank = trigger_proxy_bank
        self.device = self._resolve_device(device, model)
        self.model.to(self.device)
        self.model.eval()

        self.config = self._build_config(config)
        self._probe_cache: Optional[torch.Tensor] = None
        self._reference_state: Optional[OrderedDict[str, torch.Tensor]] = None
        self._last_group_order: List[int] = []
        self._last_y_star: Optional[int] = None
        self._last_extract_stats: Dict[str, Any] = {}
        self._temporal_cts_history: Dict[str, List[Tuple[int, float]]] = {}
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.config["seed"]))

        self._param_names = [name for name, _ in self.model.named_parameters()]
        self._param_numels = [param.numel() for _, param in self.model.named_parameters()]
        self._param_shapes = [param.shape for _, param in self.model.named_parameters()]
        self._buffer_names = [name for name, _ in self.model.named_buffers()]

    @staticmethod
    def _trigger_family(trigger_name: str) -> str:
        """Map concrete trigger names to coarse behavior families."""

        name = str(trigger_name).strip().lower()
        if any(token in name for token in ("patch", "occlusion", "cutout", "mask")):
            return "spatial"
        if any(token in name for token in ("blend", "color", "sig", "sine", "frequency", "hsv")):
            return "appearance"
        return "other"

    def extract_behavior_signals(
        self,
        group_updates: Mapping[int, ModelUpdate],
        global_model: Optional[Module] = None,
        clean_dataloader: Optional[DataLoader] = None,
        round_idx: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Extract intent-related features from secure group aggregates.

        Args:
            group_updates: Mapping from group id to aggregated update ``Delta_G``.
            global_model: Optional pre-aggregation global model.
            clean_dataloader: Optional clean loader for reference probing.
            round_idx: Optional global round index.
            metadata: Optional round-level side information.

        Returns:
            A mapping from group id to a dictionary of feature tensors.
        """

        metadata = dict(metadata or {})
        base_state = self._resolve_reference_state(global_model=global_model, metadata=metadata)
        x_syn = self._resolve_probe_set(clean_dataloader=clean_dataloader, metadata=metadata)
        normalized_groups = self._normalize_group_updates(group_updates)
        group_ids = [group_id for group_id, _ in normalized_groups]
        self._last_group_order = group_ids
        self._last_extract_stats = {}

        if len(normalized_groups) == 0:
            return {}

        known_target_label = metadata.get("known_target_label", self.config.get("known_target_label"))
        force_known_target = bool(metadata.get("force_known_target", self.config.get("force_known_target", False)))
        skip_coarse_when_known_target = bool(
            metadata.get(
                "skip_coarse_when_known_target",
                self.config.get("skip_coarse_when_known_target", False),
            )
        )

        forward_stats: Dict[str, int] = {
            "total_forward_calls": 0,
            "total_forward_samples": 0,
            "coarse_forward_calls": 0,
            "coarse_forward_samples": 0,
            "full_forward_calls": 0,
            "full_forward_samples": 0,
            "clean_forward_calls": 0,
            "clean_forward_samples": 0,
        }

        def run_forward(
            state_dict: OrderedDict[str, torch.Tensor],
            inputs: torch.Tensor,
            stage: str,
        ) -> torch.Tensor:
            logits = self._forward_logits(state_dict, inputs)
            samples = int(inputs.shape[0])
            forward_stats["total_forward_calls"] += 1
            forward_stats["total_forward_samples"] += samples
            if stage == "coarse":
                forward_stats["coarse_forward_calls"] += 1
                forward_stats["coarse_forward_samples"] += samples
            elif stage == "full":
                forward_stats["full_forward_calls"] += 1
                forward_stats["full_forward_samples"] += samples
            elif stage == "clean":
                forward_stats["clean_forward_calls"] += 1
                forward_stats["clean_forward_samples"] += samples
            return logits

        trigger_batches = self._build_trigger_batches(x_syn)
        cts_mode_requested = str(self.config.get("cts_mode", "raw")).strip().lower()
        if cts_mode_requested not in {"raw", "mdbf", "dcbd"}:
            cts_mode_requested = "raw"
        dcbd_enabled = bool(cts_mode_requested == "dcbd")
        update_by_group: Dict[int, ModelUpdate] = {
            int(group_id): group_update for group_id, group_update in normalized_groups
        }

        coarse_skipped = bool(force_known_target and known_target_label is not None and skip_coarse_when_known_target)
        coarse_ranked_classes: Optional[torch.Tensor] = None
        coarse_ranked_classes_desc: Optional[torch.Tensor] = None
        coarse_ranked_classes_asc: Optional[torch.Tensor] = None
        coarse_mode = str(self.config.get("coarse_mode", "margin")).strip().lower()
        if coarse_mode not in {"abs_delta", "margin", "softmax_delta", "ratio"}:
            coarse_mode = "margin"
        coarse_bias_coeff = float(self.config.get("coarse_bias_coeff", 0.35))
        if not coarse_skipped:
            coarse_trigger_names = list(trigger_batches.keys())
            base_logits_by_trigger: Dict[str, torch.Tensor] = {
                name: run_forward(base_state, trigger_batches[name], stage="coarse")
                for name in coarse_trigger_names
            }
            num_classes = int(next(iter(base_logits_by_trigger.values())).shape[-1])

            coarse_deltas: List[torch.Tensor] = []
            coarse_scores: List[torch.Tensor] = []
            for group_id, group_update in normalized_groups:
                updated_state = self._apply_group_update(base_state, group_update)
                per_trigger_delta: List[torch.Tensor] = []
                per_trigger_score: List[torch.Tensor] = []
                for name in coarse_trigger_names:
                    updated_logits = run_forward(updated_state, trigger_batches[name], stage="coarse")
                    base_logits = base_logits_by_trigger[name]
                    coarse_delta = torch.nan_to_num(updated_logits - base_logits, nan=0.0, posinf=0.0, neginf=0.0)
                    per_trigger_delta.append(coarse_delta.mean(dim=0))
                    per_trigger_score.append(
                        self._compute_coarse_class_scores(
                            base_logits=base_logits,
                            updated_logits=updated_logits,
                            mode=coarse_mode,
                        )
                    )
                    del updated_logits, coarse_delta
                delta_stack = torch.stack(per_trigger_delta, dim=0)
                score_stack = torch.stack(per_trigger_score, dim=0)
                coarse_deltas.append(torch.median(delta_stack, dim=0).values)
                coarse_scores.append(torch.median(score_stack, dim=0).values)
                del updated_state, delta_stack, score_stack

            coarse_delta_tensor = torch.stack(coarse_deltas, dim=0)
            coarse_delta_tensor = torch.nan_to_num(coarse_delta_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            coarse_score_tensor = torch.stack(coarse_scores, dim=0)
            coarse_score_tensor = torch.nan_to_num(coarse_score_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            coarse_q = float(self.config.get("coarse_aggregate_quantile", 0.5))
            if coarse_q <= 0.0:
                coarse_q = 0.5
            if coarse_q >= 1.0:
                coarse_q = 0.99
            if abs(coarse_q - 0.5) < 1e-9:
                aggregate_coarse_signal = torch.median(coarse_score_tensor, dim=0).values
            else:
                aggregate_coarse_signal = torch.quantile(coarse_score_tensor, q=coarse_q, dim=0)

            coarse_bias_stack = []
            bias_temp = max(float(self.config.get("coarse_bias_temperature", 1.0)), 1e-6)
            for name in coarse_trigger_names:
                probs = torch.softmax(base_logits_by_trigger[name] / bias_temp, dim=-1).mean(dim=0)
                coarse_bias_stack.append(probs)
            coarse_bias = torch.median(torch.stack(coarse_bias_stack, dim=0), dim=0).values
            coarse_bias = torch.nan_to_num(coarse_bias, nan=0.0, posinf=0.0, neginf=0.0)
            aggregate_coarse_delta = aggregate_coarse_signal - coarse_bias_coeff * coarse_bias

            coarse_ranked_classes_desc = torch.argsort(aggregate_coarse_delta, dim=0, descending=True)
            coarse_ranked_classes_asc = torch.argsort(aggregate_coarse_delta, dim=0, descending=False)
            coarse_ranked_classes = coarse_ranked_classes_desc

            if force_known_target and known_target_label is not None:
                topk = min(int(self.config["topk_candidates"]), num_classes)
                candidate_classes = torch.topk(aggregate_coarse_delta, k=topk, dim=0).indices.to(self.device)
            else:
                top_high = max(1, int(self.config.get("coarse_top_high", 3)))
                top_low = max(0, int(self.config.get("coarse_top_low", 2)))
                high_part = coarse_ranked_classes_desc[: min(top_high, num_classes)]
                low_part = coarse_ranked_classes_asc[: min(top_low, num_classes)]
                candidate_classes = torch.unique(torch.cat([high_part, low_part], dim=0), sorted=False).to(self.device)

            if force_known_target and known_target_label is not None:
                target_tensor = torch.tensor([int(known_target_label)], device=self.device, dtype=candidate_classes.dtype)
                if not bool((candidate_classes == target_tensor[0]).any().item()):
                    candidate_classes = torch.unique(
                        torch.cat([target_tensor, candidate_classes], dim=0),
                        sorted=False,
                    )
        else:
            num_classes = int(self.config.get("num_classes", 10))
            candidate_classes = torch.tensor([int(known_target_label)], device=self.device, dtype=torch.long)
            aggregate_coarse_delta = torch.zeros(num_classes, device=self.device)
            aggregate_coarse_signal = torch.zeros(num_classes, device=self.device)
            coarse_bias = torch.zeros(num_classes, device=self.device)
            coarse_delta_tensor = torch.zeros((len(group_ids), num_classes), device=self.device)
            coarse_score_tensor = torch.zeros((len(group_ids), num_classes), device=self.device)
            coarse_ranked_classes = torch.arange(num_classes, device=self.device, dtype=torch.long)
            coarse_ranked_classes_desc = coarse_ranked_classes
            coarse_ranked_classes_asc = torch.flip(coarse_ranked_classes, dims=[0])

        bootstrap_pseudo_y_star = metadata.get("bootstrap_pseudo_y_star")
        if (not force_known_target) and (bootstrap_pseudo_y_star is not None):
            try:
                bootstrap_cls = int(bootstrap_pseudo_y_star)
            except Exception:
                bootstrap_cls = -1
            if 0 <= bootstrap_cls < int(num_classes):
                bootstrap_tensor = torch.tensor([bootstrap_cls], device=self.device, dtype=candidate_classes.dtype)
                if not bool((candidate_classes == bootstrap_tensor[0]).any().item()):
                    merged = torch.cat([bootstrap_tensor, candidate_classes], dim=0).detach().cpu().tolist()
                    dedup: List[int] = []
                    for cls in merged:
                        if int(cls) not in dedup:
                            dedup.append(int(cls))
                    candidate_classes = torch.tensor(dedup, device=self.device, dtype=torch.long)
                    self._last_extract_stats["bootstrap_injected"] = True
                else:
                    self._last_extract_stats["bootstrap_injected"] = False
                self._last_extract_stats["bootstrap_pseudo_y_star"] = int(bootstrap_cls)

        trigger_names = list(trigger_batches.keys())
        trigger_family_indices: Dict[str, List[int]] = {}
        for idx, trigger_name in enumerate(trigger_names):
            family = self._trigger_family(trigger_name)
            trigger_family_indices.setdefault(str(family), []).append(int(idx))
        full_trigger_batch = torch.cat([trigger_batches[name] for name in trigger_names], dim=0)
        probe_size = int(x_syn.shape[0])
        num_triggers = len(trigger_names)
        base_logits_full_raw = run_forward(base_state, full_trigger_batch, stage="full")
        base_logits_clean_raw: Optional[torch.Tensor] = None
        if dcbd_enabled:
            base_logits_clean_raw = run_forward(base_state, x_syn, stage="clean")
        num_classes = int(base_logits_full_raw.shape[-1])
        if int(candidate_classes.min().item()) < 0 or int(candidate_classes.max().item()) >= num_classes:
            raise ValueError(
                f"candidate_classes out of range for num_classes={num_classes}: "
                f"{candidate_classes.detach().cpu().tolist()}"
            )
        base_logits_full = base_logits_full_raw.view(num_triggers, probe_size, num_classes)
        base_margins = self._margin_from_logits(base_logits_full, candidate_classes)

        group_features: Dict[int, Dict[str, torch.Tensor]] = {}
        use_ensemble = bool(self.config.get("use_ensemble", False))
        ensemble_mode = str(self.config.get("ensemble_mode", "weighted_vote")).strip().lower()
        ensemble_temperature = max(float(self.config.get("ensemble_temperature", 1.0)), 1e-6)
        cons_temp_prob = max(float(self.config.get("cons_temp_prob", 1.0)), 1e-6)
        cons_temp_margin = max(float(self.config.get("cons_temp_margin", 1.0)), 1e-6)
        cons_tau_v = max(float(self.config.get("cons_tau_v", 1.0)), 1e-6)
        for group_id in group_ids:
            updated_state = self._apply_group_update(base_state, update_by_group[group_id])
            updated_logits_full = run_forward(updated_state, full_trigger_batch, stage="full").view(
                num_triggers,
                probe_size,
                num_classes,
            )
            updated_margins = self._margin_from_logits(updated_logits_full, candidate_classes)
            # CTS(G) = E_{x,T}[phi(m_after) - phi(m_before)], not raw margin sum.
            base_phi = self._phi_transform(base_margins)
            updated_phi = self._phi_transform(updated_margins)
            phi_delta = updated_phi - base_phi
            per_trigger_candidate_scores = phi_delta.mean(dim=2)
            score_clip = float(self.config.get("cts_score_clip", 2.0))
            if score_clip > 0.0:
                per_trigger_candidate_scores = torch.clamp(
                    per_trigger_candidate_scores, min=-score_clip, max=score_clip
                )
            if use_ensemble and ensemble_mode == "weighted_vote":
                trigger_weights = torch.softmax(per_trigger_candidate_scores / ensemble_temperature, dim=1)
                candidate_scores = (trigger_weights * per_trigger_candidate_scores).sum(dim=1)
            else:
                trigger_weights = torch.full_like(per_trigger_candidate_scores, 1.0 / float(num_triggers))
                candidate_scores = per_trigger_candidate_scores.mean(dim=1)
            candidate_scores = torch.nan_to_num(candidate_scores, nan=0.0, posinf=0.0, neginf=0.0)
            if score_clip > 0.0:
                candidate_scores = torch.clamp(candidate_scores, min=-score_clip, max=score_clip)

            # Continuous consistency score (Cons_soft), first on all triggers then
            # with family-wise aggregation to avoid penalizing heterogeneous trigger
            # families (e.g., patch vs color/occlusion) too aggressively.
            logits_t = updated_logits_full.mean(dim=1)  # [T, C]
            selected_logits = logits_t.index_select(dim=1, index=candidate_classes.to(logits_t.device))  # [T, K]
            top2_vals, top2_idx = torch.topk(logits_t, k=min(2, logits_t.shape[1]), dim=1)
            top1_val = top2_vals[:, 0].unsqueeze(1)
            top1_idx = top2_idx[:, 0].unsqueeze(1)
            if top2_vals.shape[1] > 1:
                top2_val = top2_vals[:, 1].unsqueeze(1)
            else:
                top2_val = top1_val
            class_ids = candidate_classes.to(logits_t.device).view(1, -1).expand(logits_t.shape[0], -1)
            max_other = torch.where(top1_idx == class_ids, top2_val, top1_val)
            m_t = selected_logits - max_other  # [T, K]
            prob_t = torch.softmax(logits_t / cons_temp_prob, dim=1).index_select(
                dim=1, index=candidate_classes.to(logits_t.device)
            )

            def _consistency_from_subset(trigger_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                if trigger_indices.numel() <= 0:
                    zeros = torch.zeros(candidate_classes.numel(), device=logits_t.device, dtype=logits_t.dtype)
                    return zeros, zeros, zeros, zeros
                prob_sub = prob_t.index_select(dim=0, index=trigger_indices)
                margin_sub = m_t.index_select(dim=0, index=trigger_indices)
                a_sub = prob_sub.mean(dim=0)
                u_sub = torch.median(torch.sigmoid(margin_sub / cons_temp_margin), dim=0).values
                margin_center = torch.median(margin_sub, dim=0).values
                margin_mad = torch.median(torch.abs(margin_sub - margin_center.unsqueeze(0)), dim=0).values
                v_sub = torch.exp(-margin_mad / cons_tau_v)
                cons_sub = 0.35 * a_sub + 0.45 * u_sub + 0.20 * v_sub
                cons_sub = torch.clamp(
                    torch.nan_to_num(cons_sub, nan=0.0, posinf=0.0, neginf=0.0),
                    min=0.0,
                    max=1.0,
                )
                return cons_sub, a_sub, u_sub, v_sub

            all_trigger_indices = torch.arange(num_triggers, device=logits_t.device, dtype=torch.long)
            consistency_all, a_y, u_y, v_y = _consistency_from_subset(all_trigger_indices)
            family_consistency_list: List[torch.Tensor] = []
            for _, family_indices in trigger_family_indices.items():
                if len(family_indices) <= 0:
                    continue
                family_idx_tensor = torch.tensor(family_indices, device=logits_t.device, dtype=torch.long)
                family_consistency, _, _, _ = _consistency_from_subset(family_idx_tensor)
                family_consistency_list.append(family_consistency)

            if len(family_consistency_list) > 0:
                family_stack = torch.stack(family_consistency_list, dim=0)
                if family_stack.shape[0] > 1:
                    topk_vals = torch.topk(family_stack, k=min(2, int(family_stack.shape[0])), dim=0).values
                    family_consensus = topk_vals.mean(dim=0)
                else:
                    family_consensus = family_stack[0]
                family_blend = float(np.clip(self.config.get("consistency_family_blend", 0.65), 0.0, 1.0))
                consistency = (1.0 - family_blend) * consistency_all + family_blend * family_consensus
                consistency = torch.clamp(
                    torch.nan_to_num(consistency, nan=0.0, posinf=0.0, neginf=0.0),
                    min=0.0,
                    max=1.0,
                )
            else:
                consistency = consistency_all

            if dcbd_enabled:
                trig_delta_per_class = torch.nan_to_num(
                    updated_logits_full - base_logits_full,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).mean(dim=(0, 1))
                if base_logits_clean_raw is not None:
                    updated_logits_clean = run_forward(updated_state, x_syn, stage="clean")
                    clean_delta_per_class = torch.nan_to_num(
                        updated_logits_clean - base_logits_clean_raw,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).mean(dim=0)
                    del updated_logits_clean
                else:
                    clean_delta_per_class = torch.zeros(
                        num_classes,
                        device=updated_logits_full.device,
                        dtype=updated_logits_full.dtype,
                    )
            else:
                trig_delta_per_class = torch.zeros(
                    num_classes,
                    device=updated_logits_full.device,
                    dtype=updated_logits_full.dtype,
                )
                clean_delta_per_class = torch.zeros_like(trig_delta_per_class)

            group_features[group_id] = {
                "candidate_classes": candidate_classes.detach().cpu(),
                "candidate_scores": candidate_scores.detach().cpu(),
                "candidate_consistency": consistency.detach().cpu(),
                "candidate_consistency_all": consistency_all.detach().cpu(),
                "candidate_cons_a": a_y.detach().cpu(),
                "candidate_cons_u": u_y.detach().cpu(),
                "candidate_cons_v": v_y.detach().cpu(),
                "candidate_trigger_scores": per_trigger_candidate_scores.detach().cpu(),
                "candidate_trigger_weights": trigger_weights.detach().cpu(),
                "coarse_class_delta": coarse_delta_tensor[group_ids.index(group_id)].detach().cpu(),
                "coarse_class_score": coarse_score_tensor[group_ids.index(group_id)].detach().cpu(),
                "coarse_class_bias": coarse_bias.detach().cpu(),
                "cts_trig_per_class": trig_delta_per_class.detach().cpu(),
                "cts_clean_per_class": clean_delta_per_class.detach().cpu(),
                "trigger_names": torch.arange(num_triggers, dtype=torch.long),
                "trigger_family_count": torch.tensor(len(trigger_family_indices), dtype=torch.long),
                "probe_size": torch.tensor(probe_size, dtype=torch.long),
                "round_idx": torch.tensor(-1 if round_idx is None else round_idx, dtype=torch.long),
            }
            del updated_state, updated_logits_full, updated_margins, base_phi, updated_phi, phi_delta, candidate_scores, consistency

        self._last_extract_stats = {
            "coarse_skipped": bool(coarse_skipped),
            "coarse_ranked_classes": (
                []
                if coarse_ranked_classes is None
                else [int(v) for v in coarse_ranked_classes.detach().cpu().tolist()]
            ),
            "coarse_ranked_classes_desc": (
                []
                if coarse_ranked_classes_desc is None
                else [int(v) for v in coarse_ranked_classes_desc.detach().cpu().tolist()]
            ),
            "coarse_ranked_classes_asc": (
                []
                if coarse_ranked_classes_asc is None
                else [int(v) for v in coarse_ranked_classes_asc.detach().cpu().tolist()]
            ),
            "coarse_bias": [float(v) for v in coarse_bias.detach().cpu().tolist()],
            "coarse_adjusted_scores": [float(v) for v in aggregate_coarse_delta.detach().cpu().tolist()],
            "coarse_signal_scores": [float(v) for v in aggregate_coarse_signal.detach().cpu().tolist()],
            "forward_stats": {key: int(value) for key, value in forward_stats.items()},
            "num_groups": int(len(group_ids)),
            "num_triggers": int(num_triggers),
            "probe_size": int(probe_size),
            "num_classes": int(num_classes),
            "coarse_mode": coarse_mode,
            "use_ensemble": bool(use_ensemble),
            "ensemble_mode": str(ensemble_mode),
        }

        return group_features

    def score_groups(
        self,
        group_features: Mapping[int, Mapping[str, torch.Tensor]],
        round_idx: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CTSIntentResult:
        """Convert extracted behavior signals into group-level suspicion scores."""

        if len(group_features) == 0:
            return CTSIntentResult(round_idx=-1 if round_idx is None else round_idx)

        group_ids = list(group_features.keys())
        candidate_classes = group_features[group_ids[0]]["candidate_classes"].long()
        candidate_scores = torch.stack(
            [group_features[group_id]["candidate_scores"].float() for group_id in group_ids],
            dim=0,
        )
        candidate_scores = torch.nan_to_num(candidate_scores, nan=0.0, posinf=0.0, neginf=0.0)
        candidate_consistency = torch.stack(
            [group_features[group_id]["candidate_consistency"].float() for group_id in group_ids],
            dim=0,
        )
        candidate_consistency = torch.nan_to_num(candidate_consistency, nan=0.0, posinf=0.0, neginf=0.0)
        first_trigger_scores = group_features[group_ids[0]].get("candidate_trigger_scores")
        if isinstance(first_trigger_scores, torch.Tensor) and first_trigger_scores.dim() >= 2:
            num_triggers_effective = int(first_trigger_scores.shape[0])
        else:
            num_triggers_effective = int(group_features[group_ids[0]].get("trigger_names", torch.empty(0)).numel())
        single_trigger_mode = bool(num_triggers_effective <= 1)
        disable_consistency_gate_single = bool(self.config.get("single_trigger_disable_consistency_gate", True))

        metadata = dict(metadata or {})
        aggregate_candidate_scores = torch.median(candidate_scores, dim=0).values
        aggregate_candidate_scores = torch.nan_to_num(aggregate_candidate_scores, nan=0.0, posinf=0.0, neginf=0.0)
        known_target_label = metadata.get("known_target_label", self.config.get("known_target_label"))
        force_known_target = bool(metadata.get("force_known_target", self.config.get("force_known_target", False)))
        tau_cons = float(self.config["tau_cons"])
        fine_mode_requested = str(metadata.get("fine_mode", self.config.get("fine_mode", "separation_mad"))).strip().lower()
        if fine_mode_requested not in {"median_max", "separation_mad", "separation_known", "tail_topk_mean"}:
            fine_mode_requested = "separation_mad"
        # Backward compatibility with previous naming.
        if fine_mode_requested == "tail_topk_mean":
            fine_mode_requested = "separation_mad"
        fine_mode_used = fine_mode_requested
        candidate_separation_scores: Dict[int, float] = {}
        candidate_cts_vectors: Dict[int, List[float]] = {}
        candidate_refine_scores: Dict[int, float] = {}
        candidate_score_median_dict: Dict[int, float] = {}
        coarse_bias_vector = group_features[group_ids[0]].get("coarse_class_bias")
        if coarse_bias_vector is None:
            coarse_bias_tensor = torch.zeros(
                int(self.config.get("num_classes", 10)),
                dtype=candidate_scores.dtype,
                device=candidate_scores.device,
            )
        else:
            coarse_bias_tensor = coarse_bias_vector.float().to(candidate_scores.device)
        coarse_ranked_classes_desc = list(self._last_extract_stats.get("coarse_ranked_classes_desc", []))
        coarse_ranked_classes_asc = list(self._last_extract_stats.get("coarse_ranked_classes_asc", []))
        coarse_top1_class = int(coarse_ranked_classes_desc[0]) if len(coarse_ranked_classes_desc) > 0 else None
        protection_applied = False
        if force_known_target and known_target_label is not None:
            target_matches = torch.nonzero(candidate_classes == int(known_target_label), as_tuple=False).flatten()
            if target_matches.numel() > 0:
                selected_idx = int(target_matches[0].item())
            else:
                selected_idx = int(torch.argmax(aggregate_candidate_scores).item())
            fine_mode_used = "forced"
        else:
            high_conf_mask_raw = metadata.get("high_conf_backdoor_mask")
            high_conf_mask: Optional[torch.Tensor] = None
            if isinstance(high_conf_mask_raw, Mapping):
                mask_list = [
                    bool(high_conf_mask_raw.get(int(group_id), False))
                    for group_id in group_ids
                ]
                high_conf_mask = torch.tensor(mask_list, device=candidate_scores.device, dtype=torch.bool)
            elif isinstance(high_conf_mask_raw, Sequence) and not isinstance(high_conf_mask_raw, (str, bytes)):
                if len(high_conf_mask_raw) == len(group_ids):
                    mask_list = [bool(v) for v in high_conf_mask_raw]
                    high_conf_mask = torch.tensor(mask_list, device=candidate_scores.device, dtype=torch.bool)

            candidate_objectives: List[float] = []
            refine_values: List[float] = []
            for class_idx in range(candidate_scores.shape[1]):
                score_col = torch.nan_to_num(candidate_scores[:, class_idx], nan=0.0, posinf=0.0, neginf=0.0)
                cons_col = torch.nan_to_num(candidate_consistency[:, class_idx], nan=0.0, posinf=0.0, neginf=0.0)
                candidate_class = int(candidate_classes[class_idx].item())
                candidate_cts_vectors[candidate_class] = [float(v) for v in score_col.detach().cpu().tolist()]
                cts_tilde = torch.median(score_col)
                cons_soft = torch.median(cons_col)
                bias_val = float(coarse_bias_tensor[candidate_class].item()) if candidate_class < int(coarse_bias_tensor.numel()) else 0.0
                s_refine = 0.7 * float(cts_tilde.item()) + 0.2 * float(cons_soft.item()) - 0.1 * bias_val
                candidate_refine_scores[candidate_class] = float(s_refine)
                candidate_score_median_dict[candidate_class] = float(cts_tilde.item())
                refine_values.append(float(s_refine))

                if fine_mode_requested == "median_max":
                    separation = float(cts_tilde.item())
                elif fine_mode_requested == "separation_known":
                    if high_conf_mask is not None and bool(high_conf_mask.any().item()) and bool((~high_conf_mask).any().item()):
                        separation = float(score_col[high_conf_mask].mean().item() - score_col[~high_conf_mask].mean().item())
                    else:
                        med = torch.median(score_col)
                        mad_c = torch.median(torch.abs(score_col - med))
                        sep_sigma = torch.clamp(1.4826 * mad_c, min=1e-8)
                        separation = float(((torch.max(score_col) - med) / sep_sigma).item())
                        fine_mode_used = "separation_mad_fallback"
                else:
                    med = torch.median(score_col)
                    mad_c = torch.median(torch.abs(score_col - med))
                    sep_sigma = torch.clamp(1.4826 * mad_c, min=1e-8)
                    separation = float(((torch.max(score_col) - med) / sep_sigma).item())
                candidate_separation_scores[candidate_class] = separation
                candidate_objectives.append(separation)

            # Default refined selector: S_refine.
            refine_array = np.asarray(refine_values, dtype=np.float64)
            selected_idx = int(np.argmax(refine_array).item()) if refine_array.size > 0 else int(torch.argmax(aggregate_candidate_scores).item())
            if fine_mode_requested == "median_max":
                selected_idx = int(torch.argmax(aggregate_candidate_scores).item())

            if coarse_top1_class is not None:
                coarse_match = torch.nonzero(candidate_classes == int(coarse_top1_class), as_tuple=False).flatten()
                if coarse_match.numel() > 0 and refine_array.size > 1:
                    best_ref = float(refine_array[selected_idx])
                    coarse_ref = float(refine_array[int(coarse_match[0].item())])
                    med_ref = float(np.median(refine_array))
                    mad_ref = float(np.median(np.abs(refine_array - med_ref)))
                    protect_margin = 0.5 * max(mad_ref, 1e-8)
                    if (best_ref - coarse_ref) < protect_margin:
                        selected_idx = int(coarse_match[0].item())
                        protection_applied = True
                        fine_mode_used = f"{fine_mode_used}_coarse_protected"

            bootstrap_pseudo_y_star = metadata.get("bootstrap_pseudo_y_star")
            if bootstrap_pseudo_y_star is not None:
                matches = torch.nonzero(candidate_classes == int(bootstrap_pseudo_y_star), as_tuple=False).flatten()
                if matches.numel() > 0 and bool(metadata.get("bootstrap_promote", True)):
                    selected_idx = int(matches[0].item())
                    fine_mode_used = f"{fine_mode_used}_bootstrap_promoted"
        y_star = int(candidate_classes[selected_idx].item())
        self._last_y_star = y_star

        cts_scores_tensor = candidate_scores[:, selected_idx]
        consistency_tensor = candidate_consistency[:, selected_idx]
        cts_scores_tensor = torch.nan_to_num(cts_scores_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        consistency_tensor = torch.nan_to_num(consistency_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        raw_cts_scores_tensor = cts_scores_tensor.clone()

        cts_mode_requested = str(self.config.get("cts_mode", "raw")).strip().lower()
        if cts_mode_requested not in {"raw", "mdbf", "dcbd"}:
            cts_mode_requested = "raw"
        cts_mode_effective = "raw"
        mdbf_fallback_reason = "none"
        dcbd_fallback_reason = "none"

        num_classes_cfg = int(self.config.get("num_classes", 10))
        cts_per_class_rows: List[torch.Tensor] = []
        for group_id in group_ids:
            coarse_score = group_features[group_id].get("coarse_class_score")
            if isinstance(coarse_score, torch.Tensor):
                row = coarse_score.detach().to(device=candidate_scores.device, dtype=candidate_scores.dtype).flatten()
            else:
                row = torch.zeros(num_classes_cfg, device=candidate_scores.device, dtype=candidate_scores.dtype)
            if row.numel() < num_classes_cfg:
                pad = torch.zeros(num_classes_cfg - row.numel(), device=row.device, dtype=row.dtype)
                row = torch.cat([row, pad], dim=0)
            elif row.numel() > num_classes_cfg:
                row = row[:num_classes_cfg]
            cts_per_class_rows.append(torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0))
        cts_per_class_tensor = torch.stack(cts_per_class_rows, dim=0)
        cts_trig_per_class_rows: List[torch.Tensor] = []
        cts_clean_per_class_rows: List[torch.Tensor] = []
        for group_id in group_ids:
            trig_row_raw = group_features[group_id].get("cts_trig_per_class")
            clean_row_raw = group_features[group_id].get("cts_clean_per_class")
            if isinstance(trig_row_raw, torch.Tensor):
                trig_row = trig_row_raw.detach().to(device=candidate_scores.device, dtype=candidate_scores.dtype).flatten()
            else:
                trig_row = torch.zeros(num_classes_cfg, device=candidate_scores.device, dtype=candidate_scores.dtype)
            if isinstance(clean_row_raw, torch.Tensor):
                clean_row = clean_row_raw.detach().to(device=candidate_scores.device, dtype=candidate_scores.dtype).flatten()
            else:
                clean_row = torch.zeros(num_classes_cfg, device=candidate_scores.device, dtype=candidate_scores.dtype)
            if trig_row.numel() < num_classes_cfg:
                trig_row = torch.cat(
                    [trig_row, torch.zeros(num_classes_cfg - trig_row.numel(), device=trig_row.device, dtype=trig_row.dtype)],
                    dim=0,
                )
            elif trig_row.numel() > num_classes_cfg:
                trig_row = trig_row[:num_classes_cfg]
            if clean_row.numel() < num_classes_cfg:
                clean_row = torch.cat(
                    [clean_row, torch.zeros(num_classes_cfg - clean_row.numel(), device=clean_row.device, dtype=clean_row.dtype)],
                    dim=0,
                )
            elif clean_row.numel() > num_classes_cfg:
                clean_row = clean_row[:num_classes_cfg]
            cts_trig_per_class_rows.append(torch.nan_to_num(trig_row, nan=0.0, posinf=0.0, neginf=0.0))
            cts_clean_per_class_rows.append(torch.nan_to_num(clean_row, nan=0.0, posinf=0.0, neginf=0.0))
        cts_trig_per_class_tensor = torch.stack(cts_trig_per_class_rows, dim=0)
        cts_clean_per_class_tensor = torch.stack(cts_clean_per_class_rows, dim=0)
        tsc_target_tensor = torch.zeros_like(raw_cts_scores_tensor)
        cse_target_tensor = torch.zeros_like(raw_cts_scores_tensor)
        mdbf_target_tensor = torch.zeros_like(raw_cts_scores_tensor)
        dcbd_score_tensor = torch.zeros_like(raw_cts_scores_tensor)
        group_behavior_tensor = raw_cts_scores_tensor.clone()
        mdbf_eps = 1e-8
        mdbf_target_label = metadata.get("known_target_label", known_target_label)
        mdbf_alpha = float(np.clip(self.config.get("cts_mdbf_alpha", 0.7), 0.0, 1.0))
        mdbf_percentile = float(np.clip(self.config.get("cts_mdbf_percentile", 75.0), 0.0, 100.0))
        if cts_mode_requested == "mdbf":
            target_idx = int(mdbf_target_label) if mdbf_target_label is not None else -1
            if cts_per_class_tensor.shape[1] <= 1:
                mdbf_fallback_reason = "insufficient_num_classes"
            elif target_idx < 0 or target_idx >= int(cts_per_class_tensor.shape[1]):
                mdbf_fallback_reason = "invalid_target_label"
            else:
                target_col = cts_per_class_tensor[:, target_idx]
                keep_cols = [idx for idx in range(int(cts_per_class_tensor.shape[1])) if idx != target_idx]
                others = cts_per_class_tensor[:, keep_cols] if len(keep_cols) > 0 else cts_per_class_tensor[:, :0]
                if others.shape[1] > 0:
                    others_percentile = torch.quantile(others, q=mdbf_percentile / 100.0, dim=1)
                else:
                    others_percentile = torch.zeros_like(target_col)
                cse_target_tensor = torch.nan_to_num(target_col - others_percentile, nan=0.0, posinf=0.0, neginf=0.0)
                cts_mean = torch.mean(cts_per_class_tensor, dim=1)
                cts_std = torch.std(cts_per_class_tensor, dim=1, unbiased=False)
                mdbf_target_tensor = torch.nan_to_num(
                    (target_col - cts_mean) / (cts_std + mdbf_eps),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                group_behavior_tensor = torch.nan_to_num(
                    mdbf_alpha * mdbf_target_tensor + (1.0 - mdbf_alpha) * cse_target_tensor,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cts_scores_tensor = group_behavior_tensor.clone()
                cts_mode_effective = "mdbf"
        elif cts_mode_requested == "dcbd":
            use_raw_logit = bool(self.config.get("cts_dcbd_use_raw_logit", True))
            if not use_raw_logit and dcbd_fallback_reason == "none":
                dcbd_fallback_reason = "raw_logit_flag_disabled_fallback_to_raw"
            target_idx = int(mdbf_target_label) if mdbf_target_label is not None else -1
            if cts_trig_per_class_tensor.shape[1] <= 1:
                dcbd_fallback_reason = "insufficient_num_classes"
            elif target_idx < 0 or target_idx >= int(cts_trig_per_class_tensor.shape[1]):
                dcbd_fallback_reason = "invalid_target_label"
            else:
                target_trig_col = cts_trig_per_class_tensor[:, target_idx]
                target_clean_col = cts_clean_per_class_tensor[:, target_idx]
                tsc_target_tensor = torch.nan_to_num(
                    target_trig_col - target_clean_col,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                trig_mean = torch.mean(cts_trig_per_class_tensor, dim=1)
                trig_std = torch.std(cts_trig_per_class_tensor, dim=1, unbiased=False)
                mdbf_target_tensor = torch.nan_to_num(
                    (target_trig_col - trig_mean) / (trig_std + mdbf_eps),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                keep_cols = [idx for idx in range(int(cts_trig_per_class_tensor.shape[1])) if idx != target_idx]
                others = (
                    cts_trig_per_class_tensor[:, keep_cols]
                    if len(keep_cols) > 0
                    else cts_trig_per_class_tensor[:, :0]
                )
                if others.shape[1] > 0:
                    others_percentile = torch.quantile(others, q=mdbf_percentile / 100.0, dim=1)
                else:
                    others_percentile = torch.zeros_like(target_trig_col)
                cse_target_tensor = torch.nan_to_num(
                    target_trig_col - others_percentile,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                dcbd_simple = bool(self.config.get("cts_dcbd_simple", False))
                dcbd_alpha = float(self.config.get("cts_dcbd_alpha", 0.3))
                tsc_weight = float(self.config.get("cts_dcbd_tsc_weight", 0.5))
                mdbf_weight = float(self.config.get("cts_dcbd_mdbf_weight", 0.35))
                cse_weight = float(self.config.get("cts_dcbd_cse_weight", 0.15))
                if dcbd_simple:
                    dcbd_score_tensor = torch.nan_to_num(
                        tsc_target_tensor + dcbd_alpha * mdbf_target_tensor,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                else:
                    dcbd_score_tensor = torch.nan_to_num(
                        tsc_weight * tsc_target_tensor
                        + mdbf_weight * mdbf_target_tensor
                        + cse_weight * cse_target_tensor,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                group_behavior_tensor = dcbd_score_tensor.clone()
                cts_scores_tensor = dcbd_score_tensor.clone()
                cts_mode_effective = "dcbd"

        q1_active = torch.quantile(cts_scores_tensor, q=0.25) if cts_scores_tensor.numel() > 0 else torch.tensor(0.0, device=cts_scores_tensor.device)
        q3_active = torch.quantile(cts_scores_tensor, q=0.75) if cts_scores_tensor.numel() > 0 else torch.tensor(1.0, device=cts_scores_tensor.device)
        iqr_active = torch.clamp(q3_active - q1_active, min=1e-6)
        cts_asinh_tensor = torch.asinh(cts_scores_tensor / iqr_active)

        temporal_used = bool(self.config.get("use_temporal_aggregation", False))
        temporal_skip_update = bool(metadata.get("temporal_skip_update", False))
        temporal_effective_rounds = 1
        if temporal_used and (not temporal_skip_update):
            decay = min(max(float(self.config.get("temporal_decay", 0.85)), 0.0), 1.0)
            history_limit = max(int(self.config.get("temporal_history_max", 30)), 1)
            round_value = int(-1 if round_idx is None else round_idx)
            group_temporal_keys = metadata.get("group_temporal_keys", {})
            aggregated_values: List[float] = []
            history_lengths: List[int] = []
            for idx, group_id in enumerate(group_ids):
                key = str(group_temporal_keys.get(int(group_id), group_id))
                history = list(self._temporal_cts_history.get(key, []))
                history.append((round_value, float(cts_scores_tensor[idx].item())))
                history = history[-history_limit:]
                self._temporal_cts_history[key] = history

                weighted_sum = 0.0
                weighted_den = 0.0
                for hist_round, hist_score in history:
                    delta = max(round_value - int(hist_round), 0)
                    w = decay ** delta
                    weighted_sum += w * float(hist_score)
                    weighted_den += w
                aggregated_values.append(float(weighted_sum / max(weighted_den, 1e-12)))
                history_lengths.append(len(history))
            if len(aggregated_values) == len(group_ids):
                cts_scores_tensor = torch.tensor(
                    aggregated_values,
                    device=cts_scores_tensor.device,
                    dtype=cts_scores_tensor.dtype,
                )
                temporal_effective_rounds = int(max(history_lengths)) if len(history_lengths) > 0 else 1

        # Round-wise differential normalization to remove global baseline drift.
        cts_round_median = torch.median(cts_scores_tensor) if cts_scores_tensor.numel() > 0 else torch.tensor(0.0, device=cts_scores_tensor.device)
        cts_diff_tensor = cts_scores_tensor - cts_round_median

        preprocess_mode = str(self.config.get("mad_preprocess", "iqr_clip")).strip().lower()
        if preprocess_mode not in {"none", "iqr_clip", "tanh_iqr"}:
            preprocess_mode = "iqr_clip"
        iqr_mult = float(self.config.get("mad_clip_iqr_multiplier", 3.0))
        iqr_mult = max(1.0, iqr_mult)
        score_for_threshold, preprocess_stats = self._preprocess_scores_for_mad(
            cts_diff_tensor,
            mode=preprocess_mode,
            iqr_multiplier=iqr_mult,
        )

        median_score = torch.median(score_for_threshold)
        mad = torch.median(torch.abs(score_for_threshold - median_score))
        robust_sigma = 1.4826 * mad
        score_q1 = torch.quantile(score_for_threshold, q=0.25) if score_for_threshold.numel() > 0 else torch.tensor(0.0, device=score_for_threshold.device)
        score_q3 = torch.quantile(score_for_threshold, q=0.75) if score_for_threshold.numel() > 0 else torch.tensor(0.0, device=score_for_threshold.device)
        score_iqr = torch.clamp(score_q3 - score_q1, min=0.0)
        mad_sigma_floor_abs = float(max(self.config.get("mad_sigma_floor_abs", 1e-3), 1e-8))
        mad_sigma_floor_rel_iqr = float(max(self.config.get("mad_sigma_floor_rel_iqr", 0.10), 0.0))
        mad_sigma_floor = max(
            mad_sigma_floor_abs,
            mad_sigma_floor_rel_iqr * float(score_iqr.item()),
        )
        robust_sigma = torch.clamp(robust_sigma, min=mad_sigma_floor)
        sigma_safe = torch.clamp(robust_sigma, min=1e-12)
        mad_kappa = float(self.config["mad_kappa"])
        bimodal_bc = self._compute_bimodality_coefficient(score_for_threshold)
        bimodal_threshold = float(self.config.get("bimodal_bc_threshold", 0.555))
        bimodal_kappa_override = float(self.config.get("bimodal_kappa_override", 2.0))
        is_bimodal = bool(np.isfinite(bimodal_bc) and bimodal_bc >= bimodal_threshold)
        effective_kappa = float(mad_kappa)
        if is_bimodal and bimodal_kappa_override > 0.0:
            effective_kappa = min(effective_kappa, float(bimodal_kappa_override))

        right_threshold = median_score + effective_kappa * robust_sigma
        left_threshold = median_score - effective_kappa * robust_sigma

        # Primary detector for b_g: differential CTS + MAD thresholding.
        hard_right_flags = score_for_threshold > right_threshold
        hard_left_flags = score_for_threshold < left_threshold

        right_mad_z = (score_for_threshold - median_score) / sigma_safe
        left_mad_z = (median_score - score_for_threshold) / sigma_safe

        tail_mode = str(self.config.get("anomaly_tail", "right")).strip().lower()
        if tail_mode not in {"auto", "right", "left", "two_sided"}:
            tail_mode = "right"
        expected_direction = str(self.config.get("expected_score_direction", "auto")).strip().lower()
        if expected_direction == "backdoor_higher_cts":
            selected_tail = "right"
        elif expected_direction == "backdoor_lower_cts":
            selected_tail = "left"
        elif tail_mode in {"auto", "two_sided"}:
            tail_selection_min_consistency = float(
                np.clip(self.config.get("tail_selection_min_consistency", 0.40), 0.0, 1.0)
            )
            effective_tail_cons_thr = max(float(tau_cons), float(tail_selection_min_consistency))
            if single_trigger_mode and disable_consistency_gate_single:
                eligible = torch.ones_like(consistency_tensor, dtype=torch.bool)
            else:
                eligible = consistency_tensor >= effective_tail_cons_thr
            if not bool(eligible.any().item()):
                eligible = torch.ones_like(consistency_tensor, dtype=torch.bool)
            right_strength = torch.max((score_for_threshold[eligible] - median_score) / sigma_safe)
            left_strength = torch.max((median_score - score_for_threshold[eligible]) / sigma_safe)
            selected_tail = "right" if float(right_strength.item()) >= float(left_strength.item()) else "left"
        else:
            selected_tail = tail_mode

        # Direction-aligned CTS robust z-score (based on differential CTS).
        oriented_cts = cts_diff_tensor if selected_tail == "right" else -cts_diff_tensor
        cts_center = torch.median(oriented_cts)
        cts_mad = torch.median(torch.abs(oriented_cts - cts_center))
        cts_q1 = torch.quantile(oriented_cts, q=0.25) if oriented_cts.numel() > 0 else torch.tensor(0.0, device=oriented_cts.device)
        cts_q3 = torch.quantile(oriented_cts, q=0.75) if oriented_cts.numel() > 0 else torch.tensor(0.0, device=oriented_cts.device)
        cts_iqr = torch.clamp(cts_q3 - cts_q1, min=0.0)
        cts_sigma_floor_abs = float(max(self.config.get("cts_sigma_floor_abs", 1e-3), 1e-8))
        cts_sigma_floor_rel_iqr = float(max(self.config.get("cts_sigma_floor_rel_iqr", 0.10), 0.0))
        cts_sigma_floor = max(
            cts_sigma_floor_abs,
            cts_sigma_floor_rel_iqr * float(cts_iqr.item()),
        )
        cts_sigma = torch.clamp(1.4826 * cts_mad, min=cts_sigma_floor)
        z_cts_dir = (oriented_cts - cts_center) / cts_sigma

        # Consistency robust z-score in logit space.
        cons_eps = 1e-6
        cons_clipped = torch.clamp(consistency_tensor, min=cons_eps, max=1.0 - cons_eps)
        cons_logit = torch.log(cons_clipped) - torch.log1p(-cons_clipped)
        cons_center = torch.median(cons_logit)
        cons_mad = torch.median(torch.abs(cons_logit - cons_center))
        cons_sigma_floor_abs = float(max(self.config.get("cons_sigma_floor_abs", 0.05), 1e-8))
        cons_sigma = torch.clamp(1.4826 * cons_mad, min=cons_sigma_floor_abs)
        z_cons = (cons_logit - cons_center) / cons_sigma

        # Main detector score: continuous joint score.
        if single_trigger_mode:
            cons_weight = float(self.config.get("single_trigger_cons_weight", 0.0))
        else:
            cons_weight = float(self.config.get("score_joint_cons_weight", 0.3))
        score_joint = torch.nan_to_num(z_cts_dir + cons_weight * z_cons, nan=0.0, posinf=0.0, neginf=0.0)
        joint_center = torch.median(score_joint)
        joint_mad = torch.median(torch.abs(score_joint - joint_center))
        joint_sigma_floor_abs = float(max(self.config.get("joint_sigma_floor_abs", 0.05), 1e-8))
        joint_sigma = torch.clamp(1.4826 * joint_mad, min=joint_sigma_floor_abs)
        joint_threshold = joint_center + effective_kappa * joint_sigma
        joint_flags = score_joint > joint_threshold

        z_clip = float(max(self.config.get("z_score_clip", 12.0), 0.0))
        if z_clip > 0.0:
            right_mad_z = torch.clamp(right_mad_z, min=-z_clip, max=z_clip)
            left_mad_z = torch.clamp(left_mad_z, min=-z_clip, max=z_clip)
            z_cts_dir = torch.clamp(z_cts_dir, min=-z_clip, max=z_clip)
            z_cons = torch.clamp(z_cons, min=-z_clip, max=z_clip)
            score_joint = torch.clamp(score_joint, min=-z_clip, max=z_clip)

        primary_flags = hard_right_flags if selected_tail == "right" else hard_left_flags
        two_sided_enabled = bool(self.config.get("enable_two_sided_anomaly", True))
        secondary_consistency_min = float(
            self.config.get("two_sided_secondary_min_consistency", 0.10)
        )
        secondary_mad_z_min = float(
            self.config.get("two_sided_secondary_min_mad_z", 1.0)
        )
        secondary_flags = torch.zeros_like(primary_flags, dtype=torch.bool)
        if two_sided_enabled:
            opposite_hard = hard_left_flags if selected_tail == "right" else hard_right_flags
            opposite_z = left_mad_z if selected_tail == "right" else right_mad_z
            secondary_flags = (
                opposite_hard
                & (consistency_tensor >= secondary_consistency_min)
                & (opposite_z >= secondary_mad_z_min)
            )
        if not (single_trigger_mode and disable_consistency_gate_single):
            primary_min_consistency = float(
                np.clip(self.config.get("primary_min_consistency", 0.35), 0.0, 1.0)
            )
            primary_flags = primary_flags & (consistency_tensor >= primary_min_consistency)
        flags = primary_flags | secondary_flags
        if bool(self.config.get("two_sided_use_abs_anomaly_score", True)) and two_sided_enabled:
            anomaly_scores_tensor = torch.maximum(right_mad_z, left_mad_z)
        else:
            anomaly_scores_tensor = z_cts_dir
        if bool(self.config.get("primary_require_joint", True)):
            primary_keep_if_strong_z = float(
                max(self.config.get("primary_keep_if_strong_z", 3.5), 0.0)
            )
            strong_primary_mask = anomaly_scores_tensor >= primary_keep_if_strong_z
            primary_flags = primary_flags & (joint_flags | strong_primary_mask)
            flags = primary_flags | secondary_flags
        threshold = right_threshold if selected_tail == "right" else left_threshold

        fallback_enabled = bool(self.config.get("fallback_enabled", True))
        fallback_triggered = False
        fallback_selected_indices: List[int] = []
        fallback_score_gap = 0.0
        fallback_top_score = float(anomaly_scores_tensor.max().item()) if anomaly_scores_tensor.numel() > 0 else 0.0
        if fallback_enabled and (not bool(flags.any().item())):
            if single_trigger_mode and disable_consistency_gate_single:
                fallback_consistency_threshold = float(
                    self.config.get("single_trigger_fallback_min_consistency", 0.0)
                )
            else:
                fallback_consistency_threshold = max(
                    tau_cons,
                    float(self.config.get("fallback_min_consistency", tau_cons)),
                )
            candidate_mask = consistency_tensor >= fallback_consistency_threshold
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            if candidate_indices.numel() > 0:
                fallback_count = max(
                    int(self.config.get("fallback_top_k", 1)),
                    int(torch.ceil(torch.tensor(len(group_ids) * float(self.config.get("fallback_top_fraction", 0.0)))).item()),
                )
                fallback_count = min(int(candidate_indices.numel()), max(fallback_count, 1))
                ranked_indices = torch.argsort(
                    anomaly_scores_tensor[candidate_indices],
                    descending=True,
                )
                ranked_candidates = candidate_indices[ranked_indices]
                top_idx = ranked_candidates[0]
                top_score = float(anomaly_scores_tensor[top_idx].item())
                median_value = float(torch.median(anomaly_scores_tensor).item())
                second_score = (
                    float(anomaly_scores_tensor[ranked_candidates[1]].item())
                    if ranked_candidates.numel() > 1
                    else median_value
                )
                score_gap = max(top_score - second_score, top_score - median_value)
                fallback_score_gap = float(score_gap)
                min_gap = max(
                    float(self.config.get("fallback_min_score_gap", 0.25)),
                    float(self.config.get("fallback_min_score_sigma", 0.5)) * float(cts_sigma.item()),
                )
                if score_gap >= min_gap:
                    chosen = ranked_candidates[:fallback_count]
                    flags = torch.zeros_like(flags, dtype=torch.bool)
                    flags[chosen] = True
                    fallback_selected_indices = [int(index.item()) for index in chosen]
                    fallback_triggered = True

        right_flags_final = hard_right_flags & flags
        left_flags_final = hard_left_flags & flags
        hard_flags_selected = hard_right_flags if selected_tail == "right" else hard_left_flags

        per_trigger_weight_mean: Dict[str, float] = {}
        if len(group_ids) > 0 and "candidate_trigger_weights" in group_features[group_ids[0]]:
            try:
                trigger_weight_rows: List[torch.Tensor] = []
                for group_id in group_ids:
                    tw = group_features[group_id]["candidate_trigger_weights"].float()
                    trigger_weight_rows.append(tw[selected_idx])
                trigger_weight_tensor = torch.stack(trigger_weight_rows, dim=0)
                trigger_weight_avg = torch.nan_to_num(trigger_weight_tensor.mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
                trigger_names = ["patch", "blend", "sig", "color"]
                for idx, name in enumerate(trigger_names[: int(trigger_weight_avg.numel())]):
                    per_trigger_weight_mean[str(name)] = float(trigger_weight_avg[idx].item())
            except Exception:
                per_trigger_weight_mean = {}

        group_scores = {
            group_id: float(cts_scores_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        raw_group_scores = {
            group_id: float(raw_cts_scores_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        diff_group_scores = {
            group_id: float(cts_diff_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        cts_asinh_scores = {
            group_id: float(cts_asinh_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        raw_cts_target_scores = {
            group_id: float(raw_cts_scores_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        cts_per_class_scores = {
            group_id: [float(v) for v in cts_per_class_tensor[idx].detach().cpu().tolist()]
            for idx, group_id in enumerate(group_ids)
        }
        cse_target_scores = {
            group_id: float(cse_target_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        mdbf_target_scores = {
            group_id: float(mdbf_target_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        group_behavior_scores = {
            group_id: float(group_behavior_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        cts_trig_per_class_scores = {
            group_id: [float(v) for v in cts_trig_per_class_tensor[idx].detach().cpu().tolist()]
            for idx, group_id in enumerate(group_ids)
        }
        cts_clean_per_class_scores = {
            group_id: [float(v) for v in cts_clean_per_class_tensor[idx].detach().cpu().tolist()]
            for idx, group_id in enumerate(group_ids)
        }
        tsc_target_scores = {
            group_id: float(tsc_target_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        dcbd_score_scores = {
            group_id: float(dcbd_score_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }
        flagged_groups = [
            group_id for idx, group_id in enumerate(group_ids) if bool(flags[idx].item())
        ]
        dcbd_anomaly_z_tensor = torch.maximum(right_mad_z, left_mad_z)
        dcbd_anomaly_z_scores = {
            group_id: float(dcbd_anomaly_z_tensor[idx].item())
            for idx, group_id in enumerate(group_ids)
        }

        result = CTSIntentResult(
            round_idx=-1 if round_idx is None else round_idx,
            group_scores=group_scores,
            group_features={group_id: dict(group_features[group_id]) for group_id in group_ids},
            flagged_groups=flagged_groups,
            aux_stats={
                "y_star": y_star,
                "candidate_classes": candidate_classes.tolist(),
                "candidate_score_median_global": aggregate_candidate_scores.tolist(),
                "coarse_ranked_classes": list(self._last_extract_stats.get("coarse_ranked_classes", [])),
                "coarse_mode": str(self._last_extract_stats.get("coarse_mode", self.config.get("coarse_mode", "margin"))),
                "coarse_top3_classes": [
                    int(v) for v in self._last_extract_stats.get("coarse_ranked_classes", [])[:3]
                ],
                "coarse_skipped": bool(self._last_extract_stats.get("coarse_skipped", False)),
                "forward_stats": dict(self._last_extract_stats.get("forward_stats", {})),
                "finite_group_count": int(torch.isfinite(cts_scores_tensor).sum().item()),
                "y_star_source": "forced" if force_known_target and known_target_label is not None else "estimated",
                "fine_mode_used": str(fine_mode_used),
                "candidate_separation_scores": {
                    int(k): float(v) for k, v in candidate_separation_scores.items()
                },
                "candidate_refine_scores": {
                    int(k): float(v) for k, v in candidate_refine_scores.items()
                },
                "candidate_score_median": {
                    int(k): float(v) for k, v in candidate_score_median_dict.items()
                },
                "coarse_ranked_classes_desc": [int(v) for v in coarse_ranked_classes_desc],
                "coarse_ranked_classes_asc": [int(v) for v in coarse_ranked_classes_asc],
                "coarse_top1_class": coarse_top1_class,
                "refine_protection_applied": bool(protection_applied),
                "candidate_cts_vectors": {
                    int(k): [float(x) for x in v] for k, v in candidate_cts_vectors.items()
                },
                "consistency": {
                    group_id: float(consistency_tensor[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "consistency_components": {
                    "a": {
                        group_id: float(group_features[group_id]["candidate_cons_a"][selected_idx].item())
                        for group_id in group_ids
                    },
                    "u": {
                        group_id: float(group_features[group_id]["candidate_cons_u"][selected_idx].item())
                        for group_id in group_ids
                    },
                    "v": {
                        group_id: float(group_features[group_id]["candidate_cons_v"][selected_idx].item())
                        for group_id in group_ids
                    },
                },
                "b_g": {
                    group_id: bool(flags[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "b_g_joint": {
                    group_id: bool(joint_flags[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "b_g_hard": {
                    group_id: bool(hard_flags_selected[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "b_g_hard_right": {
                    group_id: bool(hard_right_flags[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "b_g_hard_left": {
                    group_id: bool(hard_left_flags[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "b_g_right": {
                    group_id: bool(right_flags_final[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "b_g_left": {
                    group_id: bool(left_flags_final[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "anomaly_scores": {
                    group_id: float(anomaly_scores_tensor[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "score_joint": {
                    group_id: float(score_joint[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "z_cts_dir": {
                    group_id: float(z_cts_dir[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "z_cons": {
                    group_id: float(z_cons[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "raw_group_scores": raw_group_scores,
                "diff_group_scores": diff_group_scores,
                "cts_asinh_scores": cts_asinh_scores,
                "cts_mode_requested": str(cts_mode_requested),
                "cts_mode_effective": str(cts_mode_effective),
                "mdbf_fallback_reason": str(mdbf_fallback_reason),
                "dcbd_fallback_reason": str(dcbd_fallback_reason),
                "mdbf_alpha": float(mdbf_alpha),
                "mdbf_percentile": float(mdbf_percentile),
                "raw_cts_target": raw_cts_target_scores,
                "cts_per_class": cts_per_class_scores,
                "cts_trig_per_class": cts_trig_per_class_scores,
                "cts_clean_per_class": cts_clean_per_class_scores,
                "tsc_target": tsc_target_scores,
                "cse_target": cse_target_scores,
                "mdbf_target": mdbf_target_scores,
                "dcbd_score": dcbd_score_scores,
                "dcbd_scores_all_groups": [float(v) for v in group_behavior_tensor.detach().cpu().tolist()],
                "dcbd_mad_median": float(median_score.item()),
                "dcbd_mad_sigma": float(robust_sigma.item()),
                "dcbd_threshold": float(threshold.item()),
                "group_behavior_score": group_behavior_scores,
                "anomaly_decision": {
                    group_id: bool(flags[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "dcbd_anomaly_decision": {
                    group_id: bool(flags[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "dcbd_anomaly_z_score": dcbd_anomaly_z_scores,
                "right_mad_z": {
                    group_id: float(right_mad_z[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "left_mad_z": {
                    group_id: float(left_mad_z[idx].item())
                    for idx, group_id in enumerate(group_ids)
                },
                "anomaly_tail": selected_tail,
                "two_sided_enabled": bool(two_sided_enabled),
                "two_sided_secondary_count": int(secondary_flags.sum().item()),
                "two_sided_primary_count": int(primary_flags.sum().item()),
                "selected_detector_variant": "joint_continuous",
                "mad_median": float(median_score.item()),
                "mad_sigma": float(robust_sigma.item()),
                "mad_sigma_floor": float(mad_sigma_floor),
                "mad_threshold": float(threshold.item()),
                "cts_round_median": float(cts_round_median.item()),
                "joint_median": float(joint_center.item()),
                "joint_sigma": float(joint_sigma.item()),
                "joint_threshold": float(joint_threshold.item()),
                "right_mad_threshold": float(right_threshold.item()),
                "left_mad_threshold": float(left_threshold.item()),
                "mad_preprocess": preprocess_mode,
                "mad_preprocess_stats": preprocess_stats,
                "mad_kappa": float(mad_kappa),
                "mad_kappa_effective": float(effective_kappa),
                "bimodal_coefficient": float(bimodal_bc),
                "bimodal_detected": bool(is_bimodal),
                "expected_score_direction": expected_direction,
                "optimization_flags": {
                    "temporal_aggregation": bool(temporal_used),
                    "temporal_skip_update": bool(temporal_skip_update),
                    "adaptive_samples": False,
                    "ensemble_voting": bool(self.config.get("use_ensemble", False)),
                },
                "num_triggers_effective": int(num_triggers_effective),
                "single_trigger_mode": bool(single_trigger_mode),
                "consistency_weight": float(cons_weight),
                "consistency_family_blend": float(self.config.get("consistency_family_blend", 0.65)),
                "temporal_aggregation_used": bool(temporal_used and (not temporal_skip_update)),
                "effective_history_rounds": int(temporal_effective_rounds),
                "z_score_clip": float(z_clip),
                "cts_sigma_floor": float(cts_sigma_floor),
                "cons_sigma_floor": float(cons_sigma_floor_abs),
                "joint_sigma_floor": float(joint_sigma_floor_abs),
                "per_trigger_weights": per_trigger_weight_mean,
                "fallback_triggered": bool(fallback_triggered),
                "fallback_enabled": bool(fallback_enabled),
                "fallback_selected_indices": list(fallback_selected_indices),
                "fallback_selected_groups": [group_ids[idx] for idx in fallback_selected_indices],
                "fallback_score_gap": float(fallback_score_gap),
                "fallback_top_score": float(fallback_top_score),
                "num_joint_positive": int(flags.sum().item()),
                "num_hard_positive": int(hard_flags_selected.sum().item()),
                "group_order": list(group_ids),
            },
        )
        if len(candidate_separation_scores) > 0:
            ranked = sorted(candidate_separation_scores.items(), key=lambda kv: kv[1], reverse=True)
            separation_winner = int(ranked[0][0])
            separation_margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else float("inf")
            result.aux_stats["separation_winner"] = separation_winner
            result.aux_stats["separation_margin"] = separation_margin
        return result

    def detect(
        self,
        group_updates: Mapping[int, ModelUpdate],
        global_model: Optional[Module] = None,
        clean_dataloader: Optional[DataLoader] = None,
        round_idx: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CTSIntentResult:
        """Run the full CTS-Intent detection pipeline for one round."""

        group_features = self.extract_behavior_signals(
            group_updates=group_updates,
            global_model=global_model,
            clean_dataloader=clean_dataloader,
            round_idx=round_idx,
            metadata=metadata,
        )
        return self.score_groups(group_features, round_idx=round_idx, metadata=metadata)

    def run(
        self,
        group_updates: Union[Mapping[int, ModelUpdate], Sequence[ModelUpdate], torch.Tensor],
        w_t: Optional[Union[Module, Mapping[str, torch.Tensor]]] = None,
        x_syn: Optional[Union[torch.Tensor, DataLoader]] = None,
        round_idx: Optional[int] = None,
    ) -> Tuple[List[float], List[bool], int]:
        """
        Convenience wrapper matching the requested CTS-Intent I/O.

        Args:
            group_updates: Group aggregated updates ``Delta_G``.
            w_t: Current global parameter state or model snapshot.
            x_syn: Optional synthetic probe set. If omitted, 100 random CIFAR-10
                test images are used.
            round_idx: Optional round index.

        Returns:
            ``(cts_scores, b_g, y_star)`` in the same group order as the input.
        """

        metadata: Dict[str, Any] = {"w_t": w_t, "x_syn": x_syn}
        result = self.detect(
            group_updates=self._coerce_group_updates_to_mapping(group_updates),
            global_model=w_t if isinstance(w_t, Module) else None,
            clean_dataloader=x_syn if isinstance(x_syn, DataLoader) else None,
            round_idx=round_idx,
            metadata=metadata,
        )
        ordered_group_ids = result.aux_stats.get("group_order", self._last_group_order)
        flags = result.aux_stats.get("b_g", {})
        cts_scores = [float(result.group_scores[group_id]) for group_id in ordered_group_ids]
        b_g = [bool(flags[group_id]) for group_id in ordered_group_ids]
        y_star = int(result.aux_stats["y_star"])
        return cts_scores, b_g, y_star

    def update_reference(
        self,
        clean_group_updates: Optional[Mapping[int, ModelUpdate]] = None,
        clean_dataloader: Optional[DataLoader] = None,
        global_model: Optional[Module] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Refresh clean reference statistics used by intent scoring."""

        metadata = dict(metadata or {})
        self._reference_state = self._resolve_reference_state(global_model=global_model, metadata=metadata)
        if metadata.get("x_syn") is not None or clean_dataloader is not None:
            self._probe_cache = self._resolve_probe_set(clean_dataloader=clean_dataloader, metadata=metadata)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize CTS-Intent state."""

        return {
            "config": dict(self.config),
            "probe_cache": None if self._probe_cache is None else self._probe_cache.cpu(),
            "last_group_order": list(self._last_group_order),
            "last_y_star": self._last_y_star,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore CTS-Intent state."""

        self.config = self._build_config(state_dict.get("config", self.config))
        probe_cache = state_dict.get("probe_cache")
        self._probe_cache = None if probe_cache is None else probe_cache.to(self.device)
        self._last_group_order = list(state_dict.get("last_group_order", []))
        self._last_y_star = state_dict.get("last_y_star")

    def _build_config(self, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = {
            "dataset": "cifar10",
            "dataset_path": "./data",
            "probe_size": 100,
            "probe_seed": 1234,
            "seed": 1234,
            "num_classes": 10,
            "batch_size": 256,
            "topk_candidates": 3,
            "force_known_target": False,
            "known_target_label": None,
            "coarse_trigger": "patch",
            "coarse_mode": "margin",
            "trigger_types": ["patch", "blend", "sig", "color"],
            "coarse_ratio_eps": 1e-8,
            "coarse_bias_coeff": 0.35,
            "coarse_bias_temperature": 1.0,
            "coarse_top_high": 3,
            "coarse_top_low": 2,
            "patch_size": 3,
            "blend_alpha": 0.2,
            "sig_delta": 0.2,
            "sig_frequency": 6.0,
            "sig_phase": 0.0,
            "color_offsets": [0.12, -0.08, 0.08],
            "consistency_margin_eps": 0.0,
            "cons_temp_prob": 1.0,
            "cons_temp_margin": 1.0,
            "cons_tau_v": 1.0,
            "tau_cons": 0.6,
            "cts_mode": "raw",
            "cts_mdbf_alpha": 0.7,
            "cts_mdbf_percentile": 75.0,
            "cts_dcbd_tsc_weight": 0.5,
            "cts_dcbd_mdbf_weight": 0.35,
            "cts_dcbd_cse_weight": 0.15,
            "cts_dcbd_use_raw_logit": True,
            "cts_dcbd_simple": False,
            "cts_dcbd_alpha": 0.3,
            "score_joint_cons_weight": 0.3,
            "single_trigger_cons_weight": 0.0,
            "single_trigger_disable_consistency_gate": True,
            "single_trigger_fallback_min_consistency": 0.0,
            "consistency_family_blend": 0.45,
            "tail_selection_min_consistency": 0.40,
            "primary_min_consistency": 0.35,
            "primary_require_joint": True,
            "primary_keep_if_strong_z": 3.5,
            "mad_kappa": 2.0,
            "mad_preprocess": "iqr_clip",
            "mad_clip_iqr_multiplier": 3.0,
            "mad_sigma_floor_abs": 1e-3,
            "mad_sigma_floor_rel_iqr": 0.10,
            "cts_sigma_floor_abs": 1e-3,
            "cts_sigma_floor_rel_iqr": 0.10,
            "cons_sigma_floor_abs": 0.05,
            "joint_sigma_floor_abs": 0.05,
            "z_score_clip": 12.0,
            "bimodal_bc_threshold": 0.555,
            "bimodal_kappa_override": 2.0,
            "anomaly_tail": "auto",
            "enable_two_sided_anomaly": True,
            "two_sided_secondary_min_mad_z": 1.0,
            "two_sided_secondary_min_consistency": 0.10,
            "two_sided_use_abs_anomaly_score": True,
            "expected_score_direction": "auto",
            "coarse_aggregate_quantile": 0.8,
            "y_star_selection": "tail_topk_mean",
            "y_star_tail_top_fraction": 0.2,
            "fine_mode": "separation_mad",
            "use_temporal_aggregation": False,
            "temporal_decay": 0.85,
            "temporal_history_max": 30,
            "use_ensemble": False,
            "ensemble_mode": "weighted_vote",
            "ensemble_temperature": 1.0,
            "fallback_enabled": True,
            "skip_coarse_when_known_target": False,
            "fallback_top_k": 1,
            "fallback_top_fraction": 0.0,
            "fallback_min_consistency": 0.8,
            "fallback_min_score_gap": 0.25,
            "fallback_min_score_sigma": 0.5,
            "normalization_mean": [0.4914, 0.4822, 0.4465],
            "normalization_std": [0.2470, 0.2430, 0.2610],
            "phi": "tanh",
            "phi_temperature": 1.0,
            "cts_score_clip": 2.0,
            "coarse_score_clip": 2.0,
            "auto_denormalize_probe": True,
            "auto_logit_from_probability_output": True,
        }
        if config is not None:
            merged.update(dict(config))
        trigger_types = merged.get("trigger_types", ["patch", "blend", "sig", "color"])
        if isinstance(trigger_types, str):
            tokens = [token.strip().lower() for token in trigger_types.split(",") if token.strip()]
            merged["trigger_types"] = tokens if len(tokens) > 0 else ["patch", "blend", "sig", "color"]
        elif isinstance(trigger_types, Sequence):
            tokens = [str(token).strip().lower() for token in trigger_types if str(token).strip()]
            merged["trigger_types"] = tokens if len(tokens) > 0 else ["patch", "blend", "sig", "color"]
        else:
            merged["trigger_types"] = ["patch", "blend", "sig", "color"]
        return merged

    def _resolve_device(self, device: Optional[Union[str, torch.device]], model: Module) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _coerce_group_updates_to_mapping(
        self,
        group_updates: Union[Mapping[int, ModelUpdate], Sequence[ModelUpdate], torch.Tensor],
    ) -> Mapping[int, ModelUpdate]:
        if isinstance(group_updates, Mapping):
            return group_updates
        if isinstance(group_updates, torch.Tensor):
            if group_updates.dim() == 1:
                return {0: group_updates}
            if group_updates.dim() == 2:
                return {idx: group_updates[idx] for idx in range(group_updates.shape[0])}
            raise ValueError("group_updates tensor must be 1D or 2D.")
        return {idx: update for idx, update in enumerate(group_updates)}

    def _normalize_group_updates(
        self,
        group_updates: Union[Mapping[int, ModelUpdate], Sequence[ModelUpdate], torch.Tensor],
    ) -> List[Tuple[int, ModelUpdate]]:
        mapping = self._coerce_group_updates_to_mapping(group_updates)
        return list(mapping.items())

    def _resolve_reference_state(
        self,
        global_model: Optional[Module] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OrderedDict[str, torch.Tensor]:
        metadata = dict(metadata or {})
        cached_state = metadata.get("reference_state_cache")
        if isinstance(cached_state, Mapping):
            model_state = self.model.state_dict()
            cache_valid = True
            for key in model_state.keys():
                if key not in cached_state or not isinstance(cached_state[key], torch.Tensor):
                    cache_valid = False
                    break
            if cache_valid:
                return OrderedDict(
                    (key, cached_state[key].detach().to(self.device))
                    for key in model_state.keys()
                )

        w_t = metadata.get("w_t")
        state_source: Optional[Union[Module, Mapping[str, torch.Tensor]]] = None
        if w_t is not None:
            state_source = w_t
        elif global_model is not None:
            state_source = global_model
        else:
            state_source = self.model

        if isinstance(state_source, Module):
            state_dict = state_source.state_dict()
        else:
            state_dict = state_source

        model_state = self.model.state_dict()
        resolved = OrderedDict()
        for key, tensor in model_state.items():
            source_tensor = state_dict[key] if state_dict is not None and key in state_dict else tensor
            resolved[key] = source_tensor.detach().clone().to(self.device)
        return resolved

    def _resolve_probe_set(
        self,
        clean_dataloader: Optional[DataLoader] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        metadata = dict(metadata or {})
        x_syn = metadata.get("x_syn")
        if isinstance(x_syn, torch.Tensor):
            return x_syn.detach().clone().float().to(self.device)
        if self._probe_cache is not None:
            return self._probe_cache.to(self.device)
        if clean_dataloader is not None:
            probe = self._collect_probe_from_dataloader(clean_dataloader)
            self._probe_cache = probe.detach().cpu()
            return probe

        probe = self._load_cifar10_probe_set()
        self._probe_cache = probe.detach().cpu()
        return probe

    def _collect_probe_from_dataloader(self, dataloader: DataLoader) -> torch.Tensor:
        batches: List[torch.Tensor] = []
        remaining = int(self.config["probe_size"])
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.detach().float()
            take = min(remaining, int(inputs.shape[0]))
            batches.append(inputs[:take])
            remaining -= take
            if remaining <= 0:
                break
        if len(batches) == 0:
            raise ValueError("Failed to collect probe samples from clean_dataloader.")
        return torch.cat(batches, dim=0).to(self.device)

    def _load_cifar10_probe_set(self) -> torch.Tensor:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor

        dataset = CIFAR10(
            root=self.config["dataset_path"],
            train=False,
            transform=ToTensor(),
            download=True,
        )
        probe_size = min(int(self.config["probe_size"]), len(dataset))
        probe_rng = torch.Generator(device="cpu")
        probe_rng.manual_seed(int(self.config["probe_seed"]))
        indices = torch.randperm(len(dataset), generator=probe_rng)[:probe_size].tolist()
        images = [dataset[idx][0] for idx in indices]
        return torch.stack(images, dim=0).float().to(self.device)

    def _build_trigger_batches(self, x_syn: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        raw = self._to_raw_inputs(x_syn)
        if raw.dim() != 4:
            raise ValueError("x_syn must have shape [B, C, H, W].")

        patch = raw.clone()
        patch_size = min(int(self.config["patch_size"]), raw.shape[-1], raw.shape[-2])
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

        trigger_batches_all = OrderedDict(
            patch=self._normalize_inputs(patch),
            blend=self._normalize_inputs(blend),
            sig=self._normalize_inputs(sig),
            color=self._normalize_inputs(color),
        )
        requested = self.config.get("trigger_types", ["patch", "blend", "sig", "color"])
        if isinstance(requested, str):
            requested_tokens = [token.strip().lower() for token in requested.split(",") if token.strip()]
        else:
            requested_tokens = [str(token).strip().lower() for token in requested if str(token).strip()]
        selected: List[str] = []
        for name in requested_tokens:
            if name in trigger_batches_all and name not in selected:
                selected.append(name)
        if len(selected) == 0:
            selected = list(trigger_batches_all.keys())
        trigger_batches = OrderedDict((name, trigger_batches_all[name]) for name in selected)
        return trigger_batches

    def _to_raw_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        tensor = inputs.detach().clone().float().to(self.device)
        if (not bool(self.config.get("auto_denormalize_probe", True))) or tensor.numel() == 0:
            return torch.clamp(tensor, 0.0, 1.0)
        min_value = float(tensor.min().item())
        max_value = float(tensor.max().item())
        looks_normalized = (min_value < -0.2) or (max_value > 1.2)
        if looks_normalized:
            mean = torch.tensor(self.config["normalization_mean"], dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
            std = torch.tensor(self.config["normalization_std"], dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
            tensor = tensor * std + mean
        return torch.clamp(tensor, 0.0, 1.0)

    def _build_sig_pattern(self, raw: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = raw.shape
        xs = torch.linspace(0.0, 1.0, steps=width, device=raw.device, dtype=raw.dtype)
        sine = torch.sin(2.0 * torch.pi * float(self.config["sig_frequency"]) * xs + float(self.config["sig_phase"]))
        pattern = sine.view(1, 1, 1, width).repeat(1, channels, height, 1)
        return float(self.config["sig_delta"]) * pattern

    def _normalize_inputs(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config["normalization_mean"], dtype=raw_inputs.dtype, device=raw_inputs.device)
        std = torch.tensor(self.config["normalization_std"], dtype=raw_inputs.dtype, device=raw_inputs.device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return (raw_inputs - mean) / std

    def _apply_group_update(
        self,
        base_state: OrderedDict[str, torch.Tensor],
        group_update: ModelUpdate,
    ) -> OrderedDict[str, torch.Tensor]:
        updated = OrderedDict()
        if isinstance(group_update, Mapping):
            for key, tensor in base_state.items():
                if key in group_update:
                    delta = group_update[key].detach().to(self.device)
                    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                    updated[key] = tensor + delta.to(dtype=tensor.dtype)
                else:
                    # Reuse immutable base tensors to avoid cloning the full model state per group.
                    updated[key] = tensor
            return updated

        if isinstance(group_update, (list, tuple)):
            flat_update = torch.cat(
                [torch.as_tensor(tensor, device=self.device).reshape(-1) for tensor in group_update],
                dim=0,
            )
        else:
            flat_update = torch.as_tensor(group_update, device=self.device).flatten()
        expected_numel = int(sum(self._param_numels))
        if flat_update.numel() != expected_numel:
            raise ValueError(
                f"Flat group update has {flat_update.numel()} elements, expected {expected_numel}."
            )

        offset = 0
        param_updates: Dict[str, torch.Tensor] = {}
        for name, numel, shape in zip(self._param_names, self._param_numels, self._param_shapes):
            param_updates[name] = torch.nan_to_num(
                flat_update[offset:offset + numel].view(shape),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            offset += numel

        for key, tensor in base_state.items():
            if key in param_updates:
                updated[key] = tensor + param_updates[key].to(dtype=tensor.dtype)
            else:
                # Reuse immutable base tensors to avoid cloning the full model state per group.
                updated[key] = tensor
        return updated

    def _forward_logits(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        batches: List[torch.Tensor] = []
        batch_size = max(1, int(self.config["batch_size"]))
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            while True:
                try:
                    batches.clear()
                    for start in range(0, inputs.shape[0], batch_size):
                        batch = inputs[start:start + batch_size]
                        outputs = functional_call(self.model, state_dict, (batch,))
                        logits = self._extract_logits(outputs)
                        logits = self._ensure_logits_space(logits)
                        batches.append(torch.nan_to_num(logits.detach(), nan=0.0, posinf=0.0, neginf=0.0))
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or batch_size <= 1:
                        raise
                    if model_device.type == "cuda":
                        torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
        return torch.nan_to_num(torch.cat(batches, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError("Model forward must return a tensor or a tuple/list containing logits.")

    def _ensure_logits_space(self, tensor: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if not bool(self.config.get("auto_logit_from_probability_output", True)):
            return logits
        if logits.dim() < 2:
            return logits
        # If model output already looks like probabilities, map to logit-like space
        # to avoid using probability values directly in margin computation.
        row_sum = logits.sum(dim=-1)
        min_value = float(logits.min().item())
        max_value = float(logits.max().item())
        looks_prob = (
            min_value >= -1e-6
            and max_value <= 1.0 + 1e-6
            and bool(torch.all(torch.abs(row_sum - 1.0) < 1e-3).item())
        )
        if looks_prob:
            eps = 1e-8
            probs = torch.clamp(logits, min=eps, max=1.0)
            logits = torch.log(probs)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

    def _margin_from_logits(self, logits: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError("Expected logits with shape [Q, B, C].")
        target_classes = target_classes.long().to(logits.device)
        selected = logits.index_select(dim=-1, index=target_classes)
        total = logits.sum(dim=-1, keepdim=True)
        others_mean = (total - selected) / max(logits.shape[-1] - 1, 1)
        margin = torch.nan_to_num(selected - others_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return margin.permute(2, 0, 1).contiguous()

    def _compute_coarse_class_scores(
        self,
        base_logits: torch.Tensor,
        updated_logits: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        base_logits = torch.nan_to_num(base_logits, nan=0.0, posinf=0.0, neginf=0.0)
        updated_logits = torch.nan_to_num(updated_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == "abs_delta":
            score = (updated_logits - base_logits).mean(dim=0)
            return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == "softmax_delta":
            score = (torch.softmax(updated_logits, dim=-1) - torch.softmax(base_logits, dim=-1)).mean(dim=0)
            return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == "ratio":
            eps = float(self.config.get("coarse_ratio_eps", 1e-8))
            denom = torch.where(
                base_logits >= 0.0,
                torch.clamp(base_logits, min=eps),
                torch.clamp(base_logits, max=-eps),
            )
            score = (updated_logits / denom).mean(dim=0)
            return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # Default/recommended: margin increment.
        c = int(base_logits.shape[-1])
        denom = max(c - 1, 1)
        base_margin = base_logits - (base_logits.sum(dim=-1, keepdim=True) - base_logits) / float(denom)
        updated_margin = updated_logits - (updated_logits.sum(dim=-1, keepdim=True) - updated_logits) / float(denom)
        # Coarse class signal also follows E[phi(m_after)-phi(m_before)].
        score = (self._phi_transform(updated_margin) - self._phi_transform(base_margin)).mean(dim=0)
        coarse_clip = float(self.config.get("coarse_score_clip", 2.0))
        if coarse_clip > 0.0:
            score = torch.clamp(score, min=-coarse_clip, max=coarse_clip)
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def _phi_transform(self, margins: torch.Tensor) -> torch.Tensor:
        phi = str(self.config.get("phi", "tanh")).strip().lower()
        if phi == "identity":
            return torch.nan_to_num(margins, nan=0.0, posinf=0.0, neginf=0.0)
        temperature = max(float(self.config.get("phi_temperature", 1.0)), 1e-6)
        transformed = torch.tanh(margins / temperature)
        return torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    def _preprocess_scores_for_mad(
        self,
        scores: torch.Tensor,
        mode: str,
        iqr_multiplier: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        clean = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if clean.numel() == 0:
            return clean, {"q1": float("nan"), "q3": float("nan"), "iqr": float("nan")}
        q1 = torch.quantile(clean, q=0.25)
        q3 = torch.quantile(clean, q=0.75)
        iqr = torch.clamp(q3 - q1, min=1e-12)
        lo = q1 - float(iqr_multiplier) * iqr
        hi = q3 + float(iqr_multiplier) * iqr

        if mode == "none":
            processed = clean
        elif mode == "tanh_iqr":
            scale = torch.clamp(iqr, min=1e-6)
            processed = torch.tanh(clean / scale)
        else:
            processed = torch.clamp(clean, min=lo, max=hi)
        stats = {
            "q1": float(q1.item()),
            "q3": float(q3.item()),
            "iqr": float(iqr.item()),
            "clip_lo": float(lo.item()),
            "clip_hi": float(hi.item()),
        }
        return processed, stats

    def _compute_bimodality_coefficient(self, scores: torch.Tensor) -> float:
        x = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0).double().flatten()
        n = int(x.numel())
        if n < 4:
            return float("nan")
        mean = torch.mean(x)
        std = torch.std(x, unbiased=False)
        if float(std.item()) < 1e-12:
            return 0.0
        z = (x - mean) / std
        skew = torch.mean(z ** 3)
        kurt = torch.mean(z ** 4)
        denom_corr = 3.0 * float((n - 1) ** 2) / max(float((n - 2) * (n - 3)), 1.0)
        bc = (float(skew.item()) ** 2 + 1.0) / max(float(kurt.item()) + denom_corr, 1e-12)
        return float(bc)


def run_cts_intent(
    model: Module,
    group_updates: Union[Mapping[int, ModelUpdate], Sequence[ModelUpdate], torch.Tensor],
    w_t: Optional[Union[Module, Mapping[str, torch.Tensor]]] = None,
    X_syn: Optional[Union[torch.Tensor, DataLoader]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[bool], int]:
    """
    Run CTS-Intent with the requested raw interface.

    Args:
        model: BackdoorBench-compatible classification model.
        group_updates: Group aggregated updates ``Delta_G`` only.
        w_t: Current global model state or model snapshot.
        X_syn: Optional synthetic probe set. If omitted, 100 random CIFAR-10 test
            images are used instead.
        config: Hyper-parameter dictionary.

    Returns:
        ``(cts_scores, b_g, y_star)`` in the same order as ``group_updates``.
    """

    detector = CTSIntent(config=config or {}, model=model)
    return detector.run(group_updates=group_updates, w_t=w_t, x_syn=X_syn)


