"""Demo script for running C3S-Guard on CIFAR-10 + ResNet18 + BadNets.

This is a compact end-to-end example showing how to integrate the C3S-Guard
controller into a federated training loop. For tractability, the audit stage
reuses the latest cached update for each client to synthesize secure group
aggregates instead of retraining every audited group from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from defense.c3s_guard.c3s_guard import C3SGuard
from utils.aggregate_block.model_trainer_generate import generate_cls_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the C3S-Guard demo on CIFAR-10.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default="preactresnet18")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-rounds", type=int, default=21)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--local-lr", type=float, default=0.02)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1)
    parser.add_argument("--malicious-fraction", type=float, default=0.1)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--poison-rate", type=float, default=0.3)
    parser.add_argument("--max-client-samples", type=int, default=128)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--audit-period", type=int, default=20)
    parser.add_argument("--audit-rounds", type=int, default=20)
    parser.add_argument("--audit-group-size", type=int, default=10)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--suspicious-topk", type=int, default=10)
    parser.add_argument("--probe-size", type=int, default=100)
    parser.add_argument("--cts-mode", type=str, default="raw", choices=["raw", "mdbf", "dcbd"])
    parser.add_argument("--cts-mdbf-alpha", type=float, default=0.7)
    parser.add_argument("--cts-mdbf-percentile", type=float, default=75.0)
    parser.add_argument("--cts-dcbd-tsc-weight", type=float, default=0.5)
    parser.add_argument("--cts-dcbd-mdbf-weight", type=float, default=0.35)
    parser.add_argument("--cts-dcbd-cse-weight", type=float, default=0.15)
    parser.add_argument("--cts-dcbd-use-raw-logit", action="store_true")
    parser.add_argument("--cts-dcbd-disable-raw-logit", dest="cts_dcbd_use_raw_logit", action="store_false")
    parser.set_defaults(cts_dcbd_use_raw_logit=True)
    parser.add_argument("--cts-dcbd-simple", action="store_true")
    parser.add_argument("--cts-dcbd-alpha", type=float, default=0.3)
    parser.add_argument("--fct-enable", action="store_true")
    parser.add_argument("--fct-disable", dest="fct_enable", action="store_false")
    parser.set_defaults(fct_enable=False)
    parser.add_argument("--fct-topn", type=int, default=6)
    parser.add_argument("--fct-num-pairs", type=int, default=6)
    parser.add_argument("--fct-z-threshold", type=float, default=1.0)
    parser.add_argument("--fct-group-size", type=int, default=0)
    parser.add_argument("--fct-clean-pool-strategy", type=str, default="bottom", choices=["bottom", "exclude_topn"])
    parser.add_argument("--fct-clean-pool-bottom-ratio", type=float, default=0.60)
    parser.add_argument("--risk-confirmed-highconf-enable", action="store_true")
    parser.add_argument("--risk-confirmed-highconf-disable", dest="risk_confirmed_highconf_enable", action="store_false")
    parser.set_defaults(risk_confirmed_highconf_enable=True)
    parser.add_argument("--risk-confirmed-highconf-min-fct-z", type=float, default=1.5)
    parser.add_argument("--risk-confirmed-highconf-min-pairs", type=int, default=4)
    parser.add_argument(
        "--risk-confirmed-highconf-require-localization-reliable",
        action="store_true",
    )
    parser.add_argument(
        "--risk-confirmed-highconf-ignore-localization-reliable",
        dest="risk_confirmed_highconf_require_localization_reliable",
        action="store_false",
    )
    parser.set_defaults(risk_confirmed_highconf_require_localization_reliable=True)
    parser.add_argument("--fct-use-matched-controls", action="store_true")
    parser.add_argument("--fct-disable-matched-controls", dest="fct_use_matched_controls", action="store_false")
    parser.set_defaults(fct_use_matched_controls=True)
    parser.add_argument("--repair-require-confirmed", action="store_true")
    parser.add_argument("--repair-allow-without-confirmed", dest="repair_require_confirmed", action="store_false")
    parser.set_defaults(repair_require_confirmed=False)
    parser.add_argument("--repair-require-confirmed-alert-override", action="store_true")
    parser.add_argument(
        "--repair-disable-require-confirmed-alert-override",
        dest="repair_require_confirmed_alert_override",
        action="store_false",
    )
    parser.set_defaults(repair_require_confirmed_alert_override=True)
    parser.add_argument("--repair-require-confirmed-alert-min-p-hat", type=float, default=0.05)
    parser.add_argument(
        "--repair-clean-accept-require-localization-reliable",
        action="store_true",
    )
    parser.add_argument(
        "--repair-clean-accept-ignore-localization-reliable",
        dest="repair_clean_accept_require_localization_reliable",
        action="store_false",
    )
    parser.set_defaults(repair_clean_accept_require_localization_reliable=True)
    parser.add_argument(
        "--repair-clean-accept-allow-unreliable-with-highconf-confirmed",
        action="store_true",
    )
    parser.add_argument(
        "--repair-clean-accept-disable-unreliable-with-highconf-confirmed",
        dest="repair_clean_accept_allow_unreliable_with_highconf_confirmed",
        action="store_false",
    )
    parser.set_defaults(repair_clean_accept_allow_unreliable_with_highconf_confirmed=True)
    parser.add_argument("--repair-clean-accept-min-highconf-confirmed", type=int, default=2)
    parser.add_argument("--repair-clean-accept-highconf-min-fct-z", type=float, default=1.8)
    parser.add_argument("--repair-clean-accept-highconf-min-pairs", type=int, default=4)
    parser.add_argument("--repair-candidate-require-confirmation", action="store_true")
    parser.add_argument("--repair-candidate-allow-without-confirmation", dest="repair_candidate_require_confirmation", action="store_false")
    parser.set_defaults(repair_candidate_require_confirmation=True)
    parser.add_argument("--repair-candidate-min-safe-acc", type=float, default=0.45)
    parser.add_argument("--repair-candidate-shadow-only", action="store_true")
    parser.add_argument("--repair-strict-alert-fallback-gated-scale", type=float, default=0.35)
    parser.add_argument("--c3s-confirmed-gate-enable", action="store_true")
    parser.add_argument("--c3s-confirmed-gate-disable", dest="c3s_confirmed_gate_enable", action="store_false")
    parser.set_defaults(c3s_confirmed_gate_enable=False)
    parser.add_argument("--c3s-confirmed-gate-beta", type=float, default=8.0)
    parser.add_argument("--c3s-confirmed-gate-threshold", type=float, default=0.20)
    parser.add_argument("--risk-aware-enable", dest="risk_aware_sampling", action="store_true")
    parser.add_argument("--risk-aware-disable", dest="risk_aware_sampling", action="store_false")
    parser.set_defaults(risk_aware_sampling=False)
    parser.add_argument("--risk-aware-gamma", type=float, default=1.0)
    parser.add_argument("--risk-aware-p-min", type=float, default=0.5)
    parser.add_argument("--risk-aware-source", type=str, default="confirmed", choices=["raw", "confirmed"])
    parser.add_argument("--risk-aware-require-fct", action="store_true")
    parser.add_argument("--risk-aware-ignore-fct", dest="risk_aware_require_fct", action="store_false")
    parser.set_defaults(risk_aware_require_fct=False)
    parser.add_argument("--risk-allow-negative-signal", action="store_true")
    parser.add_argument("--risk-confirmed-min-count", type=int, default=2)
    parser.add_argument("--risk-confirmed-min-suspicious-score", type=float, default=0.05)
    parser.add_argument("--risk-confirmed-fallback-source", type=str, default="none", choices=["none", "raw"])
    parser.add_argument(
        "--dgc-repair-safe-update-ratio-source",
        type=str,
        default="confirmed",
        choices=["raw", "confirmed", "auto"],
    )
    parser.add_argument(
        "--dgc-repair-safe-signal-source",
        type=str,
        default="dcbd",
        choices=["raw", "dcbd", "fct"],
    )
    parser.add_argument("--dgc-repair-safe-refresh-enable", action="store_true")
    parser.add_argument("--dgc-repair-safe-refresh-disable", dest="dgc_repair_safe_refresh_enable", action="store_false")
    parser.set_defaults(dgc_repair_safe_refresh_enable=False)
    parser.add_argument("--dgc-repair-safe-refresh-stale-rounds", type=int, default=20)
    parser.add_argument("--dgc-repair-safe-refresh-margin", type=float, default=0.10)
    parser.add_argument("--dgc-repair-safe-refresh-ema", type=float, default=0.5)
    parser.add_argument("--dgc-repair-safe-refresh-max-confirmed", type=int, default=3)
    parser.add_argument("--eval-subset", type=int, default=512)
    parser.add_argument("--distill-epochs", type=int, default=1)
    parser.add_argument("--num-gradient-samples", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Partition dataset indices into non-IID client splits using Dirichlet(alpha)."""

    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_positions = np.where(labels == class_id)[0]
        rng.shuffle(class_positions)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cut_points = (np.cumsum(proportions) * len(class_positions)).astype(int)[:-1]
        splits = np.split(class_positions, cut_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])
    return client_indices


def sample_client_subset(indices: Sequence[int], max_samples: int) -> List[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)
    return list(indices[:max_samples])


def make_patch_trigger(inputs: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
    patched = inputs.clone()
    patch_size = min(patch_size, patched.shape[-1], patched.shape[-2])
    patched[:, :, -patch_size:, -patch_size:] = 1.0
    return patched


def poison_batch(inputs: torch.Tensor, labels: torch.Tensor, target_label: int, poison_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if poison_rate <= 0.0:
        return inputs, labels
    batch_size = inputs.shape[0]
    poison_count = max(1, int(batch_size * poison_rate))
    poisoned_inputs = inputs.clone()
    poisoned_labels = labels.clone()
    poisoned_inputs[:poison_count] = make_patch_trigger(poisoned_inputs[:poison_count])
    poisoned_labels[:poison_count] = int(target_label)
    return poisoned_inputs, poisoned_labels


def local_train_client(
    global_model: nn.Module,
    dataset: Subset,
    device: torch.device,
    update_keys: Sequence[str],
    local_epochs: int,
    batch_size: int,
    lr: float,
    malicious: bool,
    target_label: int,
    poison_rate: float,
) -> Tuple[OrderedDict[str, torch.Tensor], int]:
    """Train one client locally and return the model update ``delta_i``."""

    model = deepcopy(global_model).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    for _ in range(local_epochs):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if malicious:
                inputs, labels = poison_batch(inputs, labels, target_label=target_label, poison_rate=poison_rate)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    update = build_state_update(
        local_model=model,
        global_model=global_model,
        update_keys=update_keys,
        device=device,
    )
    return update, len(dataset)


def get_trainable_update_keys(model: nn.Module) -> List[str]:
    keys = [name for name, _ in model.named_parameters()]
    keys.extend(name for name, buffer in model.named_buffers() if torch.is_floating_point(buffer))
    return keys


def build_state_update(
    local_model: nn.Module,
    global_model: nn.Module,
    update_keys: Sequence[str],
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    local_state = local_model.state_dict()
    global_state = global_model.state_dict()
    update = OrderedDict()
    for key in update_keys:
        local_tensor = local_state[key].detach().to(device)
        global_tensor = global_state[key].detach().to(device)
        update[key] = (local_tensor - global_tensor).cpu()
    return update


def average_updates(
    updates: Sequence[Union[Mapping[str, torch.Tensor], Sequence[torch.Tensor]]],
    weights: Sequence[float],
) -> Union[Mapping[str, torch.Tensor], List[torch.Tensor]]:
    if len(updates) == 0:
        raise ValueError("No updates provided for aggregation.")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = normalized_weights / max(normalized_weights.sum(), 1e-12)
    first_update = updates[0]
    if isinstance(first_update, Mapping):
        aggregated = OrderedDict()
        for key in first_update.keys():
            accumulator = torch.zeros_like(first_update[key], dtype=first_update[key].dtype)
            for update, weight in zip(updates, normalized_weights):
                accumulator.add_(update[key], alpha=float(weight))
            aggregated[key] = accumulator
        return aggregated

    aggregated_list: List[torch.Tensor] = []
    for layer_tensors in zip(*updates):
        accumulator = torch.zeros_like(layer_tensors[0])
        for tensor, weight in zip(layer_tensors, normalized_weights):
            accumulator.add_(tensor, alpha=float(weight))
        aggregated_list.append(accumulator)
    return aggregated_list


def apply_update(
    model: nn.Module,
    update: Union[Mapping[str, torch.Tensor], Sequence[torch.Tensor]],
    device: torch.device,
) -> None:
    with torch.no_grad():
        if isinstance(update, Mapping):
            model_state = model.state_dict()
            for key, delta in update.items():
                if key not in model_state:
                    continue
                target = model_state[key]
                model_state[key] = target + delta.to(device=device, dtype=target.dtype)
            model.load_state_dict(model_state, strict=False)
            return

        for parameter, delta in zip(model.parameters(), update):
            parameter.add_(delta.to(device=device, dtype=parameter.dtype))


def evaluate_clean_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


def evaluate_asr(model: nn.Module, loader: DataLoader, device: torch.device, target_label: int) -> float:
    model.eval()
    total = 0
    success = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = make_patch_trigger(inputs).to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            success += int((predictions == int(target_label)).sum().item())
            total += int(predictions.numel())
    return 100.0 * success / max(total, 1)


def simulate_audit_group(
    guard: C3SGuard,
    group_plan: Dict[str, List[int]],
    update_cache: Dict[int, OrderedDict[str, torch.Tensor]],
    dropout_prob: float,
) -> Tuple[List[int], Optional[Union[Mapping[str, torch.Tensor], List[torch.Tensor]]]]:
    """Create one audit-group aggregate from cached per-client updates.

    Primary members may drop out. Backup members are inserted in order until the
    group reaches the configured group size again or no backup is left.
    """

    actual_clients = []
    for client_id in group_plan["primary"]:
        if random.random() >= dropout_prob and client_id in update_cache:
            actual_clients.append(client_id)

    for backup_id in group_plan["backup"]:
        if len(actual_clients) >= len(group_plan["primary"]):
            break
        if backup_id in actual_clients:
            continue
        if backup_id in update_cache:
            actual_clients.append(backup_id)

    if len(actual_clients) < guard.k_min:
        return actual_clients, None

    group_updates = [update_cache[client_id] for client_id in actual_clients]
    group_weights = [1.0 for _ in actual_clients]
    aggregated = average_updates(group_updates, group_weights)
    return actual_clients, aggregated


def build_eval_loader(dataset: CIFAR10, subset_size: int, batch_size: int) -> DataLoader:
    indices = list(range(min(subset_size, len(dataset))))
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_dataset = CIFAR10(root=args.data_root, train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root=args.data_root, train=False, transform=ToTensor(), download=True)
    client_indices = dirichlet_partition(
        labels=train_dataset.targets,
        num_clients=args.num_clients,
        alpha=args.dirichlet_alpha,
        seed=args.seed,
    )

    model = generate_cls_model(args.model, num_classes=10).to(device)
    update_keys = get_trainable_update_keys(model)
    malicious_count = max(1, int(args.num_clients * args.malicious_fraction))
    malicious_clients = set(range(malicious_count))

    guard_config = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "audit_period": args.audit_period,
        "audit_rounds": args.audit_rounds,
        "audit_group_size": args.audit_group_size,
        "k_min": args.k_min,
        "suspicious_topk": args.suspicious_topk,
        "audit_from_selected_only": True,
        "fct_enable": bool(args.fct_enable),
        "fct_topn": int(max(args.fct_topn, 1)),
        "fct_num_pairs": int(max(args.fct_num_pairs, 1)),
        "fct_z_threshold": float(args.fct_z_threshold),
        "fct_use_matched_controls": bool(args.fct_use_matched_controls),
        "fct_group_size": int(max(args.fct_group_size, 0)),
        "fct_clean_pool_strategy": str(args.fct_clean_pool_strategy),
        "fct_clean_pool_bottom_ratio": float(max(0.05, min(1.0, args.fct_clean_pool_bottom_ratio))),
        "risk_confirmed_highconf_enable": bool(args.risk_confirmed_highconf_enable),
        "risk_confirmed_highconf_min_fct_z": float(args.risk_confirmed_highconf_min_fct_z),
        "risk_confirmed_highconf_min_pairs": int(max(args.risk_confirmed_highconf_min_pairs, 0)),
        "risk_confirmed_highconf_require_localization_reliable": bool(
            args.risk_confirmed_highconf_require_localization_reliable
        ),
        "risk_aware_enable": bool(args.risk_aware_sampling),
        "risk_aware_source": str(args.risk_aware_source),
        "risk_aware_gamma": float(max(args.risk_aware_gamma, 0.0)),
        "risk_aware_p_min": float(args.risk_aware_p_min),
        "risk_aware_require_fct": bool(args.risk_aware_require_fct),
        "repair_require_confirmed": bool(args.repair_require_confirmed),
        "repair_require_confirmed_alert_override": bool(args.repair_require_confirmed_alert_override),
        "repair_require_confirmed_alert_min_p_hat": float(
            max(args.repair_require_confirmed_alert_min_p_hat, 0.0)
        ),
        "repair_candidate_require_confirmation": bool(args.repair_candidate_require_confirmation),
        "repair_candidate_min_safe_acc": float(max(args.repair_candidate_min_safe_acc, 0.0)),
        "repair_candidate_shadow_only": bool(args.repair_candidate_shadow_only),
        "dgc_repair_safe_signal_source": str(args.dgc_repair_safe_signal_source),
        "repair_clean_accept_require_localization_reliable": bool(
            args.repair_clean_accept_require_localization_reliable
        ),
        "repair_clean_accept_allow_unreliable_with_highconf_confirmed": bool(
            args.repair_clean_accept_allow_unreliable_with_highconf_confirmed
        ),
        "repair_clean_accept_min_highconf_confirmed": int(
            max(args.repair_clean_accept_min_highconf_confirmed, 0)
        ),
        "repair_clean_accept_highconf_min_fct_z": float(
            args.repair_clean_accept_highconf_min_fct_z
        ),
        "repair_clean_accept_highconf_min_pairs": int(
            max(args.repair_clean_accept_highconf_min_pairs, 0)
        ),
        "c3s_confirmed_gate_enable": bool(args.c3s_confirmed_gate_enable),
        "c3s_confirmed_gate_beta": float(max(args.c3s_confirmed_gate_beta, 0.0)),
        "c3s_confirmed_gate_threshold": float(max(0.0, min(1.0, args.c3s_confirmed_gate_threshold))),
        "cts_intent": {
            "probe_size": args.probe_size,
            "batch_size": 128,
            "dataset_path": args.data_root,
            "normalization_mean": [0.0, 0.0, 0.0],
            "normalization_std": [1.0, 1.0, 1.0],
            "force_known_target": True,
            "known_target_label": int(args.target_label),
            "cts_mode": str(args.cts_mode),
            "cts_mdbf_alpha": float(max(0.0, min(1.0, args.cts_mdbf_alpha))),
            "cts_mdbf_percentile": float(max(0.0, min(100.0, args.cts_mdbf_percentile))),
            "cts_dcbd_tsc_weight": float(args.cts_dcbd_tsc_weight),
            "cts_dcbd_mdbf_weight": float(args.cts_dcbd_mdbf_weight),
            "cts_dcbd_cse_weight": float(args.cts_dcbd_cse_weight),
            "cts_dcbd_use_raw_logit": bool(args.cts_dcbd_use_raw_logit),
            "cts_dcbd_simple": bool(args.cts_dcbd_simple),
            "cts_dcbd_alpha": float(args.cts_dcbd_alpha),
        },
        "s3_loc": {
            "method": "counting",
            "suspicious_topk": args.suspicious_topk,
            "degenerate_threshold": 0.8,
        },
        "dgc_clean": {
            "num_gradient_samples": args.num_gradient_samples,
            "gradient_probe_batch_size": 10,
            "distill_epochs": args.distill_epochs,
            "batch_size": 32,
            "distill_lr": 1e-4,
            "normalization_mean": [0.0, 0.0, 0.0],
            "normalization_std": [1.0, 1.0, 1.0],
        },
        "dgc_repair": {
            "risk_aware_sampling": bool(args.risk_aware_sampling),
            "risk_gamma": float(max(args.risk_aware_gamma, 0.0)),
            "risk_p_min": float(args.risk_aware_p_min),
            "risk_aware_source": str(args.risk_aware_source),
            "risk_allow_negative_signal": bool(args.risk_allow_negative_signal),
            "risk_confirmed_min_count": int(max(args.risk_confirmed_min_count, 0)),
            "risk_confirmed_min_suspicious_score": float(max(args.risk_confirmed_min_suspicious_score, 0.0)),
            "risk_confirmed_fallback_source": str(args.risk_confirmed_fallback_source),
            "confirmed_gate_enable": bool(args.c3s_confirmed_gate_enable),
            "confirmed_gate_beta": float(max(args.c3s_confirmed_gate_beta, 0.0)),
            "confirmed_gate_threshold": float(max(0.0, min(1.0, args.c3s_confirmed_gate_threshold))),
            "safe_signal_source": str(args.dgc_repair_safe_signal_source),
            "safe_update_ratio_source": str(args.dgc_repair_safe_update_ratio_source),
            "safe_refresh_enable": bool(args.dgc_repair_safe_refresh_enable),
            "safe_refresh_stale_rounds": int(max(args.dgc_repair_safe_refresh_stale_rounds, 1)),
            "safe_refresh_margin": float(max(args.dgc_repair_safe_refresh_margin, 0.0)),
            "safe_refresh_ema": float(max(0.0, min(1.0, args.dgc_repair_safe_refresh_ema))),
            "safe_refresh_max_confirmed": int(max(args.dgc_repair_safe_refresh_max_confirmed, 0)),
        },
    }
    guard = C3SGuard(config=guard_config, model=model)

    clean_eval_loader = build_eval_loader(test_dataset, args.eval_subset, args.local_batch_size)
    update_cache: Dict[int, OrderedDict[str, torch.Tensor]] = {}

    for round_idx in range(args.num_rounds):
        global_model_before = deepcopy(model).to(device)
        guard.on_round_start(round_idx, global_model_before)

        if bool(args.risk_aware_sampling):
            selected_clients = guard.sample_clients_risk_aware(args.clients_per_round)
        else:
            selected_clients = random.sample(range(args.num_clients), k=args.clients_per_round)
        guard.on_clients_selected(selected_clients)

        client_updates: List[List[torch.Tensor]] = []
        aggregation_weights: List[float] = []
        for client_id in selected_clients:
            sampled_indices = sample_client_subset(client_indices[client_id], args.max_client_samples)
            if len(sampled_indices) == 0:
                continue
            client_dataset = Subset(train_dataset, sampled_indices)
            update, local_sample_count = local_train_client(
                global_model=global_model_before,
                dataset=client_dataset,
                device=device,
                update_keys=update_keys,
                local_epochs=args.local_epochs,
                batch_size=args.local_batch_size,
                lr=args.local_lr,
                malicious=client_id in malicious_clients,
                target_label=args.target_label,
                poison_rate=args.poison_rate,
            )
            update_cache[client_id] = OrderedDict((key, tensor.clone()) for key, tensor in update.items())
            client_updates.append(update)
            aggregation_weights.append(float(local_sample_count) * guard.get_client_weight(client_id))

        if len(client_updates) == 0:
            continue

        global_update = average_updates(client_updates, aggregation_weights)

        if guard.current_audit_active:
            guard.create_audit_groups()
            for group_plan in guard.current_audit_plan:
                actual_clients, group_update = simulate_audit_group(
                    guard=guard,
                    group_plan=group_plan,
                    update_cache=update_cache,
                    dropout_prob=args.dropout_prob,
                )
                if group_update is None:
                    continue
                guard.on_group_aggregated(group_plan["group_id"], group_update, actual_clients)

            raw_model = deepcopy(global_model_before).to(device)
            apply_update(raw_model, global_update, device=device)
            clean_acc_before = evaluate_clean_accuracy(raw_model, clean_eval_loader, device=device)
            asr_before = evaluate_asr(raw_model, clean_eval_loader, device=device, target_label=args.target_label)
        else:
            clean_acc_before = None
            asr_before = None

        def _fct_aggregate_from_cache(member_ids: Sequence[int]) -> Optional[Union[Mapping[str, torch.Tensor], List[torch.Tensor]]]:
            updates = [
                update_cache[int(client_id)]
                for client_id in member_ids
                if int(client_id) in update_cache
            ]
            if len(updates) == 0:
                return None
            return average_updates(updates, [1.0 for _ in updates])

        decision = guard.on_aggregation_complete(
            global_update=global_update,
            global_model=global_model_before,
            metadata={
                "clean_acc_before": clean_acc_before,
                "asr_before": asr_before,
                "target_label": int(args.target_label),
                "force_known_target": True,
                "fct_group_aggregate_fn": _fct_aggregate_from_cache,
                "fct_aggregate_mode": "simulation_direct_sum",
                "fct_clean_pool_clients": [int(client_id) for client_id in selected_clients],
                "malicious_client_ids": sorted(int(client_id) for client_id in malicious_clients),
            },
        )
        update_to_apply = decision.cleaning.cleaned_update if decision.cleaning is not None else global_update
        apply_update(model, update_to_apply, device=device)

        clean_acc = evaluate_clean_accuracy(model, clean_eval_loader, device=device)
        asr = evaluate_asr(model, clean_eval_loader, device=device, target_label=args.target_label)
        print(
            f"round={round_idx:03d} clean_acc={clean_acc:.2f} asr={asr:.2f} "
            f"audit={decision.audit_triggered} suspicious={sorted(guard.suspicious_clients)}"
        )
        if decision.audit_triggered:
            print(json.dumps(guard.audit_history[-1], ensure_ascii=False))


if __name__ == "__main__":
    main()







