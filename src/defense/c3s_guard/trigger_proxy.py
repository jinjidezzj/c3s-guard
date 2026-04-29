"""Trigger proxy definitions used by CTS-Intent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


@dataclass
class TriggerProxySpec:
    """Declarative description of a trigger proxy pattern."""

    name: str
    pattern: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    target_label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TriggerProxyBank:
    """Manage reusable trigger proxies for intent probing."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the proxy bank.

        Args:
            config: Dictionary of trigger-proxy hyper-parameters.
            device: Device used to host proxy tensors.
        """

        pass

    def register_proxy(self, proxy_spec: TriggerProxySpec) -> None:
        """Register a new proxy specification."""

        pass

    def build_default_proxies(
        self,
        input_shape: Sequence[int],
        num_classes: int,
    ) -> None:
        """Populate the bank with default proxies derived from the config."""

        pass

    def list_proxies(self) -> List[str]:
        """Return the names of all registered proxies."""

        pass

    def get_proxy(self, name: str) -> TriggerProxySpec:
        """Fetch a proxy specification by name."""

        pass

    def sample_proxy_batch(
        self,
        batch_size: int,
        strategy: str = "cyclic",
        proxy_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Sample proxy metadata for a probing batch."""

        pass

    def apply_proxy(
        self,
        inputs: torch.Tensor,
        proxy_name: str,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Overlay a named proxy pattern onto an input batch."""

        pass

    def generate_triggered_batch(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        proxy_name: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Generate a triggered input batch for CTS-Intent probing."""

        pass

    def state_dict(self) -> Dict[str, Any]:
        """Serialize the trigger-proxy bank state."""

        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the trigger-proxy bank state."""

        pass
