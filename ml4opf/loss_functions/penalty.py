"""Penalize constraint violations"""

from typing import Optional, Union

import torch
from torch import nn, Tensor

from ml4opf.formulations import OPFViolation


class PenaltyLoss(nn.Module):
    """
    PenaltyLoss penalizes constraint violations in the loss.

    `exclude_keys` is either None to use all violations, "all" to skip all violations, or a list of keys to skip specific violations.
    """

    def __init__(
        self,
        v: OPFViolation,
        exclude_keys: Optional[Union[str, list[str]]] = None,
        multipliers: Optional[Union[Tensor, dict[str, Tensor]]] = None,
    ):
        """Initialize PenaltyLoss module."""
        super().__init__()

        self.v = v
        self.exclude_keys = exclude_keys

        self.init_mults(multipliers)

    def init_mults(self, multipliers: Optional[Union[Tensor, dict[str, Tensor]]] = None):
        """Initialize multipliers for each constraint type."""
        shapes = self.v.violation_shapes

        if multipliers is None:
            multipliers = {k: torch.as_tensor(1.0) for k in shapes.keys()}
        elif isinstance(multipliers, (Tensor, int, float)):
            multipliers = {k: torch.as_tensor(multipliers) for k in shapes.keys()}
        elif isinstance(multipliers, dict):
            multipliers = {k: torch.as_tensor(v) for k, v in multipliers.items()}
        else:
            raise ValueError(f"Invalid {multipliers=}")

        for k in shapes.keys():
            self.register_buffer(f"mult_{k}", multipliers[k])

    def forward(
        self,
        base_loss: Tensor,
        exclude_keys: Optional[Union[str, list[str]]] = None,
        **calc_violation_inputs: Tensor,
    ) -> Tensor:
        """Compute the PenaltyLoss for a batch of samples."""
        if exclude_keys is None:
            exclude_keys = self.exclude_keys

        if exclude_keys == "all":
            return base_loss.mean()

        calc_violation_inputs.setdefault("reduction", "none")
        violations = self.v.calc_violations(**calc_violation_inputs)

        loss = (
            base_loss.mean(dim=1) if len(base_loss.shape) == 2 else base_loss
        )  # if base loss is already 1 entry per sample (e.g. OBJLoss), don't .mean(dim=1) it.

        for k, v in violations.items():
            if exclude_keys is None or k not in exclude_keys:
                loss += (v * self._buffers[f"mult_{k}"]).mean(dim=1)

        return loss.mean()
