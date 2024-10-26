"""Formulation-agnostic Lagrangian Dual Framework Loss Function."""

from typing import Optional, Union

import torch
from torch import nn, Tensor

from ml4opf.formulations import OPFViolation
from ml4opf import debug


class LDFLoss(nn.Module):
    """
    LDFLoss implements the Lagrangian Dual Framework.

    `exclude_keys` is either None to use all violations, "all" to skip all violations, or a list of keys to skip specific violations.
    """

    def __init__(
        self,
        v: OPFViolation,
        step_size: float,
        kickin: int,
        update_freq: int,
        divide_by_counter: bool = True,
        exclude_keys: Optional[Union[str, list[str]]] = None,
    ):
        """Initialize LDFLoss module."""
        super().__init__()

        self.v = v

        self.register_buffer("step", torch.tensor(step_size))
        self.register_buffer("kickin", torch.tensor(kickin))
        self.register_buffer("update_freq", torch.tensor(update_freq))

        self.register_buffer("divide_by_counter", torch.tensor(divide_by_counter))
        self.exclude_keys = exclude_keys

        self.register_buffer("best_viol", torch.tensor(0.0))
        self.register_buffer("sample_counter", torch.tensor(0))

        self.register_buffer(
            "short_circuit", torch.tensor(True)
        )  # short_circuit meaning do not compute violations if all λ are zero (before first call to .update())
        self.register_buffer("update_epoch", torch.tensor(False))

        self.init_mults()

    def start_epoch(self, epoch) -> Optional[str]:
        """Call this method at the start of each epoch."""
        if epoch < self.kickin:
            update = False
        else:
            update = bool(epoch % self.update_freq == 0)

        ## short circuit / pre-training special cases
        if self.short_circuit and not update:
            # pretraining where λ are zero so we don't need to compute violations
            return "short_circuit"

        if self.short_circuit and update:
            # disable short circuit (compute violations over this epoch) if we are about to update
            debug("LDFLoss: First update triggered!")
            self.short_circuit = self.short_circuit.logical_not()  # expects to start at True, flip to False
            self.update_epoch = self.update_epoch.logical_not()  # expects to start at False, flip to True
            return "first_update_this_epoch"

        ## base behavior
        if update:  # update λ next time end_epoch() is called
            self.update_epoch = self.update_epoch.logical_not()  # flip to True
            self.reset_trackers()  # make sure trackers are reset so we get exactly one pass through the dataset
            return "update_this_epoch"

    def end_epoch(self):
        """Call this method at the end of each epoch."""
        if self.update_epoch:  # update λ now (we have collected the violations)
            self.update_epoch = self.update_epoch.logical_not()  # flip back to False
            self.update()
            self.reset_trackers()  # reset to zeros

    def init_mults(self, shapes=None):
        """Initialize λ and trackers to zeros."""
        if shapes is None:
            shapes = self.v.violation_shapes
        self.violation_trackers = []
        self.lambdas = []
        for k, v in shapes.items():
            if k in self.exclude_keys:
                continue
            self.register_buffer(f"lambda_{k}", torch.zeros(v, dtype=torch.float))
            self.register_buffer(f"violation_{k}", torch.zeros(v, dtype=torch.float))
            self.violation_trackers.append(f"violation_{k}")
            self.lambdas.append(f"lambda_{k}")

    def reset_trackers(self):
        """Reset the violation trackers to zeros."""
        for k in self.violation_trackers:
            self._buffers[k].fill_(0)

        self.sample_counter.fill_(0)

    @torch.no_grad()
    def update(self):
        """Update the lagrangian dual multipliers (λ)"""
        debug("LDFLoss: Updating lagrangian dual multipliers (λ)")

        if self.divide_by_counter:
            debug("LDFLoss: Dividing by iteration counter")
            for k in self.violation_trackers:
                self._buffers[k] /= self.sample_counter

        for kl, kv in zip(self.lambdas, self.violation_trackers):
            self._buffers[kl] += self.step * self._buffers[kv]

    def forward(
        self,
        base_loss: Tensor,
        exclude_keys: Optional[Union[str, list[str]]] = None,
        **calc_violation_inputs: Tensor,
    ) -> Tensor:
        """Compute the LDF Loss for a batch of samples."""
        if exclude_keys is None:
            exclude_keys = self.exclude_keys

        if self.short_circuit or exclude_keys == "all":
            return base_loss.mean()

        calc_violation_inputs.setdefault("reduction", "none")
        violations = self.v.calc_violations(**calc_violation_inputs)

        loss = (
            base_loss.mean(dim=1) if len(base_loss.shape) == 2 else base_loss
        )  # if base loss is already 1 entry per sample (e.g. OBJLoss), don't .mean(dim=1) it.

        if exclude_keys is None:
            exclude_keys = self.exclude_keys

        for k, v in violations.items():
            if exclude_keys is None or k not in exclude_keys:
                loss += (v * self._buffers[f"lambda_{k}"]).mean(dim=1)

        if self.update_epoch:
            with torch.no_grad():
                for k, v in violations.items():
                    if exclude_keys is None or k in exclude_keys:
                        continue
                    self._buffers[f"violation_{k}"] += v.sum(dim=0)

                if self.divide_by_counter:
                    batch_size = next(iter(violations.values())).shape[0]
                    self.sample_counter += batch_size

        return loss.mean()
