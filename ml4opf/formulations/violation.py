""" Abstract base class for OPFViolation classes.

Each formulation should define a class that inherits from `OPFViolation` and implements the following methods:

- `calc_violations`: Calculate the violations of all the constraints. Returns a dictionary of tensors.

- `violations_shapes`: Return the shapes of the violations as a dictionary.

- `compute_objective`: Compute the objective value for a batch of samples.
"""

import torch, torch.nn as nn

from torch import Tensor
from typing import Union
from abc import ABC, abstractmethod

import ml4opf.functional as MOF
from ml4opf.formulations.incidence_mixin import IncidenceMixin


class OPFViolation(IncidenceMixin, nn.Module, ABC):
    """
    The `OPFViolation` class is where all the problem expressions (objective, constraints, etc.) are defined.
    The classes provide a convenient interface for when the only varying quantity in the formulation
    per sample is the load demand `pd`. If other quantities vary, the user should use the functional
    interface at `ml4opf.functional`.

    When `clamp` is true, the values of g(x) are clamped to be non-negative and the values of h(x) are absolute-valued.
    Otherwise the raw values are returned.

    `OPFViolation` is a `torch.nn.Module`; all tensors used in computation are registered as non-persistent buffers.
    To move the module to a different device, use `.to(device)` as you would with any other `nn.Module`.
    Make sure that when you pass data to `OPFViolation`, it is on the same device as `OPFViolation`.
    """

    SUPPORTED_REDUCTIONS = ("none", "mean", "sum", "max")

    def __init__(self, json_data: dict[str, Union[bool, int, float, dict[str, Tensor]]]):
        super().__init__()

        branch_data: dict[str, Tensor] = json_data["branch"]
        shunt_data: dict[str, Tensor] = json_data["shunt"]
        gen_data: dict[str, Tensor] = json_data["gen"]
        bus_data: dict[str, Tensor] = json_data["bus"]
        load_data: dict[str, Tensor] = json_data["load"]

        self.register_buffer("n_gen", torch.tensor(json_data["n_gen"]), persistent=False)
        self.register_buffer("n_bus", torch.tensor(json_data["n_bus"]), persistent=False)
        self.register_buffer("n_branch", torch.tensor(json_data["n_branch"]), persistent=False)
        self.register_buffer("n_load", torch.tensor(json_data["n_load"]), persistent=False)

        self.register_buffer("gen_per_bus", bus_data["gens"], persistent=False)
        self.register_buffer("shunt_per_bus", bus_data["shunts"], persistent=False)
        self.register_buffer("load_per_bus", bus_data["loads"], persistent=False)
        self.register_buffer("branch_from_per_bus", bus_data["br_f"], persistent=False)
        self.register_buffer("branch_to_per_bus", bus_data["br_t"], persistent=False)
        self.register_buffer("fbus_per_branch", branch_data["f_bus"], persistent=False)
        self.register_buffer("tbus_per_branch", branch_data["t_bus"], persistent=False)

        self.register_buffer("gen_bus", gen_data["gen_bus"], persistent=False)
        self.register_buffer("load_bus", load_data["load_bus"], persistent=False)
        self.register_buffer("shunt_bus", shunt_data["shunt_bus"], persistent=False)

    @abstractmethod
    def calc_violations(self, *args, reduction: str = "mean", clamp: bool = True) -> dict[str, Tensor]:
        """Calculate the violations of the constraints. Returns a dictionary of tensors."""
        pass

    @property
    @abstractmethod
    def violation_shapes(self) -> dict[str, int]:
        """Return the shapes of the violations returned by `OPFViolation.calc_violations`."""
        pass

    @abstractmethod
    def objective(self, *args) -> Tensor:
        """Compute the objective value for a batch of samples."""
        pass

    @staticmethod
    def _reduce_violations(violations: dict[str, Tensor], reduction: str, dim: int = 1):
        """Apply mean/sum/max to every value in a dictionary."""
        for k, v in violations.items():
            violations[k] = OPFViolation._reduce_violation(v, reduction, dim=dim)

        return violations
    
    @staticmethod
    def _reduce_violation(violation: Tensor, reduction: str, dim: int = 1):
        """Apply mean/sum/max to a single tensor."""
        if reduction == "mean":
            return violation.mean(dim=dim)
        elif reduction == "sum":
            return violation.sum(dim=dim)
        elif reduction == "max":
            return violation.max(dim=dim).values
        elif reduction is None or reduction == "none":
            return violation
        else:
            raise ValueError(f"Invalid {reduction=}. Supported: {OPFViolation.SUPPORTED_REDUCTIONS}")

    def forward(self, *args, **kwargs):
        """Pass-through for `OPFViolation.calc_violations`"""
        return self.calc_violations(*args, **kwargs)

    @staticmethod
    def _clamped_bound_residual(x: Tensor, xmin: Tensor, xmax: Tensor, clamp: bool):
        g_lower, g_upper = MOF.bound_residual(x, xmin, xmax)

        return (
            MOF.inequality_violation(g_lower, clamp=clamp),
            MOF.inequality_violation(g_upper, clamp=clamp),
        )
