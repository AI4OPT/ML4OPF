""" Functional interface """

import torch

from torch import Tensor

from ml4opf.functional.incidence import (
    branch_incidence,
    branch_from_incidence,
    branch_to_incidence,
    generator_incidence,
    load_incidence,
    adjacency,
    map_to_bus_pad,
    map_to_bus_matrix,
)


@torch.jit.script
def bound_residual(x: Tensor, xmin: Tensor, xmax: Tensor) -> tuple[Tensor, Tensor]:
    r"""Calculate the bound residuals for a tensor x.

    \[  g_\text{lower} = x_\text{min} - x \]
    \[  g_\text{upper} = x - x_\text{max} \]

    Args:
        x (Tensor): Input tensor. (batch, n)
        xmin (Tensor): Lower bounds. (batch, n)
        xmax (Tensor): Upper bounds. (batch, n)

    Returns:
        tuple[Tensor, Tensor]: Lower and upper bound residuals. (batch, n)
    """
    x_min = xmin.expand_as(x)
    x_max = xmax.expand_as(x)

    g_lower = x_min - x
    g_upper = x - x_max

    return (g_lower, g_upper)


@torch.jit.script
def inequality_violation(residual: Tensor, clamp: bool = False) -> Tensor:
    """Return only the violation (always positive) if clamp=True. Otherwise, pass-through the residual."""
    if clamp:
        return torch.clamp(residual, min=0)
    else:
        return residual


@torch.jit.script
def equality_violation(residual: Tensor, clamp: bool = False) -> Tensor:
    """Return the absolute violation (always positive) if clamp=True. Otherwise, pass-through the residual."""
    if clamp:
        return residual.abs()
    else:
        return residual


@torch.jit.script
def angle_difference(va: Tensor, i: Tensor, j: Tensor) -> Tensor:
    va_i = va[:, i]
    va_j = va[:, j]

    return va_i - va_j


import ml4opf.functional.ac as AC
import ml4opf.functional.dc as DC
import ml4opf.functional.incidence as incidence

__all__ = [
    "branch_incidence",
    "branch_from_incidence",
    "branch_to_incidence",
    "generator_incidence",
    "load_incidence",
    "adjacency",
    "map_to_bus_pad",
    "map_to_bus_matrix",
    "bound_residual",
    "inequality_violation",
    "equality_violation",
    "angle_difference",
    "AC",
    "DC",
    "incidence",
]
