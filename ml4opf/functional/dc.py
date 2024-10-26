"""DCOPF Functional interface"""

import torch
from torch import Tensor


@torch.jit.script
def pf_from_dva(dva: Tensor, b: Tensor) -> Tensor:
    r"""Compute power flow given voltage angle differences.

    \[  \text{pf} = -b \cdot \text{dva}  \]

    Args:
        dva (Tensor): Voltage angle differences. (batch_size, nbranch)
        b (Tensor): Branch susceptance. (nbranch,)

    Returns:
        Tensor: Power flow. (batch_size, nbranch)
    """
    return -b * dva


@torch.jit.script
def balance_residual(pf_bus: Tensor, pt_bus: Tensor, pg_bus: Tensor, pd_bus: Tensor, gs_bus: Tensor) -> Tensor:
    r"""Compute power balance residual.

    \[  \text{g_balance} = \text{pg_bus} - \text{pd_bus} - \text{gs_bus} - \text{pf_bus} + \text{pt_bus}  \]

    Args:
        pf_bus (Tensor): Power flow from bus. (batch_size, nbus)
        pt_bus (Tensor): Power flow to bus. (batch_size, nbus)
        pg_bus (Tensor): Power generation at bus. (batch_size, nbus)
        pd_bus (Tensor): Power demand at bus. (batch_size, nbus)
        gs_bus (Tensor): Shunt conductance at bus. (batch_size, nbus)

    Returns:
        Tensor: Power balance residual. (batch_size, nbus)
    """
    return pg_bus - pd_bus - gs_bus - pf_bus + pt_bus


@torch.jit.script
def ohm_violation(pf: Tensor, dva: Tensor, b: Tensor) -> Tensor:
    r"""Compute Ohm's law violation.

    \[  \text{g_ohm} = - b \cdot \text{dva} - \text{pf} \]

    Args:
        pf (Tensor): Power flow. (batch_size, nbranch)
        dva (Tensor): Voltage angle differences. (batch_size, nbranch)
        b (Tensor): Branch susceptance. (nbranch,)

    Returns:
        Tensor: Ohm's law violation. (batch_size, nbranch)
    """

    return pf - pf_from_dva(dva, b)


@torch.jit.script
def objective(pg: Tensor, c0: Tensor, c1: Tensor) -> Tensor:
    r"""Compute DCOPF objective function.

    \[  \text{objective} = \sum_{i=1}^{n} \text{cost}_{2,i} + \text{cost}_{1,i} \cdot \text{pg}_i  \]

    Args:
        pg (Tensor): Power generation. (batch_size, ngen)
        cost (Tensor): Cost coefficients. (ngen, 3)

    Returns:
        Tensor: Objective function. (batch_size,)
    """
    return (c1 * pg + c0).sum(dim=1)
