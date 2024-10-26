"""EconomicDispatch Functional interface"""

import torch
from torch import Tensor


@torch.jit.script
def calc_balance_residual(pd: Tensor, pg: Tensor, dpb_surplus: Tensor, dpb_shortage: Tensor) -> Tensor:
    """Compute global power balance residual."""
    return pd.sum(dim=-1) - pg.sum(dim=-1) - dpb_surplus + dpb_shortage


@torch.jit.script
def pf_from_pdpg(pd: Tensor, pg: Tensor, Al: Tensor, Ag: Tensor, ptdf: Tensor) -> Tensor:
    """Compute power flow from power demand and power generation."""
    p = pg @ Ag.T - pd @ Al.T
    return p @ ptdf.T


@torch.jit.script
def objective(
    pg: Tensor,
    df: Tensor,
    c0: Tensor,
    c1: Tensor,
    tr_penalty: Tensor,
) -> Tensor:
    """Compute EDSoftThermal objective function."""
    return (c0 + c1 * pg).sum(dim=-1) + (tr_penalty * df).sum(dim=-1)
