import torch
from torch import Tensor


@torch.jit.script
def calc_balance_residual(pd: Tensor, pg: Tensor, dpb_surplus: Tensor, dpb_shortage: Tensor) -> Tensor:
    return pd.sum(dim=-1) - pg.sum(dim=-1) - dpb_surplus + dpb_shortage


# @torch.jit.script
# def calc_ptdf_residual(pd: Tensor, Al: Tensor, pg: Tensor, Ag: Tensor, pf: Tensor, ptdf: Tensor):
#     p = pg @ Ag.T - pd @ Al.T
#     return pf - p @ ptdf.T


@torch.jit.script
def pf_from_pdpg(pd: Tensor, pg: Tensor, Al: Tensor, Ag: Tensor, ptdf: Tensor) -> Tensor:
    """Compute power flow from power demand and power generation."""
    p = pg @ Ag.T - pd @ Al.T
    return p @ ptdf.T


@torch.jit.script
def compute_objective(
    pg: Tensor,
    pr: Tensor,
    pb_surplus: Tensor,
    pb_shortage: Tensor,
    pr_shortage: Tensor,
    df: Tensor,
    cost: Tensor,
    rcost: Tensor,
    pb_penalty: Tensor,
    pr_penalty: Tensor,
    tr_penalty: Tensor,
) -> Tensor:
    return (
        (cost[:, 2] + cost[:, 1] * pg).sum(dim=-1)
        + (rcost[:, 2] + rcost[:, 1] * pr).sum(dim=-1)
        + (pb_penalty * pb_surplus)
        + (pb_penalty * pb_shortage)
        + (pr_penalty * pr_shortage)
        + (tr_penalty * df).sum(dim=-1)
    )
