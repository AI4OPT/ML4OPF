import torch, torch.nn as nn

from typing import Optional
from torch import Tensor

# Original implementation by Wenbo Chen
class HyperSimplexRepair(nn.Module):
    r"""
    Recovery for hyper simplex constraint \(\{ x_i \in [l_i, u_i] \,\vert\, \sum x = b \}\).
    """
    # TODO: throw if infeasible
    def __init__(self, lb: Optional[Tensor] = None, ub: Optional[Tensor] = None, sanity_check: bool = True):
        super().__init__()
        if ub is not None:
            self.register_buffer("ub", ub)
        if lb is not None:
            self.register_buffer("lb", lb)
        self.sanity_check = sanity_check

    def forward(self, x_: Tensor, b: Tensor):
        # x = x_.clone()

        # if ub is None:
        ub = self.ub.clone()
        # if lb is None:
        lb = self.lb.clone()

        fixed_mask = (ub == lb)
        x  = x_[:, ~fixed_mask]
        ub = ub[~fixed_mask]
        lb = lb[~fixed_mask]
        b -= x_[:, fixed_mask].sum(dim=-1)

        if self.sanity_check:
            assert (ub > lb).all(), "lb must be less than ub"
            assert (x <= ub).all(), "x must be less than or equal to ub"
            assert (x >= lb).all(), "x must be greater than or equal to lb"
            assert x.shape[0] == b.shape[0], "x and b should have the same batch size"
            assert x.shape[1] == ub.shape[0], "x and ub should have the same number of columns"
            assert x.shape[1] == lb.shape[0], "x and lb should have the same number of columns"

        b = b.view(-1)

        x = self.proportional_projection(x.clone(), lb.expand_as(x), ub.expand_as(x), b)

        ret = x_.clone()
        ret[:, ~fixed_mask] = x
        return ret

    @staticmethod
    @torch.jit.script
    def proportional_projection(x: Tensor, lb: Tensor, ub: Tensor, b: Tensor):

        # set x to lb if b <= ∑ lb
        b_less_lb_mask = b <= lb.sum(dim=-1)
        x[b_less_lb_mask, :] = lb[b_less_lb_mask, :]

        # set x to ub if b >= ∑ ub
        b_greater_lb_mask = b >= ub.sum(dim=-1)
        x[b_greater_lb_mask, :] = ub[b_less_lb_mask, :]
        project_mask = torch.logical_and(~b_less_lb_mask, ~b_greater_lb_mask)

        G = x.shape[1]
        sum_x = x.sum(dim=-1)
        sum_ub = ub.sum(dim=-1)
        sum_lb = lb.sum(dim=-1)
        # proportionally increase x if ∑ x < b
        project_up = torch.logical_and(b > sum_x, project_mask)
        ratio_up = ((b - sum_x) / (sum_ub - sum_x)).unsqueeze(dim=-1).repeat(1, G)
        ratio_up = ratio_up[project_up, :]
        x[project_up, :] = ratio_up * ub[project_up, :] + (1 - ratio_up) * x[project_up, :]

        # proportionally decrease x if ∑ x > b
        project_dn = torch.logical_and(b < sum_x, project_mask)
        ratio_dn = ((b - sum_x) / (sum_lb - sum_x)).unsqueeze(dim=-1).repeat(1, G)
        ratio_dn = ratio_dn[project_dn, :]
        x[project_dn, :] = ratio_dn * lb[project_dn, :] + (1 - ratio_dn) * x[project_dn, :]
        return x
