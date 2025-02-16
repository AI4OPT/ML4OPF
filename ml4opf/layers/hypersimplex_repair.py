"""Differentiable repair layer for the hyper-simplex constraint, ∑x=X, x̲≤x≤x̅."""

from typing import Optional

import torch
from torch import nn, Tensor


class HyperSimplexRepair(nn.Module):
    """Repair layer for the hyper-simplex constraint  ∑x=X, x̲≤x≤x̅."""

    def __init__(
        self,
        xmin: Optional[Tensor] = None,
        xmax: Optional[Tensor] = None,
        X: Optional[Tensor] = None,
    ):
        super().__init__()

        self.register_buffer("xmin", xmin, persistent=False)
        self.register_buffer("xmax", xmax, persistent=False)
        self.register_buffer("X", X, persistent=False)

        self.repair_up = torch.vmap(
            HyperSimplexRepair._repair_up,
            in_dims=(
                0,
                0,
                0 if X is None else None,
            ),
        )

        self.repair_dn = torch.vmap(
            HyperSimplexRepair._repair_dn,
            in_dims=(
                0,
                0,
                0 if X is None else None,
            ),
        )

    @staticmethod
    def _repair_up(x: Tensor, xmax: Tensor, X: Tensor):
        ratio = (X - x.sum()) / (xmax.sum() - x.sum())
        return ratio * xmax + (1 - ratio) * x

    @staticmethod
    def _repair_dn(x: Tensor, xmin: Tensor, X: Tensor):
        ratio = (X - x.sum()) / (xmin.sum() - x.sum())
        return ratio * xmin + (1 - ratio) * x

    def forward(
        self,
        x: Tensor,
        xmin: Optional[Tensor] = None,
        xmax: Optional[Tensor] = None,
        X: Optional[Tensor] = None,
    ):
        """Project onto ∑x=X, x̲≤x≤x̅"""
        xmin = self.xmin if xmin is None else xmin
        xmax = self.xmax if xmax is None else xmax
        X = self.X if X is None else X

        raw_X = x.sum(dim=-1)
        need_up = raw_X < X
        need_dn = raw_X > X

        if xmin.ndim == 1:
            xmin = xmin.expand_as(x)
        if xmax.ndim == 1:
            xmax = xmax.expand_as(x)

        x_ = x.clone()
        if need_up.any():
            x_[need_up, :] = self.repair_up(x[need_up, :], xmax[need_up, :], X[need_up])
        if need_dn.any():
            x_[need_dn, :] = self.repair_dn(x[need_dn, :], xmin[need_dn, :], X[need_dn])

        return x_
