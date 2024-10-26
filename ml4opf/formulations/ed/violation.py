"""Class interface for EconomicDispatch constraints, objective, etc."""

from typing import Optional

import torch
from torch import Tensor

import ml4opf.functional as MOF
import ml4opf.functional.ed as MOFED
from ml4opf.formulations.violation import OPFViolation


class EDViolation(OPFViolation):
    """`OPFViolation` for EconomicDispatch"""

    def __init__(self, data: dict, ptdf: Tensor):
        super().__init__(data)

        self.register_buffer("ptdf", ptdf, persistent=False)

        penalty = data.get("penalty", {})
        self.register_buffer("tr_penalty", torch.as_tensor(penalty.get("transmission", 150000.0)), persistent=False)

    def objective(
        self,
        pd: Tensor,
        pg: Tensor,
    ) -> Tensor:
        """Compute ED objective function."""

        pf = self.pf_from_pdpg(pd, pg)
        g_pf_lower, g_pf_upper = self.pf_bound_residual(pf, clamp=True)
        df = g_pf_lower + g_pf_upper

        return MOFED.objective(
            pg,
            df,
            self.c0,
            self.c1,
            self.tr_penalty,
        )

    def pg_bound_residual(self, pg: Tensor, clamp: bool = False) -> Tensor:
        """Compute power generation bound residual."""
        return self.clamped_bound_residual(pg, self.pgmin, self.pgmax, clamp)

    def pf_bound_residual(self, pf: Tensor, df: Optional[Tensor] = None, clamp: bool = False) -> Tensor:
        """Compute power flow bound residual."""
        if df is None:
            df = torch.zeros_like(pf)

        g_pf_lower, g_pf_upper = MOF.bound_residual(pf, -self.smax - df, self.smax + df)

        return (
            MOF.inequality_violation(g_pf_lower, clamp=clamp),
            MOF.inequality_violation(g_pf_upper, clamp=clamp),
        )

    def balance_residual(
        self,
        pd: Tensor,
        pg: Tensor,
        dpb_surplus: Optional[Tensor] = None,
        dpb_shortage: Optional[Tensor] = None,
        clamp: bool = False,
    ) -> Tensor:
        """Compute power balance residual."""
        if dpb_surplus is None:
            dpb_surplus = torch.zeros((pg.shape[0],), dtype=pg.dtype, device=pg.device, requires_grad=pg.requires_grad)
        if dpb_shortage is None:
            dpb_shortage = torch.zeros((pg.shape[0],), dtype=pg.dtype, device=pg.device, requires_grad=pg.requires_grad)
        g_p_balance = MOFED.calc_balance_residual(pd, pg, dpb_surplus, dpb_shortage)

        return MOF.equality_violation(g_p_balance, clamp=clamp)

    def pf_from_pdpg(self, pd: Tensor, pg: Tensor, dense_incidence: bool = False) -> Tensor:
        """Compute power flow from power demand and power generation."""
        if dense_incidence:
            Al = self.load_incidence_dense
            Ag = self.generator_incidence_dense
        else:
            Al = self.load_incidence
            Ag = self.generator_incidence

        return MOFED.pf_from_pdpg(pd, pg, Al, Ag, self.ptdf)

    def calc_violations(self, pd: Tensor, pg: Tensor, reduction: str = "mean", clamp: bool = True) -> dict[str, Tensor]:
        """Compute all EconomicDispatch violations."""
        # NOTE: assumes soft thermal limits, no reserves

        g_pg_lower, g_pg_upper = self.pg_bound_residual(pg, clamp=clamp)
        g_p_balance = self.balance_residual(pd, pg, clamp=clamp).unsqueeze(-1)

        pf = self.pf_from_pdpg(pd, pg)

        g_pf_lower, g_pf_upper = self.pf_bound_residual(pf, clamp=clamp)

        violations = {
            "pg_lower": g_pg_lower,
            "pg_upper": g_pg_upper,
            "p_balance": g_p_balance,
            "pf_lower": g_pf_lower,
            "pf_upper": g_pf_upper,
        }

        return self.reduce_violations(violations, reduction)

    @property
    def violation_shapes(self) -> dict[str, int]:
        return {
            "pg_lower": self.n_gen.item(),
            "pg_upper": self.n_gen.item(),
            "p_balance": 1,
            "pf_lower": self.n_branch.item(),
            "pf_upper": self.n_branch.item(),
        }
