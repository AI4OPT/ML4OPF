import torch

from torch import Tensor
from typing import Union, Optional
from functools import partial

import ml4opf.functional as MOF
import ml4opf.functional.ed as MOFED
from ml4opf.formulations.violation import OPFViolation


class EDViolation(OPFViolation):
    def __init__(self, json_data: dict[str, Union[bool, int, float, dict[str, Tensor]]], ptdf: Tensor):
        super().__init__(json_data)

        self.pad2d = partial(
            torch.nn.functional.pad, pad=(0, 1, 0, 0), mode="constant", value=0
        )  # adds a column of zeros to the right of a 2d tensor
        self.pad1d = partial(
            torch.nn.functional.pad, pad=(0, 1), mode="constant", value=0
        )  # adds a zero to the right of a 1d tensor

        branch_data: dict[str, Tensor] = json_data["branch"]
        gen_data: dict[str, Tensor] = json_data["gen"]

        self.register_buffer("ptdf", ptdf, persistent=False)

        self.register_buffer("rate_a", branch_data["rate_a"], persistent=False)

        self.register_buffer("cost", gen_data["cost"], persistent=False)
        self.register_buffer("pmin", gen_data["pmin"], persistent=False)
        self.register_buffer("pmax", gen_data["pmax"], persistent=False)

        self.register_buffer("rcost", torch.zeros_like(self.cost), persistent=False)
        # self.register_buffer("rcost", gen_data.get("reserve_cost", torch.zeros_like(self.cost)), persistent=False)
        # self.register_buffer("rmin", gen_data.get("rmin", torch.zeros_like(self.pmin)), persistent=False
        # self.register_buffer("rmax", gen_data["rmax"], persistent=False)

        penalty = json_data.get("penalty", {})  # TODO: verify OPFGenerator puts penalties here
        self.register_buffer("pb_penalty", torch.as_tensor(penalty.get("balance", 350000.0)), persistent=False)
        self.register_buffer("pr_penalty", torch.as_tensor(penalty.get("reserve", 110000.0)), persistent=False)
        self.register_buffer("tr_penalty", torch.as_tensor(penalty.get("transmission", 150000.0)), persistent=False)

    def objective(
        self,
        pd: Tensor,
        pg: Tensor,
        pr: Optional[Tensor] = None,
        pb_surplus: Optional[Tensor] = None,
        pb_shortage: Optional[Tensor] = None,
        pr_shortage: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute DCOPF objective function."""
        if pb_surplus is None:
            pb_surplus = torch.zeros((pg.shape[0],), dtype=pg.dtype, device=pg.device, requires_grad=pg.requires_grad)
        if pb_shortage is None:
            pb_shortage = torch.zeros((pg.shape[0],), dtype=pg.dtype, device=pg.device, requires_grad=pg.requires_grad)
        if pr_shortage is None:
            pr_shortage = torch.zeros((pg.shape[0],), dtype=pg.dtype, device=pg.device, requires_grad=pg.requires_grad)
        if pr is None:
            pr = torch.zeros_like(pg)

        pf = self.pf_from_pdpg(pd, pg)
        g_pf_lower, g_pf_upper = self.calc_pf_bound_residual(pf, clamp=True)
        df = g_pf_lower + g_pf_upper

        return MOFED.objective(
            pg,
            pr,
            pb_surplus,
            pb_shortage,
            pr_shortage,
            df,
            self.cost,
            self.rcost,
            self.pb_penalty,
            self.pr_penalty,
            self.tr_penalty,
        )

    def calc_pg_bound_residual(self, pg: Tensor, clamp: bool = False) -> Tensor:
        """Compute power generation bound residual."""
        g_pg_lower, g_pg_upper = MOF.bound_residual(pg, self.pmin, self.pmax)

        return (
            MOF.inequality_violation(g_pg_lower, clamp=clamp),
            MOF.inequality_violation(g_pg_upper, clamp=clamp),
        )

    def calc_pf_bound_residual(self, pf: Tensor, df: Optional[Tensor] = None, clamp: bool = False) -> Tensor:
        """Compute power flow bound residual."""
        if df is None:
            df = torch.zeros_like(pf)

        g_pf_lower, g_pf_upper = MOF.bound_residual(pf, -self.rate_a - df, self.rate_a + df)

        return (
            MOF.inequality_violation(g_pf_lower, clamp=clamp),
            MOF.inequality_violation(g_pf_upper, clamp=clamp),
        )

    def calc_balance_residual(
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
        """Compute all DCOPF violations."""
        # NOTE: assumes soft thermal limits, no reserves

        g_pg_lower, g_pg_upper = self.calc_pg_bound_residual(pg, clamp=clamp)
        g_p_balance = self.calc_balance_residual(pd, pg, clamp=clamp).unsqueeze(-1)

        pf = self.pf_from_pdpg(pd, pg)

        g_pf_lower, g_pf_upper = self.calc_pf_bound_residual(pf, clamp=clamp)

        violations = {
            "pg_lower": g_pg_lower,
            "pg_upper": g_pg_upper,
            "p_balance": g_p_balance,
            "pf_lower": g_pf_lower,
            "pf_upper": g_pf_upper,
        }

        return self._reduce_violations(violations, reduction)

    @property
    def violation_shapes(self) -> dict[str, int]:
        return {
            "pg_lower": self.n_gen.item(),
            "pg_upper": self.n_gen.item(),
            "p_balance": 1,
            "pf_lower": self.n_branch.item(),
            "pf_upper": self.n_branch.item(),
        }

    # def _calc_reserve_residual(pr: Tensor, MRR: Tensor, dr: Tensor) -> Tensor:
    #     return MRR - pr.sum(dim=-1) - dr

    # def calc_reserve_residual(
    #     self, pr: Tensor, MRR: Tensor, dr: Optional[Tensor] = None, clamp: bool = False
    # ) -> Tensor:
    #     """Compute reserve violation."""
    #     if dr is None:
    #         dr = torch.zeros((pr.shape[0],), dtype=pr.dtype, device=pr.device, requires_grad=pr.requires_grad)
    #     g_reserve = self._calc_reserve_residual(pr, MRR, dr)

    #     return MOF.inequality_violation(g_reserve, clamp=clamp)

    # @staticmethod
    # @torch.jit.script
    # def _calc_total_generation_residual(pg: Tensor, pr: Tensor, pmax: Tensor) -> Tensor:
    #     return pg + pr - pmax

    # def calc_total_generation_residual(self, pg: Tensor, pr: Tensor, pmax: Tensor, clamp: bool = False) -> Tensor:
    #     """Compute total generation violation."""
    #     g_total_generation = self._calc_total_generation_residual(pg, pr, pmax)

    #     return MOF.inequality_violation(g_total_generation, clamp=clamp)

    # def calc_pr_bound_residual(self, pr: Tensor, rmin: Tensor, rmax: Tensor, clamp: bool = False) -> Tensor:
    #     """Compute reserve bound residual."""
    #     g_pr_lower, g_pr_upper = MOF.bound_residual(pr, rmin, rmax)

    #     return (
    #         MOF.inequality_violation(g_pr_lower, clamp=clamp),
    #         MOF.inequality_violation(g_pr_upper, clamp=clamp),
    #     )

    # def calc_ptdf_residual(
    #     self, pd: Tensor, pg: Tensor, pf: Tensor, dense_incidence: bool = False, clamp: bool = False
    # ) -> Tensor:
    #     """Compute PTDF residual."""
    #     if dense_incidence:
    #         Al = self.load_incidence_dense
    #         Ag = self.generator_incidence_dense
    #     else:
    #         Al = self.load_incidence
    #         Ag = self.generator_incidence

    #     g_ptdf = MOFED.calc_ptdf_residual(pd, Al, pg, Ag, pf, self.ptdf)

    #     return MOF.equality_violation(g_ptdf, clamp=clamp)