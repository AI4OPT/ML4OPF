import torch

from torch import Tensor
from typing import Union, Optional
from functools import partial

import ml4opf.functional as MOF
import ml4opf.functional.dcp as MOFDCP
from ml4opf.formulations.violation import OPFViolation
from ml4opf import warn


class DCPViolation(OPFViolation):
    """`OPFViolation` for DCPPowerModel/DCOPF"""

    def __init__(self, json_data: dict[str, Union[bool, int, float, dict[str, Tensor]]]):
        super().__init__(json_data)

        self.pad2d = partial(
            torch.nn.functional.pad, pad=(0, 1, 0, 0), mode="constant", value=0
        )  # adds a column of zeros to the right of a 2d tensor
        self.pad1d = partial(
            torch.nn.functional.pad, pad=(0, 1), mode="constant", value=0
        )  # adds a zero to the right of a 1d tensor

        branch_data: dict[str, Tensor] = json_data["branch"]
        gen_data: dict[str, Tensor] = json_data["gen"]
        bus_data: dict[str, Tensor] = json_data["bus"]
        shunt_data: dict[str, Tensor] = json_data["shunt"]

        r = branch_data["br_r"]
        x = branch_data["br_x"]
        r2x2 = torch.pow(r, 2) + torch.pow(x, 2)
        b = -x / (r2x2)

        gs = shunt_data["gs"]

        slack_buses = (bus_data["bus_type"] == 3).nonzero()

        if len(slack_buses) != 1:
            warn(f"Expected 1 slack bus, found {len(slack_buses)} buses with bus_type == 3.")  # pragma: no cover

        self.register_buffer("slackbus_idx", slack_buses[0].squeeze(), persistent=False)

        self.register_buffer("b", b, persistent=False)
        self.register_buffer("angmin", branch_data["angmin"], persistent=False)
        self.register_buffer("angmax", branch_data["angmax"], persistent=False)
        self.register_buffer("rate_a", branch_data["rate_a"], persistent=False)

        self.register_buffer("cost", gen_data["cost"], persistent=False)
        self.register_buffer("pmin", gen_data["pmin"], persistent=False)
        self.register_buffer("pmax", gen_data["pmax"], persistent=False)

        self.register_buffer("gs_bus", self.pad1d(gs)[self.shunt_per_bus].sum(dim=1), persistent=False)

    def angle_difference(self, va: Tensor) -> Tensor:
        r"""Compute the angle differences per branch given the voltage angles per bus.

        The branch indices are assumed to be constant for the batch, matching the reference case.

        \[  \text{dva} = \boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}  \]

        Args:
            va (Tensor): Voltage angles per bus ( \(\boldsymbol{\theta}\) ). (batch_size, nbus)

        Returns:
            Tensor: Angle differences per branch. (batch_size, nbranch)
        """
        return MOF.angle_difference(va, self.fbus_per_branch, self.tbus_per_branch)

    def objective(self, pg: Tensor) -> Tensor:
        r"""Compute DCOPF objective function.

        Cost is assumed to be constant for the batch, matching the reference case.

        \[  \text{objective} = \sum_i^n \text{cost}_{2,i} + \text{cost}_{1,i} \cdot \text{pg}_i  \]

        Args:
            pg (Tensor): Power generation per generator. (batch_size, ngen)

        Returns:
            Tensor: Objective function value. (batch_size, 1)
        """
        return MOFDCP.objective(pg, self.cost)

    def pf_from_va(self, va: Tensor) -> Tensor:
        r"""Compute power flow given voltage angles.

        \[ \mathbf{p}_f = -\text{b} \cdot (\boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}) \]

        Args:
            va (Tensor): Voltage angles per bus ( \(\boldsymbol{\theta}\) ). (batch_size, nbus)

        Returns:
            Tensor: Power flow per branch ( \(\mathbf{p}_f\) ). (batch_size, nbranch)
        """
        dva = self.angle_difference(va)
        return MOFDCP.pf_from_dva(dva, self.b)

    @property
    def nrate_a(self) -> Tensor:
        return -self.rate_a

    def pf_bound_residual(self, pf: Tensor, clamp: bool = False) -> Tensor:
        r"""Calculate the power flow bound residual.

        \[ g_{\text{lower}} = -\text{rate_a} - \text{pf} \]
        \[ g_{\text{upper}} = \text{pf} - \text{rate_a} \]

        Args:
            pf (Tensor): Power flow per branch. (batch_size, nbranch)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbranch)
            Tensor: Upper bound residual. (batch_size, nbranch)
        """
        return self._clamped_bound_residual(pf, self.nrate_a, self.rate_a, clamp=clamp)

    def pg_bound_residual(self, pg: Tensor, clamp: bool = False) -> Tensor:
        r"""Calculate the power generation bound residual.

        \[ g_{\text{lower}} = \text{pmin} - \text{pg} \]
        \[ g_{\text{upper}} = \text{pg} - \text{pmax} \]

        Args:
            pg (Tensor): Active power generation per generator. (batch_size, ngen)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Lower bound residual. (batch_size, ngen)
            Tensor: Upper bound residual. (batch_size, ngen)
        """
        return self._clamped_bound_residual(pg, self.pmin, self.pmax, clamp=clamp)

    def dva_bound_residual(self, dva: Tensor, clamp: bool = False) -> Tensor:
        r"""Calculate the voltage angle difference bound residual.

        \[ g_{\text{lower}} = \text{angmin} - \text{dva} \]
        \[ g_{\text{upper}} = \text{dva} - \text{angmax} \]

        Args:
            dva (Tensor): Voltage angle differences per branch. (batch_size, nbranch)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbranch)
            Tensor: Upper bound residual. (batch_size, nbranch)
        """
        return self._clamped_bound_residual(dva, self.angmin, self.angmax, clamp=clamp)

    def balance_residual(
        self, pd: Tensor, pg: Tensor, pf: Tensor, embed_method: str = "pad", clamp: bool = True
    ) -> Tensor:
        r"""Compute power balance residual.

        \[  \text{g_balance} = \text{pg_bus} - \text{pd_bus} - \text{gs_bus} - \text{pf_bus} - \text{pt_bus}  \]

        Args:
            pd (Tensor): Power demand per bus. (batch_size, nbus)
            pg (Tensor): Power generation per generator. (batch_size, ngen)
            pf (Tensor): Power flow per branch. (batch_size, nbranch)
            embed_method (str, optional): Embedding method to convert component-wise values to bus-wise -- one of "pad", "dense_matrix", or "matrix". Defaults to "pad".
            clamp (bool, optional): Clamp to extract only violations. Defaults to True.

        Returns:
            Tensor: Power balance residual. (batch_size, nbus)
        """

        pf_bus = self.branch_from_to_bus(pf, method=embed_method)
        pt_bus = -self.branch_to_to_bus(pf, method=embed_method)
        pg_bus = self.gen_to_bus(pg, method=embed_method)
        pd_bus = self.load_to_bus(pd, method=embed_method)

        g_balance = MOFDCP.balance_residual(pf_bus, pt_bus, pg_bus, pd_bus, self.gs_bus)

        return MOF.equality_violation(g_balance, clamp=clamp)

    def ohm_residual(self, pf: Tensor, dva: Tensor, clamp: bool = False) -> Tensor:
        r"""Compute Ohm's law violation.

        \[  \text{g_ohm} = - b \cdot \text{dva} - \text{pf}  \]

        Args:
            pf (Tensor): Power flow per branch. (batch_size, nbranch)
            dva (Tensor): Voltage angle differences per branch. (batch_size, nbranch)
            clamp (bool, optional): Clamp to extract only violations. Defaults to False.

        Returns:
            Tensor: Ohm's law violation. (batch_size, nbranch)
        """
        g_ohm = MOFDCP.ohm_violation(pf, dva, self.b)

        return MOF.equality_violation(g_ohm, clamp=clamp)

    def calc_violations(
        self,
        pd: Tensor,
        pg: Tensor,
        va: Tensor,
        pf: Optional[Tensor]=None,
        reduction: str = "mean",
        clamp: bool = True,
        embed_method: str = "pad",
    ) -> dict[str, Tensor]:
        """Compute all DCOPF violations.

        Args:
            pd (Tensor): Power demand per bus. (batch_size, nbus)
            pg (Tensor): Power generation per generator. (batch_size, ngen)
            va (Tensor): Voltage angles per bus. (batch_size, nbus)
            pf (Tensor, optional): Power flow per branch. Defaults to None.
            reduction (str, optional): Reduction method. Defaults to "mean".
            clamp (bool, optional): Clamp to extract only violations. Defaults to True.
            embed_method (str, optional): Embedding method to convert component-wise values to bus-wise -- one of "pad", "dense_matrix", or "matrix". Defaults to "pad".

        Returns:
            dict[str, Tensor]: Dictionary of all violations:

            - "pg_lower": Lower bound violation of power generation. (batch_size, ngen)

            - "pg_upper": Upper bound violation of power generation. (batch_size, ngen)

            - "dva_lower": Lower bound violation of voltage angle difference. (batch_size, nbranch)

            - "dva_upper": Upper bound violation of voltage angle difference. (batch_size, nbranch)

            - "pf_lower": Lower bound violation of power flow. (batch_size, nbranch)

            - "pf_upper": Upper bound violation of power flow. (batch_size, nbranch)

            - "p_balance": Power balance violation. (batch_size, nbus)

            - "ohm": Ohm's law violation. (batch_size, nbranch)
        """
        dva = self.angle_difference(va)

        if pf is None:
            pf = self.pf_from_va(va)

        g_pg_lower, g_pg_upper = self.pg_bound_residual(pg, clamp=clamp)
        g_dva_lower, g_dva_upper = self.dva_bound_residual(dva, clamp=clamp)
        g_pf_lower, g_pf_upper = self.pf_bound_residual(pf, clamp=clamp)
        g_p_balance = self.balance_residual(pd, pg, pf, embed_method=embed_method, clamp=clamp)
        g_ohm = self.ohm_residual(pf, dva, clamp=clamp)

        violations = {
            "pg_lower": g_pg_lower,
            "pg_upper": g_pg_upper,
            "dva_lower": g_dva_lower,
            "dva_upper": g_dva_upper,
            "pf_lower": g_pf_lower,
            "pf_upper": g_pf_upper,
            "p_balance": g_p_balance,
            "ohm": g_ohm,
        }

        return self._reduce_violations(violations, reduction)

    @property
    def violation_shapes(self) -> dict[str, int]:
        return {
            "pg_lower": self.n_gen.item(),
            "pg_upper": self.n_gen.item(),
            "dva_lower": self.n_branch.item(),
            "dva_upper": self.n_branch.item(),
            "pf_lower": self.n_branch.item(),
            "pf_upper": self.n_branch.item(),
            "p_balance": self.n_bus.item(),
            "ohm": self.n_branch.item(),
        }
