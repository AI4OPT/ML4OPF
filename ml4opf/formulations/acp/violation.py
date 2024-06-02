import torch, numpy as np

from torch import Tensor
from functools import partial
from typing import Optional, Union

import ml4opf.functional as MOF
import ml4opf.functional.acp as MOFACP
from ml4opf.formulations.violation import OPFViolation
from ml4opf import warn


class ACPViolation(OPFViolation):
    """`OPFViolation` for ACPPowerModel/ACOPF"""

    def __init__(self, json_data: dict[str, Union[bool, int, float, dict[str, Tensor]]]):
        super().__init__(json_data)

        self.pad2d = partial(
            torch.nn.functional.pad, pad=(0, 1, 0, 0), mode="constant", value=0
        )  # adds a column of zeros to the right of a 2d tensor
        self.pad1d = partial(
            torch.nn.functional.pad, pad=(0, 1), mode="constant", value=0
        )  # adds a zero to the right of a 1d tensor

        branch_data: dict[str, Tensor] = json_data["branch"]
        shunt_data: dict[str, Tensor] = json_data["shunt"]
        gen_data: dict[str, Tensor] = json_data["gen"]
        bus_data: dict[str, Tensor] = json_data["bus"]

        # load network data
        gs = shunt_data["gs"]
        bs = shunt_data["bs"]
        s1max = s2max = branch_data["rate_a"]

        slack_buses = (bus_data["bus_type"] == 3).nonzero()

        if len(slack_buses) != 1:
            warn(f"Expected 1 slack bus, found {len(slack_buses)} buses with bus_type == 3.")  # pragma: no cover

        self.register_buffer("slackbus_idx", slack_buses[0].squeeze(), persistent=False)

        # load + store bounds, cost coefficients, vnom
        self.register_buffer("pmin", gen_data["pmin"], persistent=False)
        self.register_buffer("pmax", gen_data["pmax"], persistent=False)
        self.register_buffer("qmin", gen_data["qmin"], persistent=False)
        self.register_buffer("qmax", gen_data["qmax"], persistent=False)
        self.register_buffer("cost", gen_data["cost"], persistent=False)
        self.register_buffer("vmin", bus_data["vmin"], persistent=False)
        self.register_buffer("vnom", bus_data["base_kv"], persistent=False)
        self.register_buffer("vmax", bus_data["vmax"], persistent=False)
        self.register_buffer("angmin", branch_data["angmin"], persistent=False)
        self.register_buffer("angmax", branch_data["angmax"], persistent=False)

        # embed gs/bs on bus level. equivalent when max 1 shunt per bus
        self.register_buffer("gs_bus", self.pad1d(gs)[self.shunt_per_bus].sum(dim=1), persistent=False)
        self.register_buffer("bs_bus", self.pad1d(bs)[self.shunt_per_bus].sum(dim=1), persistent=False)

        # pre-compute some constants
        c1, c2, c3, c4, c7, c8, c9, c10 = MOFACP._compute_flow_constants(
            branch_data["tap"],
            branch_data["shift"],
            branch_data["br_r"],
            branch_data["br_x"],
            branch_data["g_fr"],
            branch_data["g_to"],
            branch_data["b_fr"],
            branch_data["b_to"],
        )
        self.register_buffer("c1", c1, persistent=False)
        self.register_buffer("c2", c2, persistent=False)
        self.register_buffer("c3", c3, persistent=False)
        self.register_buffer("c4", c4, persistent=False)
        self.register_buffer("c7", c7, persistent=False)
        self.register_buffer("c8", c8, persistent=False)
        self.register_buffer("c9", c9, persistent=False)
        self.register_buffer("c10", c10, persistent=False)
        self.register_buffer("s1max_sq", torch.pow(s1max, 2), persistent=False)
        self.register_buffer("s2max_sq", torch.pow(s2max, 2), persistent=False)

    @property
    def _flow_constants(self) -> tuple[Tensor, ...]:
        return self.c1, self.c2, self.c3, self.c4, self.c7, self.c8, self.c9, self.c10

    def angle_difference(self, va: Tensor) -> Tensor:
        r"""Compute the angle differences per branch given the voltage angles per bus.

        \[  \text{dva} = \boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}  \]

        Args:
            va (Tensor): Voltage angles per bus ( \(\boldsymbol{\theta}\) ). (batch_size, nbus)

        Returns:
            Tensor: Angle differences per branch. (batch_size, nbranch)
        """
        return MOF.angle_difference(va, self.fbus_per_branch, self.tbus_per_branch)

    def objective(self, pg: Tensor) -> Tensor:
        r"""Compute the objective function given the active power generation per generator.

        \[  \text{objective} = \sum_{i=1}^{n} \text{cost}_{2,i} + \text{cost}_{1,i} \cdot \text{pg}_i  \]

        Args:
            pg (Tensor): Active power generation per generator. (batch_size, ngen)

        Returns:
            Tensor: Objective function value. (batch_size)
        """
        return MOFACP.objective(pg, self.cost)

    def flows_from_voltage(self, vm: Tensor, dva: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Compute the power flows given the voltage magnitude per bus and angle differences per branch.

        The branch (transformer + line) parameters \( g, b, g_f, b_f, g_t, b_t, t_r, t_i \) and \( \text{tap} \)
        are assumed to be constant *for the life of this object*, matching the reference case.
        These constants are pre-computed using the `MOFACP._compute_flow_constants` function.

        \[ \mathbf{p}_f = \frac{g + g_{f}}{\text{tap}^2} \mathbf{v}_{f}^2 + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \cos(\boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}) + \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \sin(\boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}) \]
        \[ \mathbf{p}_t = (g + g_{t}) \mathbf{v}_{t}^2 + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \cos(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{f}) + \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \sin(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{f}) \]
        \[ \mathbf{q}_f = \frac{-(b + b_{f})}{\text{tap}^2} \mathbf{v}_{f}^2 - \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \cos(\boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}) + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \sin(\boldsymbol{\theta}_{f} - \boldsymbol{\theta}_{t}) \]
        \[ \mathbf{q}_t = -(b + b_{t}) \mathbf{v}_{t}^2 - \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \cos(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{f}) + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{v}_{f} \mathbf{v}_{t} \sin(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{f}) \]

        Args:
            vm (Tensor): Voltage magnitude per bus ( \(\mathbf{v}\) ). (batch_size, nbus)
            dva (Tensor): Angle differences per branch ( \( \boldsymbol{\theta}_f - \boldsymbol{\theta}_t \) ). (batch_size, nbranch)

        Returns:
            Tensor: Real power flow per branch ( \(\mathbf{p}_f\) ). (batch_size, nbranch)
            Tensor: Real power flow per branch ( \(\mathbf{p}_t\) ). (batch_size, nbranch)
            Tensor: Reactive power flow per branch ( \(\mathbf{q}_f\) ). (batch_size, nbranch)
            Tensor: Reactive power flow per branch ( \(\mathbf{q}_t\) ). (batch_size, nbranch)
        """

        return MOFACP._flows_from_voltage(
            vm,
            dva,
            self.fbus_per_branch,
            self.tbus_per_branch,
            *(self._flow_constants),
        )

    def flows_from_voltage_bus(self, vm: Tensor, va: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Compute the power flows given the voltage magnitude per bus and voltage angles per bus.

        This function computes angle differences then calls `ACPViolation.flows_from_voltage`.
        See the docstring of `ACPViolation.flows_from_voltage` for more details.

        Args:
            vm (Tensor): Voltage magnitude per bus ( \(\mathbf{v}\) ). (batch_size, nbus)
            va (Tensor): Voltage angle per bus ( \(\boldsymbol{\theta}\) ). (batch_size, nbus)

        Returns:
            Tensor: Real power flow per branch ( \(\mathbf{p}_f\) ). (batch_size, nbranch)
            Tensor: Real power flow per branch ( \(\mathbf{p}_t\) ). (batch_size, nbranch)
            Tensor: Reactive power flow per branch ( \(\mathbf{q}_f\) ). (batch_size, nbranch)
            Tensor: Reactive power flow per branch ( \(\mathbf{q}_t\) ). (batch_size, nbranch)
        """
        dva = self.angle_difference(va)
        return self.flows_from_voltage(vm, dva)

    def dva_bound_residual(self, dva: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the voltage angle difference bound residual.

        \[ g_{\text{lower}} = \text{angmin} - \text{dva} \]
        \[ g_{\text{upper}} = \text{dva} - \text{angmax} \]

        Args:
            dva (Tensor): Voltage angle difference per branch. (batch_size, nbranch)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbranch)
            Tensor: Upper bound residual. (batch_size, nbranch)
        """
        return self._clamped_bound_residual(dva, self.angmin, self.angmax, clamp=clamp)

    def vm_bound_residual(self, vm: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the voltage magnitude bound residual.

        \[ g_{\text{lower}} = \text{vmin} - \text{vm} \]
        \[ g_{\text{upper}} = \text{vm} - \text{vmax} \]

        Args:
            vm (Tensor): Voltage magnitude per bus. (batch_size, nbus)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbus)
            Tensor: Upper bound residual. (batch_size, nbus)
        """
        return self._clamped_bound_residual(vm, self.vmin, self.vmax, clamp=clamp)

    def pg_bound_residual(self, pg: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the active power generation bound residual.

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

    def qg_bound_residual(self, qg: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the reactive power generation bound residual.

        \[ g_{\text{lower}} = \text{qmin} - \text{qg} \]
        \[ g_{\text{upper}} = \text{qg} - \text{qmax} \]

        Args:
            qg (Tensor): Reactive power generation per generator. (batch_size, ngen)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Lower bound residual. (batch_size, ngen)
            Tensor: Upper bound residual. (batch_size, ngen)
        """
        return self._clamped_bound_residual(qg, self.qmin, self.qmax, clamp=clamp)

    def thermal_residual(
        self, pf: Tensor, pt: Tensor, qf: Tensor, qt: Tensor, clamp: bool = False
    ) -> tuple[Tensor, Tensor]:
        r"""Calculate the thermal limit residual.

        \[ g_{\text{thrm}_1} = \text{pf}^2 + \text{qf}^2 - \text{s1max} \]
        \[ g_{\text{thrm}_2} = \text{pt}^2 + \text{qt}^2 - \text{s2max} \]

        Args:
            pf (Tensor): Active power flow from bus per branch. (batch_size, nbranch)
            pt (Tensor): Active power flow to bus per branch. (batch_size, nbranch)
            qf (Tensor): Reactive power flow from bus per branch. (batch_size, nbranch)
            qt (Tensor): Reactive power flow to bus per branch. (batch_size, nbranch)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Thermal limit residual for from branch. (batch_size, nbranch)
            Tensor: Thermal limit residual for to branch. (batch_size, nbranch)
        """
        thrm_1, thrm_2 = MOFACP._thermal_residual_sq(pf, pt, qf, qt, self.s1max_sq, self.s2max_sq)

        return (
            MOF.inequality_violation(thrm_1, clamp=clamp),
            MOF.inequality_violation(thrm_2, clamp=clamp),
        )

    def balance_residual(
        self,
        pd: Tensor,
        qd: Tensor,
        pg: Tensor,
        qg: Tensor,
        vm: Tensor,
        pf: Tensor,
        pt: Tensor,
        qf: Tensor,
        qt: Tensor,
        clamp: bool = False,
        embed_method: str = "pad",
    ) -> tuple[Tensor, Tensor]:
        r"""Calculate the power balance residual.

        Component-wise tensors are first embedded to the bus level using `embed_method`.

        The shunt parameters \( g_s, b_s \) are assumed to be constant, matching the reference case.

        \[ \text{p_viol} = \text{pg_bus} - \text{pd_bus} - \text{pt_bus} - \text{pf_bus} - \text{gs_bus} \times \text{vm}^2 \]
        \[ \text{q_viol} = \text{qg_bus} - \text{qd_bus} - \text{qt_bus} - \text{qf_bus} + \text{bs_bus} \times \text{vm}^2 \]

        Args:
            pd (Tensor): Active power demand per bus. (batch_size, nbus)
            qd (Tensor): Reactive power demand per bus. (batch_size, nbus)
            pg (Tensor): Active power generation per generator. (batch_size, ngen)
            qg (Tensor): Reactive power generation per generator. (batch_size, ngen)
            vm (Tensor): Voltage magnitude per bus. (batch_size, nbus)
            pf (Tensor): Active power flow from bus per branch. (batch_size, nbranch)
            pt (Tensor): Active power flow to bus per branch. (batch_size, nbranch)
            qf (Tensor): Reactive power flow from bus per branch. (batch_size, nbranch)
            qt (Tensor): Reactive power flow to bus per branch. (batch_size, nbranch)
            clamp (bool, optional): Apply an absolute value to the residual. Defaults to False.
            embed_method (str, optional): Embedding method for bus-level components. Defaults to 'pad'. Must be one of 'pad', 'dense_matrix', or 'matrix. See `IncidenceMixin.*_to_bus`.

        Returns:
            Tensor: Power balance residual for active power. (batch_size, nbus)
            Tensor: Power balance residual for reactive power. (batch_size, nbus)
        """
        pd_bus = self.load_to_bus(pd, method=embed_method)
        qd_bus = self.load_to_bus(qd, method=embed_method)
        pg_bus = self.gen_to_bus(pg, method=embed_method)
        qg_bus = self.gen_to_bus(qg, method=embed_method)
        pf_bus = self.branch_from_to_bus(pf, method=embed_method)
        pt_bus = self.branch_to_to_bus(pt, method=embed_method)
        qf_bus = self.branch_from_to_bus(qf, method=embed_method)
        qt_bus = self.branch_to_to_bus(qt, method=embed_method)

        p_balance_violation, q_balance_violation = MOFACP.balance_residual_bus(
            pd_bus, qd_bus, pg_bus, qg_bus, vm, pf_bus, pt_bus, qf_bus, qt_bus, self.gs_bus, self.bs_bus
        )

        return (
            MOF.equality_violation(p_balance_violation, clamp=clamp),
            MOF.equality_violation(q_balance_violation, clamp=clamp),
        )

    def calc_violations(
        self,
        pd: Tensor,
        qd: Tensor,
        pg: Tensor,
        qg: Tensor,
        vm: Tensor,
        va: Optional[Tensor] = None,
        dva: Optional[Tensor] = None,
        flows: Optional[tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        reduction: Optional[str] = "mean",
        clamp: bool = True,
    ) -> dict[str, Tensor]:
        """Calculate the violation of all the constraints.

        The reduction is applied across the component dimension - e.g., 'mean' will do violation.mean(dim=1) where each violation is (batch, components)

        Args:
            pd (Tensor): Real power demand. (batch, loads)
            qd (Tensor): Reactive power demand. (batch, loads)
            pg (Tensor): Real power generation. (batch, gens)
            qg (Tensor): Reactive power generation. (batch, gens)
            vm (Tensor): Voltage magnitude. (batch, buses)
            va (Tensor, optional): Voltage angle. (batch, buses)
            dva (Tensor, optional): Voltage angle difference. (batch, branches)
            flows (tuple[Tensor, Tensor, Tensor, Tensor], optional): Power flows. (pf, pt, qf, qt)
            reduction (str, optional): Reduction method. Defaults to 'mean'. Must be one of 'mean', 'sum', 'none'.
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to True.

        Returns:
            - dict[str, Tensor]: Dictionary of violations.

            `vm_lower`: Voltage magnitude lower bound violation.

            `vm_upper`: Voltage magnitude upper bound violation.

            `pg_lower`: Real power generation lower bound violation.

            `pg_upper`: Real power generation upper bound violation.

            `qg_lower`: Reactive power generation lower bound violation.

            `qg_upper`: Reactive power generation upper bound violation.

            `thrm_1`: Thermal limit from violation.

            `thrm_2`: Thermal limit to violation.

            `p_balance`: Real power balance violation.

            `q_balance`: Reactive power balance violation.

            `dva_lower`: Voltage angle difference lower bound violation.

            `dva_upper`: Voltage angle difference upper bound violation.

        """
        vm_lower_violation, vm_upper_violation = self.vm_bound_residual(vm, clamp=clamp)
        pg_lower_violation, pg_upper_violation = self.pg_bound_residual(pg, clamp=clamp)
        qg_lower_violation, qg_upper_violation = self.qg_bound_residual(qg, clamp=clamp)
        if dva is None:
            dva = self.angle_difference(va)

        pf, pt, qf, qt = self.flows_from_voltage(vm, dva) if flows is None else flows

        thrm_1_violation, thrm_2_violation = self.thermal_residual(pf, pt, qf, qt, clamp=clamp)
        p_balance_violation, q_balance_violation = self.balance_residual(
            pd, qd, pg, qg, vm, pf, pt, qf, qt, clamp=clamp
        )

        dva_lower_violation, dva_upper_violation = self.dva_bound_residual(dva, clamp=clamp)

        violations = {
            "vm_lower": vm_lower_violation,
            "vm_upper": vm_upper_violation,
            "pg_lower": pg_lower_violation,
            "pg_upper": pg_upper_violation,
            "qg_lower": qg_lower_violation,
            "qg_upper": qg_upper_violation,
            "thrm_1": thrm_1_violation,
            "thrm_2": thrm_2_violation,
            "p_balance": p_balance_violation,
            "q_balance": q_balance_violation,
            "dva_lower": dva_lower_violation,
            "dva_upper": dva_upper_violation,
        }

        return self._reduce_violations(violations, reduction)

    @property
    def violation_shapes(self) -> dict[str, int]:
        return {
            "vm_lower": self.n_bus.item(),
            "vm_upper": self.n_bus.item(),
            "pg_lower": self.n_gen.item(),
            "pg_upper": self.n_gen.item(),
            "qg_lower": self.n_gen.item(),
            "qg_upper": self.n_gen.item(),
            "thrm_1": self.n_branch.item(),
            "thrm_2": self.n_branch.item(),
            "p_balance": self.n_bus.item(),
            "q_balance": self.n_bus.item(),
            "dva_lower": self.n_branch.item(),
            "dva_upper": self.n_branch.item(),
        }
