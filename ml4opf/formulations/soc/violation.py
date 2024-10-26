"""Class interface for SOCOPF constraints, objective, etc."""

from typing import Optional
from torch import Tensor

import ml4opf.functional as MOF
import ml4opf.functional.ac as MOFAC
import ml4opf.functional.soc as MOFSOC
from ml4opf.formulations.violation import OPFViolation


class SOCViolation(OPFViolation):
    """`OPFViolation` for SOCOPF"""

    def __init__(self, data: dict):
        super().__init__(data)

        self.register_buffer("wmin", self.vmin.pow(2), persistent=False)
        self.register_buffer("wmax", self.vmax.pow(2), persistent=False)

        wrmin, wrmax, wimin, wimax = MOFSOC.voltage_phasor_bounds(
            self.E, self.vmin, self.vmax, self.dvamin, self.dvamax, self.bus_fr, self.bus_to
        )

        self.register_buffer("wrmin", wrmin, persistent=False)
        self.register_buffer("wrmax", wrmax, persistent=False)
        self.register_buffer("wimin", wimin, persistent=False)
        self.register_buffer("wimax", wimax, persistent=False)

        self.register_buffer("tandvamin", self.dvamin.tan(), persistent=False)
        self.register_buffer("tandvamax", self.dvamax.tan(), persistent=False)

    def objective(self, pg: Tensor) -> Tensor:
        r"""Compute the objective function given the active power generation per generator.

        Args:
            pg (Tensor): Active power generation per generator. (batch_size, ngen)

        Returns:
            Tensor: Objective function value. (batch_size)
        """
        return MOFAC.objective(pg, self.c0, self.c1, self.c2)

    def flows_from_voltage(self, w: Tensor, wr: Tensor, wi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Compute the power flows.

        Returns:
            Tensor: Real power flow per branch ( \(\mathbf{p}_f\) ). (batch_size, nbranch)
            Tensor: Real power flow per branch ( \(\mathbf{p}_t\) ). (batch_size, nbranch)
            Tensor: Reactive power flow per branch ( \(\mathbf{q}_f\) ). (batch_size, nbranch)
            Tensor: Reactive power flow per branch ( \(\mathbf{q}_t\) ). (batch_size, nbranch)
        """

        return MOFSOC.flows_from_voltage(
            w,
            wr,
            wi,
            self.bus_fr,
            self.bus_to,
            self.gff,
            self.gft,
            self.gtf,
            self.gtt,
            self.bff,
            self.bft,
            self.btf,
            self.btt,
        )

    def angle_difference_residual(self, wr: Tensor, wi: Tensor, clamp: bool = False) -> Tensor:
        """Compute the angle difference bound residual."""
        return self.clamped_bound_residual(wi, self.tandvamin * wr, self.tandvamax * wr, clamp=clamp)

    def w_bound_residual(self, w: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the bound residual of w.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbranch)
            Tensor: Upper bound residual. (batch_size, nbranch)
        """
        return self.clamped_bound_residual(w, self.wmin, self.wmax, clamp=clamp)

    def wr_bound_residual(self, wr: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the bound residual of wr.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbranch)
            Tensor: Upper bound residual. (batch_size, nbranch)
        """
        return self.clamped_bound_residual(wr, self.wrmin, self.wrmax, clamp=clamp)

    def wi_bound_residual(self, wr: Tensor, clamp: bool = False) -> tuple[Tensor, Tensor]:
        r"""Calculate the bound residual of wi.

        Returns:
            Tensor: Lower bound residual. (batch_size, nbranch)
            Tensor: Upper bound residual. (batch_size, nbranch)
        """
        return self.clamped_bound_residual(wr, self.wimin, self.wimax, clamp=clamp)

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
        return self.clamped_bound_residual(pg, self.pgmin, self.pgmax, clamp=clamp)

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
        return self.clamped_bound_residual(qg, self.qgmin, self.qgmax, clamp=clamp)

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
        thrm_1, thrm_2 = MOFAC.thermal_residual(pf, pt, qf, qt, self.smax)

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
        w: Tensor,
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

        p_balance_violation, q_balance_violation = MOFSOC.balance_residual_bus(
            pd_bus, qd_bus, pg_bus, qg_bus, w, pf_bus, pt_bus, qf_bus, qt_bus, self.gs, self.bs
        )

        return (
            MOF.equality_violation(p_balance_violation, clamp=clamp),
            MOF.equality_violation(q_balance_violation, clamp=clamp),
        )

    def jabr_residual(self, w: Tensor, wr: Tensor, wi: Tensor, clamp: bool = False) -> Tensor:
        r"""Compute the Jabr constraint residual.

        \[ g_{\text{jabr}} = \text{wr}^2 + \text{wi}^2 - \text{w}_{fr} * \text{w}_{to} \]

        Args:
            w (Tensor): Squared voltage magnitude per bus. (batch_size, nbus)
            wr (Tensor): Real part of the voltage phasor. (batch_size, nbus)
            wi (Tensor): Imaginary part of the voltage phasor. (batch_size, nbus)
            clamp (bool, optional): Clamp the residual to be non-negative (extract violations). Defaults to False.

        Returns:
            Tensor: Jabr constraint residual. (batch_size, nbus)
        """
        g_jabr = MOFSOC.jabr_residual(w, wr, wi, self.bus_fr, self.bus_to)
        return MOF.inequality_violation(g_jabr, clamp=clamp)

    def calc_violations(
        self,
        pd: Tensor,
        qd: Tensor,
        pg: Tensor,
        qg: Tensor,
        w: Tensor,
        wr: Tensor,
        wi: Tensor,
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
        # vm_lower_violation, vm_upper_violation = self.vm_bound_residual(vm, clamp=clamp)
        wr_lower_violation, wr_upper_violation = self.wr_bound_residual(wr, clamp=clamp)
        wi_lower_violation, wi_upper_violation = self.wi_bound_residual(wi, clamp=clamp)
        w_lower_violation, w_upper_violation = self.w_bound_residual(w, clamp=clamp)
        pg_lower_violation, pg_upper_violation = self.pg_bound_residual(pg, clamp=clamp)
        qg_lower_violation, qg_upper_violation = self.qg_bound_residual(qg, clamp=clamp)

        jabr_violation = self.jabr_residual(w, wr, wi, clamp=clamp)
        pf, pt, qf, qt = self.flows_from_voltage(w, wr, wi) if flows is None else flows

        thrm_1_violation, thrm_2_violation = self.thermal_residual(pf, pt, qf, qt, clamp=clamp)
        p_balance_violation, q_balance_violation = self.balance_residual(pd, qd, pg, qg, w, pf, pt, qf, qt, clamp=clamp)

        dva_lower_violation, dva_upper_violation = self.angle_difference_residual(wr, wi, clamp=clamp)

        violations = {
            "wr_lower": wr_lower_violation,
            "wr_upper": wr_upper_violation,
            "wi_lower": wi_lower_violation,
            "wi_upper": wi_upper_violation,
            "w_lower": w_lower_violation,
            "w_upper": w_upper_violation,
            "pg_lower": pg_lower_violation,
            "pg_upper": pg_upper_violation,
            "qg_lower": qg_lower_violation,
            "qg_upper": qg_upper_violation,
            "thrm_1": thrm_1_violation,
            "thrm_2": thrm_2_violation,
            "p_balance": p_balance_violation,
            "q_balance": q_balance_violation,
            "jabr": jabr_violation,
            "dva_lower": dva_lower_violation,
            "dva_upper": dva_upper_violation,
        }

        return self.reduce_violations(violations, reduction)

    @property
    def violation_shapes(self) -> dict[str, int]:
        return {
            "w_lower": self.n_bus.item(),
            "w_upper": self.n_bus.item(),
            "wr_lower": self.n_branch.item(),
            "wr_upper": self.n_branch.item(),
            "wi_lower": self.n_branch.item(),
            "wi_upper": self.n_branch.item(),
            "pg_lower": self.n_gen.item(),
            "pg_upper": self.n_gen.item(),
            "qg_lower": self.n_gen.item(),
            "qg_upper": self.n_gen.item(),
            "thrm_1": self.n_branch.item(),
            "thrm_2": self.n_branch.item(),
            "p_balance": self.n_bus.item(),
            "q_balance": self.n_bus.item(),
            "jabr": self.n_branch.item(),
            "dva_lower": self.n_branch.item(),
            "dva_upper": self.n_branch.item(),
        }
