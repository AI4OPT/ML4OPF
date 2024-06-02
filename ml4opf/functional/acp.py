from typing import Optional
import torch
from torch import Tensor

from ml4opf.functional import angle_difference


@torch.jit.script
def _thermal_residual_sq(
    pf: Tensor, pt: Tensor, qf: Tensor, qt: Tensor, s1max_sq: Tensor, s2max_sq: Tensor
) -> tuple[Tensor, Tensor]:
    s1maxsq = s1max_sq.expand_as(pf)
    s2maxsq = s2max_sq.expand_as(pt)

    thrm_1 = torch.pow(pf, 2) + torch.pow(qf, 2) - s1maxsq
    thrm_2 = torch.pow(pt, 2) + torch.pow(qt, 2) - s2maxsq

    return (thrm_1, thrm_2)


def thermal_residual(
    pf: Tensor, pt: Tensor, qf: Tensor, qt: Tensor, s1max: Tensor, s2max: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Calculate the thermal residual for a set of flows.

    \[ \text{thrm}_1 = \text{pf}^2 + \text{qf}^2 - \text{s1max}^2 \]
    \[ \text{thrm}_2 = \text{pt}^2 + \text{qt}^2 - \text{s2max}^2 \]

    Args:
        pf (Tensor): From power flow. (batch, n_branch)
        pt (Tensor): To power flow. (batch, n_branch)
        qf (Tensor): From reactive power flow. (batch, n_branch)
        qt (Tensor): To reactive power flow. (batch, n_branch)
        s1max (Tensor): From power flow limit. (n_branch,)
        s2max (Tensor): To power flow limit. (n_branch,)

    Returns:
        tuple[Tensor]: Thermal residuals \( (\text{thrm}_1, \text{thrm}_2) \) each of shape (batch, n_branch).

    """
    s1maxsq = torch.pow(s1max, 2)
    s2maxsq = torch.pow(s2max, 2)

    return _thermal_residual_sq(pf, pt, qf, qt, s1maxsq, s2maxsq)


@torch.jit.script
def balance_residual_bus(
    pd_bus: Tensor,
    qd_bus: Tensor,
    pg_bus: Tensor,
    qg_bus: Tensor,
    vm: Tensor,
    pf_bus: Tensor,
    pt_bus: Tensor,
    qf_bus: Tensor,
    qt_bus: Tensor,
    gs_bus: Tensor,
    bs_bus: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute power balance residual.

    \[ \text{p_viol} = \text{pg_bus} - \text{pd_bus} - \text{pt_bus} - \text{pf_bus} - \text{gs_bus} \times \text{vm}^2 \]
    \[ \text{q_viol} = \text{qg_bus} - \text{qd_bus} - \text{qt_bus} - \text{qf_bus} + \text{bs_bus} \times \text{vm}^2 \]

    Args:
        pd_bus (Tensor): Active power demand at each bus. (batch, n_bus)
        qd_bus (Tensor): Reactive power demand at each bus. (batch, n_bus)
        pg_bus (Tensor): Active power generation at each bus. (batch, n_bus)
        qg_bus (Tensor): Reactive power generation at each bus. (batch, n_bus)
        vm (Tensor): Voltage magnitude at each bus. (batch, n_bus)
        pf_bus (Tensor): Active power flow from each bus. (batch, n_bus)
        pt_bus (Tensor): Active power flow to each bus. (batch, n_bus)
        qf_bus (Tensor): Reactive power flow from each bus. (batch, n_bus)
        qt_bus (Tensor): Reactive power flow to each bus. (batch, n_bus)
        gs_bus (Tensor): Shunt conductance at each bus. (batch, n_bus)
        bs_bus (Tensor): Shunt susceptance at each bus. (batch, n_bus)

    Returns:
        tuple[Tensor]: Active and reactive power balance violation \( \text{p_viol}, \text{q_viol} \) each of shape (batch, n_bus).
    """
    vm2 = torch.pow(vm, 2)

    p_balance_violation = pg_bus - pd_bus - pt_bus - pf_bus - gs_bus * vm2
    q_balance_violation = qg_bus - qd_bus - qt_bus - qf_bus + bs_bus * vm2

    return (p_balance_violation, q_balance_violation)

@torch.jit.script
def objective(pg: Tensor, cost: Tensor) -> Tensor:
    r"""Compute ACOPF objective function.

    \[ \text{objective} = \sum_i \left( \text{cost}_{2,i} + \text{cost}_{1,i} \times \text{pg}_i + \text{cost}_{0,i} \times \text{pg}_i^2 \right) \]

    Args:
        pg (Tensor): Active power generation. (batch, n_gen)
        cost (Tensor): Cost coefficients. (n_gen, 3)

    Returns:
        Tensor: Objective function value for each sample. (batch,)
    """
    return (cost[:, 2] + cost[:, 1] * pg + cost[:, 0] * torch.pow(pg, 2)).sum(dim=1)


@torch.jit.script
def _compute_flow_constants(
    tap: Tensor,
    shift: Tensor,
    r: Tensor,
    x: Tensor,
    g_fr: Tensor,
    g_to: Tensor,
    b_fr: Tensor,
    b_to: Tensor,
):
    r"""Compute the constants for power flow calculation.

    \[ t_r = \text{tap} \times \cos(\text{shift}) \]
    \[ t_i = \text{tap} \times \sin(\text{shift}) \]
    \[ g = \frac{\text{r}}{\text{r}^2 + \text{x}^2} \]
    \[ b = -\frac{\text{x}}{\text{r}^2 + \text{x}^2} \]
    \[ c_1 = \frac{g + \text{g_{fr}}}{\text{tap}^2} \]
    \[ c_2 = \frac{-(b + \text{b_{fr}})}{\text{tap}^2} \]
    \[ c_3 = \frac{-g t_r + b t_i}{\text{tap}^2} \]
    \[ c_4 = \frac{-b t_r - g t_i}{\text{tap}^2} \]
    \[ c_7 = g + \text{g_{to}} \]
    \[ c_8 = -(b + \text{b_{to}}) \]
    \[ c_9 = \frac{-g t_r - b t_i}{\text{tap}^2} \]
    \[ c_{10} = \frac{-b t_r + g t_i}{\text{tap}^2} \]

    Args:
        tap (Tensor): Transformer tap ratio. (batch, n_branch)
        shift (Tensor): Transformer phase shift. (batch, n_branch)
        r (Tensor): Branch resistance. (batch, n_branch)
        x (Tensor): Branch reactance. (batch, n_branch)
        g_fr (Tensor): From bus shunt conductance. (batch, n_branch)
        g_to (Tensor): To bus shunt conductance. (batch, n_branch)
        b_fr (Tensor): From bus shunt susceptance. (batch, n_branch)
        b_to (Tensor): To bus shunt susceptance. (batch, n_branch)

    Returns:
        tuple[Tensor]: Constants for power flow calculation \( c_1, c_2, c_3, c_4, c_7, c_8, c_9, c_{10} \) each of shape (batch, n_branch).
    """
    tr = tap * torch.cos(shift)
    ti = tap * torch.sin(shift)
    r2x2 = torch.pow(r, 2) + torch.pow(x, 2)
    g = r / (r2x2)
    b = -x / (r2x2)
    tap2 = torch.pow(tap, 2)  # torch.pow(tr, 2) + torch.pow(ti, 2)
    c1 = (g + g_fr) / tap2
    c2 = -(b + b_fr) / tap2
    c7 = g + g_to
    c8 = -(b + b_to)
    c3 = (-g * tr + b * ti) / tap2
    c4 = (-b * tr - g * ti) / tap2
    c9 = (-g * tr - b * ti) / tap2
    c10 = (-b * tr + g * ti) / tap2

    return c1, c2, c3, c4, c7, c8, c9, c10


@torch.jit.script
def _flows_from_voltage(
    vm: Tensor,
    dva: Tensor,
    fbus: Tensor,
    tbus: Tensor,
    c1: Tensor,
    c2: Tensor,
    c3: Tensor,
    c4: Tensor,
    c7: Tensor,
    c8: Tensor,
    c9: Tensor,
    c10: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute power flows from voltage magnitude and angle.

    \[ \mathbf{p^f} = c_1 \mathbf{v_{fr}}^2 + c_3 \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{fr} - \theta_{to}) + c_4 \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{fr} - \theta_{to}) \]
    \[ \mathbf{q^f} = c_2 \mathbf{v_{fr}}^2 - c_4 \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{fr} - \theta_{to}) + c_3 \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{fr} - \theta_{to}) \]
    \[ \mathbf{p^t} = c_7 \mathbf{v_{to}}^2 + c_9 \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{to} - \theta_{fr}) + c_{10} \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{to} - \theta_{fr}) \]
    \[ \mathbf{q^t} = c_8 \mathbf{v_{to}}^2 - c_{10} \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{to} - \theta_{fr}) + c_9 \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{to} - \theta_{fr}) \]

    Args:
        vm (Tensor): Voltage magnitudes. (batch, n_bus)
        dva (Tensor, optional): Voltage angle differences. (batch, n_branch)
        fbus (Tensor): From bus indices. (batch, n_branch)
        tbus (Tensor): To bus indices. (batch, n_branch)
        c (tuple[Tensor]): Constants for power flow calculation.

    Returns:
        tuple[Tensor]: Power flows \( \text{pf}, \text{pt}, \text{qf}, \text{qt} \), each of shape (batch, n_branch)
    """
    vm_fr = vm[:, fbus]
    vm_to = vm[:, tbus]

    vm_fr2 = torch.pow(vm_fr, 2)
    vm_to2 = torch.pow(vm_to, 2)

    cosdva = torch.cos(dva)
    sindva = torch.sin(dva)

    vm_frto = vm_fr * vm_to
    vm_frto_cos = vm_frto * cosdva
    vm_frto_sin = vm_frto * sindva

    pf = (c1 * vm_fr2) + (c3 * vm_frto_cos) + (c4 * vm_frto_sin)

    qf = (c2 * vm_fr2) - (c4 * vm_frto_cos) + (c3 * vm_frto_sin)

    pt = (c7 * vm_to2) + (c9 * vm_frto_cos) + (c10 * -vm_frto_sin)

    qt = (c8 * vm_to2) - (c10 * vm_frto_cos) + (c9 * -vm_frto_sin)

    return (pf, pt, qf, qt)


def flows_from_voltage(
    vm: Tensor,
    va: Tensor,
    fbus: Tensor,
    tbus: Tensor,
    tap: Tensor,
    shift: Tensor,
    r: Tensor,
    x: Tensor,
    g_fr: Tensor,
    g_to: Tensor,
    b_fr: Tensor,
    b_to: Tensor,
    dva: Optional[Tensor] = None,
):
    r"""Compute power flows from voltage magnitude and angle.

    \[ \mathbf{p^f} = \frac{g + g_{fr}}{\text{tap}^2} \mathbf{v_{fr}}^2 + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{fr} - \theta_{to}) + \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{fr} - \theta_{to}) \]
    \[ \mathbf{p^t} = (g + g_{to}) \mathbf{v_{to}}^2 + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{to} - \theta_{fr}) + \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{to} - \theta_{fr}) \]
    \[ \mathbf{q^f} = \frac{-(b + b_{fr})}{\text{tap}^2} \mathbf{v_{fr}}^2 - \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{fr} - \theta_{to}) + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{fr} - \theta_{to}) \]
    \[ \mathbf{q^t} = -(b + b_{to}) \mathbf{v_{to}}^2 - \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \cos(\theta_{to} - \theta_{fr}) + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{v_{fr}} \mathbf{v_{to}} \sin(\theta_{to} - \theta_{fr}) \]

    where \( g = \frac{r}{r^2 + x^2} \), and \( b = -\frac{x}{r^2 + x^2} \), \( t_r = \text{tap} \times \cos(\text{shift}) \), and \( t_i = \text{tap} \times \sin(\text{shift}) \)

    Args:
        vm (Tensor): Voltage magnitudes. (batch, n_bus)
        va (Tensor): Voltage angles. (batch, n_bus)
        fbus (Tensor): From bus indices. (batch, n_branch)
        tbus (Tensor): To bus indices. (batch, n_branch)
        tap (Tensor): Transformer tap ratio. (batch, n_branch)
        shift (Tensor): Transformer phase shift. (batch, n_branch)
        r (Tensor): Branch resistance. (batch, n_branch)
        x (Tensor): Branch reactance. (batch, n_branch)
        g_fr (Tensor): From bus shunt conductance. (batch, n_branch)
        g_to (Tensor): To bus shunt conductance. (batch, n_branch)
        b_fr (Tensor): From bus shunt susceptance. (batch, n_branch)
        b_to (Tensor): To bus shunt susceptance. (batch, n_branch)
        dva (Tensor, optional): Voltage angle differences -- if provided, `va` is ignored.

    Returns:
        tuple[Tensor]: Power flows \( \text{pf}, \text{pt}, \text{qf}, \text{qt} \), each of shape (batch, n_branch)
    """
    if dva is None:
        dva = angle_difference(va, fbus, tbus)

    c = _compute_flow_constants(tap, shift, r, x, g_fr, g_to, b_fr, b_to)

    return _flows_from_voltage(vm, dva, fbus, tbus, *c)
