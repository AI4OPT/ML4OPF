"""ACOPF Functional interface"""

from typing import Optional
import torch
from torch import Tensor

from ml4opf.functional import angle_difference


@torch.jit.script
def _thermal_residual_sq(
    pf: Tensor,
    pt: Tensor,
    qf: Tensor,
    qt: Tensor,
    smaxsq: Tensor,
) -> tuple[Tensor, Tensor]:
    smaxsq_ = smaxsq.expand_as(pf)
    smaxsq_ = smaxsq.expand_as(pt)

    thrm_1 = torch.pow(pf, 2) + torch.pow(qf, 2) - smaxsq_
    thrm_2 = torch.pow(pt, 2) + torch.pow(qt, 2) - smaxsq_

    return (thrm_1, thrm_2)


def thermal_residual(
    pf: Tensor,
    pt: Tensor,
    qf: Tensor,
    qt: Tensor,
    smax: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Calculate the thermal residual for a set of flows.

    \[ \text{thrm}_1 = \text{pf}^2 + \text{qf}^2 - \text{smax}^2 \]
    \[ \text{thrm}_2 = \text{pt}^2 + \text{qt}^2 - \text{smax}^2 \]

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
    smaxsq = torch.pow(smax, 2)

    return _thermal_residual_sq(pf, pt, qf, qt, smaxsq)


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
def objective(pg: Tensor, c0: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
    r"""Compute ACOPF objective function.

    Args:
        pg (Tensor): Active power generation. (batch, n_gen)
        cost (Tensor): Cost coefficients. (n_gen, 3)

    Returns:
        Tensor: Objective function value for each sample. (batch,)
    """
    return (c0 + c1 * pg + c2 * torch.pow(pg, 2)).sum(dim=1)


@torch.jit.script
def flows_from_voltage(
    vm: Tensor,
    dva: Tensor,
    fbus: Tensor,
    tbus: Tensor,
    gff: Tensor,
    gft: Tensor,
    gtf: Tensor,
    gtt: Tensor,
    bff: Tensor,
    bft: Tensor,
    btf: Tensor,
    btt: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    vm_fr = vm[:, fbus]
    vm_to = vm[:, tbus]

    wf = torch.pow(vm_fr, 2)
    wt = torch.pow(vm_to, 2)

    cosdva = torch.cos(dva)
    sindva = torch.sin(dva)

    vm_frto = vm_fr * vm_to

    wr = vm_frto * cosdva
    wi = vm_frto * sindva

    pf = (gff * wf) + (gft * wr) + (bft * wi)
    qf = (-bff * wf) - (bft * wr) + (gft * wi)
    pt = (gtt * wt) + (gtf * wr) - (btf * wi)
    qt = (-btt * wt) - (btf * wr) - (gtf * wi)

    return (pf, pt, qf, qt)


def flows_from_voltage_bus(
    vm: Tensor,
    va: Tensor,
    fbus: Tensor,
    tbus: Tensor,
    gff: Tensor,
    gft: Tensor,
    gtf: Tensor,
    gtt: Tensor,
    bff: Tensor,
    bft: Tensor,
    btf: Tensor,
    btt: Tensor,
    dva: Optional[Tensor] = None,
):
    """Compute power flows from voltage magnitude, angle, and branch info."""
    if dva is None:
        dva = angle_difference(va, fbus, tbus)

    return flows_from_voltage(vm, dva, fbus, tbus, gff, gft, gtf, gtt, bff, bft, btf, btt)
