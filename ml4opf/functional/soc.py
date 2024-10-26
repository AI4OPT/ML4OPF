"""SOCOPF Functional interface"""

import torch
from torch import Tensor


@torch.jit.script
def flows_from_voltage(
    w: Tensor,
    wr: Tensor,
    wi: Tensor,
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
    """Compute power flows from w, wr, wi, and branch info."""
    wf = w[:, fbus]
    wt = w[:, tbus]

    pf = (gff * wf) + (gft * wr) + (bft * wi)
    qf = (-bff * wf) - (bft * wr) + (gft * wi)
    pt = (gtt * wt) + (gtf * wr) - (btf * wi)
    qt = (-btt * wt) - (btf * wr) - (gtf * wi)

    return (pf, pt, qf, qt)


@torch.jit.script
def jabr_residual(
    w: Tensor,
    wr: Tensor,
    wi: Tensor,
    fbus: Tensor,
    tbus: Tensor,
):
    """Compute the Jabr constraint residual."""
    wf = w[:, fbus]
    wt = w[:, tbus]

    return torch.pow(wr, 2) + torch.pow(wi, 2) - (wf * wt)


@torch.jit.script
def balance_residual_bus(
    pd_bus: Tensor,
    qd_bus: Tensor,
    pg_bus: Tensor,
    qg_bus: Tensor,
    w: Tensor,
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
        w (Tensor): Voltage magnitude squared at each bus. (batch, n_bus)
        pf_bus (Tensor): Active power flow from each bus. (batch, n_bus)
        pt_bus (Tensor): Active power flow to each bus. (batch, n_bus)
        qf_bus (Tensor): Reactive power flow from each bus. (batch, n_bus)
        qt_bus (Tensor): Reactive power flow to each bus. (batch, n_bus)
        gs_bus (Tensor): Shunt conductance at each bus. (batch, n_bus)
        bs_bus (Tensor): Shunt susceptance at each bus. (batch, n_bus)

    Returns:
        tuple[Tensor]: Active and reactive power balance violation \( \text{p_viol}, \text{q_viol} \) each of shape (batch, n_bus).
    """

    p_balance_violation = pg_bus - pd_bus - pt_bus - pf_bus - gs_bus * w
    q_balance_violation = qg_bus - qd_bus - qt_bus - qf_bus + bs_bus * w

    return (p_balance_violation, q_balance_violation)


def voltage_phasor_bounds(
    E: int,
    vmin: Tensor,
    vmax: Tensor,
    dvamin: Tensor,
    dvamax: Tensor,
    bus_fr: Tensor,
    bus_to: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute the voltage phasor bounds.

    Args:
        E (int): Number of branches.
        vmin (Tensor): Minimum voltage magnitude at each bus. (n_bus)
        vmax (Tensor): Maximum voltage magnitude at each bus. (n_bus)
        dvamin (Tensor): Minimum voltage angle difference across each branch. (E)
        dvamax (Tensor): Maximum voltage angle difference across each branch. (E)
        bus_fr (Tensor): From bus of each branch. (E)
        bus_to (Tensor): To bus of each branch. (E)

    Returns:
        tuple[Tensor]: Minimum and maximum real and imaginary parts of the voltage phasor at each branch. \( \text{wr_min}, \text{wr_max}, \text{wi_min}, \text{wi_max} \) each of shape (E).
    """

    wr_min = torch.zeros(E, dtype=vmin.dtype, device=vmin.device)
    wr_max = torch.zeros(E, dtype=vmin.dtype, device=vmin.device)
    wi_min = torch.zeros(E, dtype=vmin.dtype, device=vmin.device)
    wi_max = torch.zeros(E, dtype=vmin.dtype, device=vmin.device)

    for e in range(E):
        wrmin, wrmax, wimin, wimax = _voltage_phasor_bounds(
            vmin[bus_fr[e]],
            vmax[bus_fr[e]],
            vmin[bus_to[e]],
            vmax[bus_to[e]],
            dvamin[e],
            dvamax[e],
        )
        wr_min[e] = wrmin
        wr_max[e] = wrmax
        wi_min[e] = wimin
        wi_max[e] = wimax

    return (wr_min, wr_max, wi_min, wi_max)


def _voltage_phasor_bounds(
    vfmin: Tensor, vfmax: Tensor, vtmin: Tensor, vtmax: Tensor, dvamin: Tensor, dvamax: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute the voltage phasor bounds.

    Args:
        vfmin (Tensor): Minimum voltage magnitude at the from bus. (1,)
        vfmax (Tensor): Maximum voltage magnitude at the from bus. (1,)
        vtmin (Tensor): Minimum voltage magnitude at the to bus. (1,)
        vtmax (Tensor): Maximum voltage magnitude at the to bus. (1,)
        dvamin (Tensor): Minimum voltage angle difference across the branch. (1,)
        dvamax (Tensor): Maximum voltage angle difference across the branch. (1,)

    Returns:
        tuple[Tensor]: Minimum and maximum real and imaginary parts of the voltage phasor. \( \text{wr_min}, \text{wr_max}, \text{wi_min}, \text{wi_max} \) each of shape (batch).
    """

    if torch.any(torch.abs(dvamin) > torch.tensor([torch.pi / 2])):
        raise ValueError("dvamin must be in [-π/2, π/2]")

    if torch.any(torch.abs(dvamax) > torch.tensor([torch.pi / 2])):
        raise ValueError("dvamax must be in [-π/2, π/2]")

    sinmin, cosmin = torch.sin(dvamin), torch.cos(dvamin)
    sinmax, cosmax = torch.sin(dvamax), torch.cos(dvamax)

    wmin = vfmin * vtmin
    wmax = vfmax * vtmax

    if dvamin >= 0:
        wr_min = wmin * cosmax
        wr_max = wmax * cosmin
        wi_min = wmin * sinmin
        wi_max = wmax * sinmax
    elif dvamax <= 0:
        wr_min = wmin * cosmin
        wr_max = wmax * cosmax
        wi_min = wmax * sinmin
        wi_max = wmin * sinmax
    else:
        wr_min = wmin * torch.min(cosmin, cosmax)
        wr_max = wmax * 1.0
        wi_min = wmax * sinmin
        wi_max = wmax * sinmax

    return (wr_min, wr_max, wi_min, wi_max)
