from typing import Optional
import torch
from torch import Tensor

import ml4opf.functional.acp as MOFACP


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

    \[ \text{p_viol} = \text{pg_bus} - \text{pd_bus} - \text{pt_bus} - \text{pf_bus} - \text{gs_bus} \times \text{w} \]
    \[ \text{q_viol} = \text{qg_bus} - \text{qd_bus} - \text{qt_bus} - \text{qf_bus} + \text{bs_bus} \times \text{w} \]

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


@torch.jit.script
def _flows_from_voltage(
    w: Tensor,
    wr: Tensor,
    wi: Tensor,
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
    r"""Compute the power flows given w, wr, and wi.

    The branch (transformer + line) parameters \( g, b, g_f, b_f, g_t, b_t, t_r, t_i \) and \( \text{tap} \)
    are assumed to be constant *for the life of this object*, matching the reference case.
    These constants are pre-computed using the `MOFACP._compute_flow_constants` function.

    \[ \mathbf{p}_f = \frac{g + g_{f}}{\text{tap}^2} \mathbf{w}_{f} + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{wr}_{ft} + \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]
    \[ \mathbf{p}_t = (g + g_{t}) \mathbf{v}_{t}^2 + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{wr}_{ft} + \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]
    \[ \mathbf{q}_f = \frac{-(b + b_{f})}{\text{tap}^2} \mathbf{w}_{f} - \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{wr}_{f} + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]
    \[ \mathbf{q}_t = -(b + b_{t}) \mathbf{v}_{t}^2 - \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{wr}_{ft} + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]

    Args:
        w (Tensor): Squared voltage magnitude per bus. (batch_size, nbus)
        wr (Tensor): Real part of the voltage phasor. (batch_size, nbranch)
        wi (Tensor): Imaginary part of the voltage phasor. (batch_size, nbranch)
        fbus (Tensor): From bus indices. (batch_size, nbranch)
        tbus (Tensor): To bus indices. (batch_size, nbranch)
        c* (Tensor): Constants for the power flow calculation -- see `MOFACP._compute_flow_constants`.

    Returns:
        Tensor: Real power flow per branch ( \(\mathbf{p}_f\) ). (batch_size, nbranch)
        Tensor: Real power flow per branch ( \(\mathbf{p}_t\) ). (batch_size, nbranch)
        Tensor: Reactive power flow per branch ( \(\mathbf{q}_f\) ). (batch_size, nbranch)
        Tensor: Reactive power flow per branch ( \(\mathbf{q}_t\) ). (batch_size, nbranch)
    """
    w_fr = w[:, fbus]
    w_to = w[:, tbus]

    pf = (c1 * w_fr) + (c3 * wr) + (c4 * wi)

    qf = (c2 * w_fr) - (c4 * wr) + (c3 * wi)

    pt = (c7 * w_to) + (c9 * wr) - (c10 * wi) # TODO: verify correctness of socwr flows!

    qt = (c8 * w_to) - (c10 * wr) - (c9 * wi)

    return (pf, pt, qf, qt)


def flows_from_voltage(
    w: Tensor,
    wr: Tensor,
    wi: Tensor,
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

    \[ \mathbf{p}_f = \frac{g + g_{f}}{\text{tap}^2} \mathbf{w}_{f} + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{wr}_{ft} + \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]
    \[ \mathbf{p}_t = (g + g_{t}) \mathbf{v}_{t}^2 + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{wr}_{ft} + \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]
    \[ \mathbf{q}_f = \frac{-(b + b_{f})}{\text{tap}^2} \mathbf{w}_{f} - \frac{-b t_r - g t_i}{\text{tap}^2} \mathbf{wr}_{f} + \frac{-g t_r + b t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]
    \[ \mathbf{q}_t = -(b + b_{t}) \mathbf{v}_{t}^2 - \frac{-b t_r + g t_i}{\text{tap}^2} \mathbf{wr}_{ft} + \frac{-g t_r - b t_i}{\text{tap}^2} \mathbf{wi}_{ft} \]

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

    c = MOFACP._compute_flow_constants(tap, shift, r, x, g_fr, g_to, b_fr, b_to)

    return _flows_from_voltage(w, wr, wi, fbus, tbus, *c)
