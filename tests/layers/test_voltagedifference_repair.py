from pathlib import Path

import torch
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.acp.problem import ACPProblem

from ml4opf.layers.voltagedifference_repair import VoltageDifferenceRepair


def test_vdiff_repair():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    p1 = ACPProblem(data_dir, "300_ieee", "ACOPF", make_test_set=False)
    v1 = p1.violation

    dva = v1.angle_difference(p1.train_data["primal/va"])

    vdr = VoltageDifferenceRepair(v1.branch_incidence, slackbus_idx=v1.slackbus_idx)
    vdr1 = VoltageDifferenceRepair(v1.branch_incidence)
    lsva = vdr(dva)
    lsva1 = vdr1(dva)
    assert (lsva - p1.train_data["primal/va"]).abs().mean() < 1e-5

    dva.requires_grad_(True)
    lsva2 = vdr(dva)
    lsva2.sum().backward()
    assert dva.grad.abs().sum() > 0
