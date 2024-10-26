from pathlib import Path

import torch
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.ac.problem import ACProblem

from ml4opf.layers.voltagedifference_repair import VoltageDifferenceRepair


def test_vdiff_repair():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = ACProblem(data_dir)
    v1 = p1.violation

    dva = v1.angle_difference(p1.train_data["primal/va"])

    vdr = VoltageDifferenceRepair(v1.branch_incidence, slackbus_idx=v1.ref_bus)
    vdr1 = VoltageDifferenceRepair(v1.branch_incidence)
    lsva = vdr(dva)
    lsva1 = vdr1(dva)
    assert torch.allclose(v1.angle_difference(lsva), dva, atol=1e-5)
    assert torch.allclose(v1.angle_difference(lsva1), dva, atol=1e-5)

    dva.requires_grad_(True)
    lsva2 = vdr(dva)
    lsva2.sum().backward()
    assert dva.grad.abs().sum() > 0
