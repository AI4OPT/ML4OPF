import pytest

import torch
from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.dcp.problem import DCPProblem
from ml4opf.formulations.dcp.violation import DCPViolation


def test_dcp_violation():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    p1 = DCPProblem(data_dir, "300_ieee", "DCOPF", make_test_set=False)
    v1 = p1.violation

    viols = v1.calc_violations(
        pd=p1.train_data["input/pd"],
        pg=p1.train_data["primal/pg"],
        pf=p1.train_data["primal/pf"],
        va=p1.train_data["primal/va"],
        reduction="none",
        # default clamp=True
    )

    for k, v in viols.items():
        assert v.shape[1] == v1.violation_shapes[k]

    DCPViolation._reduce_violations(viols, reduction="mean", dim=1)
    DCPViolation._reduce_violations(viols, reduction="mean", dim=0)

    for k, v in viols.items():
        assert v.item() < 1e-6, f"{k} violation is {v.item()}"

    for reduction in ["sum", "max", "mean", "none"]:
        v1.calc_violations(
            pd=p1.train_data["input/pd"],
            pg=p1.train_data["primal/pg"],
            pf=p1.train_data["primal/pf"],
            va=p1.train_data["primal/va"],
            reduction=reduction,
            # default clamp=True
        )

    with pytest.raises(ValueError):
        v1.calc_violations(
            pd=p1.train_data["input/pd"],
            pg=p1.train_data["primal/pg"],
            pf=p1.train_data["primal/pf"],
            va=p1.train_data["primal/va"],
            reduction="invalid",
            # default clamp=True
        )

    fv = v1(
        pd=p1.train_data["input/pd"],
        pg=p1.train_data["primal/pg"],
        pf=p1.train_data["primal/pf"],
        va=p1.train_data["primal/va"],
        reduction="mean",
        clamp=False,
    )
    cv = v1.calc_violations(
        pd=p1.train_data["input/pd"],
        pg=p1.train_data["primal/pg"],
        pf=p1.train_data["primal/pf"],
        va=p1.train_data["primal/va"],
        reduction="mean",
        clamp=False,
    )

    assert fv.keys() == cv.keys()
    for k in fv.keys():
        assert (fv[k] == cv[k]).all()

    v1.objective(p1.train_data["primal/pg"])
    pf = v1.pf_from_va(p1.train_data["primal/va"])
    assert (pf - p1.train_data["primal/pf"]).abs().mean() < 1e-6
