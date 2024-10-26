import pytest

from pathlib import Path

import torch
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.ac.problem import ACProblem
from ml4opf.formulations.ac.violation import ACViolation


def test_ac_violation():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = ACProblem(data_dir)
    v1 = p1.violation
    v1.to(torch.float32)

    viols = v1.calc_violations(
        pd=p1.train_data["input/pd"],
        qd=p1.train_data["input/qd"],
        pg=p1.train_data["primal/pg"],
        qg=p1.train_data["primal/qg"],
        vm=p1.train_data["primal/vm"],
        va=p1.train_data["primal/va"],
        flows=(
            p1.train_data["primal/pf"],
            p1.train_data["primal/pt"],
            p1.train_data["primal/qf"],
            p1.train_data["primal/qt"],
        ),
        reduction="none",
        # default clamp=True
    )

    for k, v in viols.items():
        assert v.shape[1] == v1.violation_shapes[k]

    ACViolation.reduce_violations(viols, reduction="mean", dim=1)
    ACViolation.reduce_violations(viols, reduction="mean", dim=0)

    for k, v in viols.items():
        assert v.item() < 1e-5, f"{k} violation is {v.item()}"

    for reduction in ["sum", "max", "mean", "none"]:
        viols = v1.calc_violations(
            pd=p1.train_data["input/pd"],
            qd=p1.train_data["input/qd"],
            pg=p1.train_data["primal/pg"],
            qg=p1.train_data["primal/qg"],
            vm=p1.train_data["primal/vm"],
            va=p1.train_data["primal/va"],
            flows=(
                p1.train_data["primal/pf"],
                p1.train_data["primal/pt"],
                p1.train_data["primal/qf"],
                p1.train_data["primal/qt"],
            ),
            reduction=reduction,
            # default clamp=True
        )

    with pytest.raises(ValueError):
        v1.calc_violations(
            pd=p1.train_data["input/pd"],
            qd=p1.train_data["input/qd"],
            pg=p1.train_data["primal/pg"],
            qg=p1.train_data["primal/qg"],
            vm=p1.train_data["primal/vm"],
            va=p1.train_data["primal/va"],
            flows=(
                p1.train_data["primal/pf"],
                p1.train_data["primal/pt"],
                p1.train_data["primal/qf"],
                p1.train_data["primal/qt"],
            ),
            reduction="invalid",
            # default clamp=True
        )

    fv = v1(
        pd=p1.train_data["input/pd"],
        qd=p1.train_data["input/qd"],
        pg=p1.train_data["primal/pg"],
        qg=p1.train_data["primal/qg"],
        vm=p1.train_data["primal/vm"],
        va=p1.train_data["primal/va"],
        flows=(
            p1.train_data["primal/pf"],
            p1.train_data["primal/pt"],
            p1.train_data["primal/qf"],
            p1.train_data["primal/qt"],
        ),
        reduction="mean",
        # default clamp=True
    )

    cv = v1.calc_violations(
        pd=p1.train_data["input/pd"],
        qd=p1.train_data["input/qd"],
        pg=p1.train_data["primal/pg"],
        qg=p1.train_data["primal/qg"],
        vm=p1.train_data["primal/vm"],
        va=p1.train_data["primal/va"],
        flows=(
            p1.train_data["primal/pf"],
            p1.train_data["primal/pt"],
            p1.train_data["primal/qf"],
            p1.train_data["primal/qt"],
        ),
        reduction="mean",
        # default clamp=True
    )

    dva = v1.angle_difference(p1.train_data["primal/va"])
    dv = v1.calc_violations(
        pd=p1.train_data["input/pd"],
        qd=p1.train_data["input/qd"],
        pg=p1.train_data["primal/pg"],
        qg=p1.train_data["primal/qg"],
        vm=p1.train_data["primal/vm"],
        dva=dva,
        flows=(
            p1.train_data["primal/pf"],
            p1.train_data["primal/pt"],
            p1.train_data["primal/qf"],
            p1.train_data["primal/qt"],
        ),
        reduction="mean",
        clamp=True,
    )
    assert fv.keys() == cv.keys()
    for k in fv.keys():
        assert (fv[k] == cv[k]).all()
        assert (fv[k] == dv[k]).all()

    v1.objective(p1.train_data["primal/pg"])
    pf, pt, qf, qt = v1.flows_from_voltage_bus(p1.train_data["primal/vm"], p1.train_data["primal/va"])
    assert (pf - p1.train_data["primal/pf"]).abs().mean() < 1e-4  # NOTE: relaxed tolerance
    assert (pt - p1.train_data["primal/pt"]).abs().mean() < 1e-4
    assert (qf - p1.train_data["primal/qf"]).abs().mean() < 1e-4
    assert (qt - p1.train_data["primal/qt"]).abs().mean() < 1e-4

    pd = p1.train_data["input/pd"]
    qd = p1.train_data["input/qd"]
    pg = p1.train_data["primal/pg"]
    qg = p1.train_data["primal/qg"]
    vm = p1.train_data["primal/vm"]
    assert torch.allclose(
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="pad")[0],
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="matrix")[0],
        atol=1e-5,
    )

    assert torch.allclose(
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="dense_matrix")[0],
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="matrix")[0],
        atol=1e-5,
    )

    assert torch.allclose(
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="pad")[1],
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="dense_matrix")[1],
        atol=1e-5,
    )

    assert torch.allclose(
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="dense_matrix")[1],
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="matrix")[1],
        atol=1e-5,
    )

    with pytest.raises(ValueError):
        v1.balance_residual(pd, qd, pg, qg, vm, pf, pt, qf, qt, embed_method="invalid")

    with pytest.raises(ValueError):
        v1.gen_to_bus(pg, "invalid")

    with pytest.raises(ValueError):
        v1.branch_from_to_bus(pf, "invalid")

    with pytest.raises(ValueError):
        v1.branch_to_to_bus(pt, "invalid")

    assert v1.branch_incidence.shape == (v1.n_bus, v1.n_branch)
    assert v1.branch_incidence_dense.shape == (v1.n_bus, v1.n_branch)
    assert v1.branch_incidence.shape == (v1.n_bus, v1.n_branch)
    assert v1.branch_incidence_dense.shape == (v1.n_bus, v1.n_branch)

    assert v1.adjacency_matrix.shape == (v1.n_bus, v1.n_bus)
    assert v1.adjacency_matrix_dense.shape == (v1.n_bus, v1.n_bus)
    assert v1.adjacency_matrix.shape == (v1.n_bus, v1.n_bus)
    assert v1.adjacency_matrix_dense.shape == (v1.n_bus, v1.n_bus)

    import ml4opf.functional.ac as MOFAC

    f1 = MOFAC.flows_from_voltage_bus(
        p1.train_data["primal/vm"],
        p1.train_data["primal/va"],
        v1.bus_fr,
        v1.bus_to,
        p1.case_data["gff"],
        p1.case_data["gft"],
        p1.case_data["gtf"],
        p1.case_data["gtt"],
        p1.case_data["bff"],
        p1.case_data["bft"],
        p1.case_data["btf"],
        p1.case_data["btt"],
    )

    f2 = MOFAC.flows_from_voltage_bus(
        p1.train_data["primal/vm"],
        None,
        v1.bus_fr,
        v1.bus_to,
        p1.case_data["gff"],
        p1.case_data["gft"],
        p1.case_data["gtf"],
        p1.case_data["gtt"],
        p1.case_data["bff"],
        p1.case_data["bft"],
        p1.case_data["btf"],
        p1.case_data["btt"],
        dva=v1.angle_difference(p1.train_data["primal/va"]),
    )

    assert set((len(f1), len(f2))) == {4}
    assert torch.allclose(f1[0], f2[0])
    assert torch.allclose(f1[1], f2[1])
    assert torch.allclose(f1[2], f2[2])
    assert torch.allclose(f1[3], f2[3])

    MOFAC.thermal_residual(
        p1.train_data["primal/pf"],
        p1.train_data["primal/pt"],
        p1.train_data["primal/qf"],
        p1.train_data["primal/qt"],
        p1.case_data["smax"],
    )
