import pytest

import torch

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.acp.problem import ACPProblem


def test_acp_problem():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    p1 = ACPProblem(data_dir, "300_ieee", "ACOPF", test_set_size=10)

    assert hasattr(p1, "train_data")
    assert hasattr(p1, "test_data")

    assert p1.train_data.keys() == p1.test_data.keys()
    assert p1.test_data.keys() == {
        "dual/lam_kirchhoff_active",
        "dual/lam_kirchhoff_reactive",
        "dual/lam_ohm_active_fr",
        "dual/lam_ohm_active_to",
        "dual/lam_ohm_reactive_fr",
        "dual/lam_ohm_reactive_to",
        "dual/lam_slack_bus",
        "dual/mu_pg_lb",
        "dual/mu_pg_ub",
        "dual/mu_qg_lb",
        "dual/mu_qg_ub",
        "dual/mu_sm_fr",
        "dual/mu_sm_to",
        "dual/mu_va_diff",
        "dual/mu_vm_lb",
        "dual/mu_vm_ub",
        "input/br_status",
        "input/meta/config",
        "input/meta/seed",
        "input/pd",
        "input/qd",
        "meta/config",
        "meta/dual_objective_value",
        "meta/dual_status",
        "meta/primal_objective_value",
        "meta/primal_status",
        "meta/seed",
        "meta/solve_time",
        "meta/termination_status",
        "primal/pf",
        "primal/pg",
        "primal/pt",
        "primal/qf",
        "primal/qg",
        "primal/qt",
        "primal/va",
        "primal/vm",
    }

    assert p1.train_data["input/pd"].shape == (24, 201)
    assert p1.test_data["input/pd"].shape == (10, 201)
    assert isinstance(p1.violation, torch.nn.Module)
    del p1

    p2 = ACPProblem(data_dir, "300_ieee", "ACOPF", test_set_size=2, total_load_range=(200, 210))
    assert p2.train_data["input/pd"].shape == (8, 201)
    del p2

    p3 = ACPProblem(data_dir, "300_ieee", "ACOPF", test_set_size=20, feasible_only=False)
    assert p3.train_data["primal/pg"].shape == (80, 69)
    ds3, s3 = p3.make_dataset(test_set=True)
    assert s3[0].keys() == {"input/pd", "input/qd"}
    assert s3[1].keys() == {"primal/pg", "primal/qg", "primal/vm", "primal/va"}

    p4 = ACPProblem(data_dir, "300_ieee", "ACOPF", make_test_set=False)
    assert p4.test_data is None
    del p4

    p5 = ACPProblem(data_dir, "300_ieee", "ACOPF", convert_to_float32=False, make_test_set=False)
    assert p5.train_data["input/pd"].dtype == torch.float64

    ds, slices = p5.make_dataset()
    assert len(ds) == 34
    assert len(slices) == 2
    assert slices[0].keys() == {"input/pd", "input/qd"}
    assert slices[1].keys() == {"primal/pg", "primal/qg", "primal/vm", "primal/va"}
    iter_ds = iter(ds)
    assert len(next(iter_ds)) == 2
    sliced = p5.slice_batch(next(iter_ds), slices)
    assert len(sliced) == 2
    assert sliced[0].keys() == slices[0].keys()
    assert sliced[1].keys() == slices[1].keys()

    del p5

    p6 = ACPProblem(
        data_dir,
        "300_ieee",
        "ACOPF",
        feasible_only={
            "meta/primal_status": "FEASIBLE_POINT",
        },
        make_test_set=False,
        total_load_range=(None, None),
    )
    assert p6.train_data["primal/pg"].shape == (34, 69)
