import torch

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.dcp.problem import DCPProblem


def test_dcp_problem():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    p1 = DCPProblem(data_dir, "300_ieee", "DCOPF", test_set_size=10)

    assert hasattr(p1, "train_data")
    assert hasattr(p1, "test_data")

    assert p1.train_data.keys() == p1.test_data.keys()
    assert p1.test_data.keys() == {
        "dual/lam_kirchhoff",
        "dual/lam_ohm",
        "dual/lam_slack_bus",
        "dual/mu_pg_lb",
        "dual/mu_pg_ub",
        "dual/mu_sm_lb",
        "dual/mu_sm_ub",
        "dual/mu_va_diff",
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
        "primal/va",
    }

    assert p1.train_data["input/pd"].shape == (53, 201)
    assert p1.test_data["input/pd"].shape == (10, 201)

    ds1, s1 = p1.make_dataset(test_set=True)
    assert s1[0].keys() == {"input/pd"}
    assert s1[1].keys() == {"primal/pg", "primal/va"}

    assert isinstance(p1.violation, torch.nn.Module)
