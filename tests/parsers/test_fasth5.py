import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.parsers.read_hdf5 import parse_hdf5


def test_fasth5():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    D = parse_hdf5(data_dir / "300_ieee_ACOPF.h5")
    assert D.keys() == {
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
    Dt = parse_hdf5(data_dir / "300_ieee_DCOPF.h5", preserve_shape=True)
    assert Dt.keys() == {"dual", "meta", "primal"}
    assert Dt["primal"].keys() == {"pf", "pg", "va"}
