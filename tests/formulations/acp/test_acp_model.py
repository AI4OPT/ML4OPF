import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.acp.problem import ACPProblem

from ml4opf.formulations.acp.model import PerfectACPModel


def test_acp_model():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    p1 = ACPProblem(data_dir, "300_ieee", "ACOPF", test_set_size=10)

    model = PerfectACPModel(p1)
    res = model.evaluate_model(reduction="mean", inner_reduction="mean")

    for k, v in res.items():
        assert v.item() < 1e-5, f"{k} violation is {v.item()}"
