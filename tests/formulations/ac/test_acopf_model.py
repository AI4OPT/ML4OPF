import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.ac.problem import ACProblem

from ml4opf.formulations.ac.model import PerfectACPModel


def test_ac_model():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = ACProblem(data_dir, convert_to_float32=False)

    model = PerfectACPModel(p1)
    res = model.evaluate_model(reduction="mean", inner_reduction="mean")

    for k, v in res.items():
        assert v.item() < 2e-5, f"{k} violation is {v.item()}"
