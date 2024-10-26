import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.dc.problem import DCProblem

from ml4opf.formulations.dc.model import PerfectDCModel


def test_dc_model():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = DCProblem(data_dir)

    model = PerfectDCModel(p1)
    res = model.evaluate_model(reduction="mean", inner_reduction="mean")

    for k, v in res.items():
        assert v.item() < 1e-5, f"{k} violation is {v.item()}"
