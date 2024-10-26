import torch

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.dc.problem import DCProblem


def test_dc_problem():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = DCProblem(data_dir)

    assert hasattr(p1, "train_data")
    assert hasattr(p1, "test_data")

    assert p1.train_data.keys() == p1.test_data.keys()

    assert p1.train_data["input/pd"].shape == (74, 35)
    assert p1.test_data["input/pd"].shape == (19, 35)

    ds1, s1 = p1.make_dataset(test_set=True)
    assert s1[0].keys() == {"input/pd"}
    assert s1[1].keys() == {"primal/pg", "primal/va"}

    assert isinstance(p1.violation, torch.nn.Module)
