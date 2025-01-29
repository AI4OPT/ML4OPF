import pytest

import torch

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.ac.problem import ACProblem


def test_ac_problem():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = ACProblem(data_dir)

    assert hasattr(p1, "train_data")
    assert hasattr(p1, "test_data")

    assert p1.train_data.keys() == p1.test_data.keys()

    assert p1.train_data["input/pd"].shape == (75, 35)
    assert p1.test_data["input/pd"].shape == (19, 35)
    assert isinstance(p1.violation, torch.nn.Module)

    ds, slices = p1.make_dataset()
    assert len(ds) == 75
    assert len(slices) == 2
    assert slices[0].keys() == {"input/pd", "input/qd"}
    assert slices[1].keys() == {"primal/pg", "primal/qg", "primal/vm", "primal/va"}
    iter_ds = iter(ds)
    assert len(next(iter_ds)) == 2
    sliced = p1.slice_batch(next(iter_ds), slices)
    assert len(sliced) == 2
    assert sliced[0].keys() == slices[0].keys()
    assert sliced[1].keys() == slices[1].keys()

    del p1
