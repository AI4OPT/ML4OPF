import pytest

import torch
from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf import ACProblem, DCProblem

from ml4opf.loss_functions.objective import ObjectiveLoss


def test_objective_loss():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"
    p1 = DCProblem(data_dir)
    v1 = p1.violation

    p2 = ACProblem(data_dir)
    v2 = p2.violation

    for reduction in ["sum", "max", "mean", "none"]:
        rf = {"sum": torch.sum, "max": torch.max, "mean": torch.mean, "none": lambda x: x}[reduction]
        ol = ObjectiveLoss(v1, reduction=reduction)
        assert (ol.forward(pg=p1.train_data["primal/pg"]) == rf(v1.objective(p1.train_data["primal/pg"]))).all()

        ol = ObjectiveLoss(v2, reduction=reduction)
        assert (ol.forward(p2.train_data["primal/pg"]) == rf(v2.objective(p2.train_data["primal/pg"]))).all()

    with pytest.raises(AssertionError):

        class InvalidViolation:
            pass

        iv = InvalidViolation()
        ol = ObjectiveLoss(iv, reduction="mean")

    with pytest.raises(AssertionError):
        ol = ObjectiveLoss(v1, reduction="invalid")

    ol = ObjectiveLoss(v1, reduction="mean")
    ol.reduction = "invalid"

    with pytest.raises(ValueError):
        ol.forward(pg=p1.train_data["primal/pg"])
