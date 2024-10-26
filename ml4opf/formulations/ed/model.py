""" Base class for EconomicDispatch proxy models """

from typing import Optional, Any
from torch import Tensor

from ml4opf.formulations.model import OPFModel

from ml4opf.formulations.ed.problem import EDProblem
from ml4opf.formulations.ed.violation import EDViolation


class EDModel(OPFModel):
    """`OPFModel` for EconomicDispatch"""

    problem: EDProblem
    violation: EDViolation

    def evaluate_model(
        self, reduction: Optional[str] = None, inner_reduction: Optional[str] = None
    ) -> dict[str, Tensor]:
        test_pd = self.problem.test_data["input/pd"]

        test_pg = self.problem.test_data["primal/pg"]

        test_obj = self.violation.objective(test_pd, test_pg)

        predictions = self.predict(test_pd)

        pred_pg = predictions["pg"]

        pred_obj = self.violation.objective(test_pd, pred_pg)

        violations = self.violation.calc_violations(test_pd, pred_pg, reduction=inner_reduction)

        violations["pg_gap"] = (pred_pg - test_pg).abs().mean(dim=1)
        violations["obj_mape"] = ((pred_obj - test_obj) / test_obj).abs()

        return EDViolation.reduce_violations(violations, reduction=reduction, dim=0)


class PerfectEDModel(EDModel):
    """Returns the ground truth, only works with test data."""

    def __init__(self, problem: EDProblem):
        self.problem = problem
        self.violation = problem.violation

    def predict(self, pd: Tensor, _: Any = None) -> dict[str, Tensor]:
        """Return the ground truth. Only works for `self.problem.test_data`."""
        assert (
            pd.shape == self.problem.test_data["input/pd"].shape and (pd == self.problem.test_data["input/pd"]).all()
        ), "Perfect model inference is only for test data."
        return {"pg": self.problem.test_data["EDPowerModel/primal/pg"]}

    def save_checkpoint(self, path_to_folder: str):
        pass

    @staticmethod
    def load_from_checkpoint(path_to_folder: str, problem: EDProblem):
        pass
