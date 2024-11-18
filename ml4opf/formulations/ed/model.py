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
        """Evaluate the model on the test data.

        Args:
            reduction (str, optional): Reduction method for the metrics. Defaults to None. Must be one of "mean", "sum","max", "none".
                                        If specified, each value in the returned dictionary will be a scalar. Otherwise, they are arrays of shape (n_test_samples,)

            inner_reduction (str, optional): Reduction method for turning metrics calculated per component to per sample. Defaults to None. Must be one of "mean", "sum","max", "none".


        Returns:
            dict[str, Tensor]: Dictionary containing Tensor metrics of the model's performance.

                `pg_lower`: Generator lower bound violation.

                `pg_upper`: Generator upper bound violation.

                `pf_lower`: Branch power flow lower bound violation.

                `pf_upper`: Branch power flow upper bound violation.

                `p_balance`: Power balance violation.

                `pg_mae`: Mean absolute error of the real power generation.

                `obj_mape`: Mean absolute percent error of the objective value.

        """
        test_pd = self.problem.test_data["input/pd"]

        test_pg = self.problem.test_data["primal/pg"]

        test_obj = self.violation.objective(test_pd, test_pg)

        predictions = self.predict(test_pd)

        pred_pg = predictions["pg"]

        pred_obj = self.violation.objective(test_pd, pred_pg)

        violations = self.violation.calc_violations(test_pd, pred_pg, reduction=inner_reduction)

        violations["pg_mae"] = (pred_pg - test_pg).abs().mean(dim=1)
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
