""" Base class for DCOPF proxy models """

from typing import Optional, Any
from abc import ABC, abstractmethod
from torch import Tensor

from ml4opf.formulations.model import OPFModel

from ml4opf.formulations.dc.problem import DCProblem
from ml4opf.formulations.dc.violation import DCViolation


class DCModel(OPFModel, ABC):
    """`OPFModel` for DCOPF"""

    problem: DCProblem
    violation: DCViolation

    @abstractmethod
    def predict(self, pd: Tensor) -> dict[str, Tensor]:
        """Predict the DCOPF primal solution for a given set of loads.

        Args:
            pd (Tensor): Active power demand per load.

        Returns:
            dict[str, Tensor]: Dictionary containing the predicted primal solution.

                `pg`: Active power generation per generator or per bus.

                `va`: Voltage angle per bus.
        """

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

                `dva_lower`: Angle difference limit lower bound violation.

                `dva_upper`: Angle difference limit upper bound violation.

                `pf_lower`: Flow limit lower bound violation.

                `pf_upper`: Flow limit upper bound violation.

                `p_balance`: Power balance violation.

                `pg_mae`: Mean absolute error of the real power generation.

                `va_mae`: Mean absolute error of the voltage angle. (if not bus-wise and va not in predictions, skipped)

                `pf_mae`: Mean absolute error of the real power flow.

                `obj_mape`: Mean absolute percent error of the objective value.

        """
        test_pd = self.problem.test_data["input/pd"]

        test_pg = self.problem.test_data["primal/pg"]

        test_va = self.problem.test_data["primal/va"]
        test_pf = self.problem.test_data["primal/pf"]

        test_obj = self.violation.objective(test_pg)

        predictions = self.predict(test_pd)

        pred_pg = predictions["pg"]
        pred_va = predictions["va"]
        pred_pf = self.violation.pf_from_va(pred_va)

        pred_obj = self.violation.objective(pred_pg)

        violations = self.violation.calc_violations(test_pd, pred_pg, pred_va, pred_pf, reduction=inner_reduction)

        violations["pg_mae"] = (pred_pg - test_pg).abs().mean(dim=1)
        violations["va_mae"] = (pred_va - test_va).abs().mean(dim=1)
        violations["pf_mae"] = (pred_pf - test_pf).abs().mean(dim=1)
        violations["obj_mape"] = ((pred_obj - test_obj) / test_obj).abs()

        return DCViolation.reduce_violations(violations, reduction=reduction, dim=0)


class PerfectDCModel(DCModel):
    """Returns the ground truth, only works with test data."""

    def __init__(self, problem: DCProblem):
        self.problem = problem
        self.violation = problem.violation

    def predict(self, pd: Tensor, _: Any = None) -> dict[str, Tensor]:
        """Return the ground truth. Only works for `self.problem.test_data`."""
        assert (
            pd.shape == self.problem.test_data["input/pd"].shape and (pd == self.problem.test_data["input/pd"]).all()
        ), "Perfect model inference is only for test data."
        return {
            "pg": self.problem.test_data["primal/pg"],
            "va": self.problem.test_data["primal/va"],
        }

    def save_checkpoint(self, path_to_folder: str):
        raise NotImplementedError("Perfect model cannot to be saved.")

    @staticmethod
    def load_from_checkpoint(path_to_folder: str, problem: DCProblem):
        raise NotImplementedError("Perfect model cannot to be loaded.")
