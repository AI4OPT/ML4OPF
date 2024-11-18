""" Base class for ACOPF proxy models """

from typing import Optional
from abc import abstractmethod, ABC

from torch import Tensor

from ml4opf.formulations.ac.problem import ACProblem
from ml4opf.formulations.ac.violation import ACViolation
from ml4opf.formulations.model import OPFModel


class ACModel(OPFModel, ABC):
    """`OPFModel` for ACOPF"""

    problem: ACProblem
    violation: ACViolation

    @abstractmethod
    def predict(self, pd: Tensor, qd: Tensor) -> dict[str, Tensor]:
        """Predict the ACOPF primal solution for a given set of loads.

        Args:
            pd (Tensor): Active power demand per load.
            qd (Tensor): Reactive power demand per load.

        Returns:
            dict[str, Tensor]: Dictionary containing the predicted primal solution.

                `pg`: Active power generation per generator or per bus.

                `qg`: Reactive power generation per generator or per bus.

                `vm`: Voltage magnitude per bus.

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

                `vm_lower`: Lower bound on the voltage magnitude.

                `vm_upper`: Upper bound on the voltage magnitude.

                `pg_lower`: Lower bound on the real power generation.

                `pg_upper`: Upper bound on the real power generation.

                `qg_lower`: Lower bound on the reactive power generation.

                `qg_upper`: Upper bound on the reactive power generation.

                `thrm_1`: Thermal limit violation from

                `thrm_2`: Thermal limit violation to

                `p_balance`: Active power balance violation.

                `q_balance`: Reactive power balance violation.

                `pg_mae`: Mean absolute error of the real power generation.

                `qg_mae`: Mean absolute error of the reactive power generation.

                `vm_mae`: Mean absolute error of the voltage magnitude.

                `va_mae`: Mean absolute error of the voltage angle. (if not bus-wise and va not in predictions, skipped)

                `dva_mae`: Mean absolute error of the angle difference. (only if not bus-wise)

                `obj_mape`: Mean absolute percent error of the objective value.

        """
        test_pd = self.problem.test_data["input/pd"]
        test_qd = self.problem.test_data["input/qd"]

        test_pg = self.problem.test_data["primal/pg"]
        test_qg = self.problem.test_data["primal/qg"]
        test_vm = self.problem.test_data["primal/vm"]
        test_va = self.problem.test_data["primal/va"]

        test_pf = self.problem.test_data["primal/pf"]
        test_qf = self.problem.test_data["primal/qf"]
        test_pt = self.problem.test_data["primal/pt"]
        test_qt = self.problem.test_data["primal/qt"]

        test_obj = self.violation.objective(test_pg)
        test_dva = self.violation.angle_difference(test_va)

        predictions = self.predict(test_pd, test_qd)

        pred_pg = predictions["pg"]
        pred_qg = predictions["qg"]
        pred_vm = predictions["vm"]
        pred_va = predictions["va"]

        pred_obj = self.violation.objective(pred_pg)
        pred_dva = self.violation.angle_difference(pred_va)

        flows = self.violation.flows_from_voltage(pred_vm, pred_dva)

        violations = self.violation.calc_violations(
            pd=test_pd,
            qd=test_qd,
            pg=pred_pg,
            qg=pred_qg,
            vm=pred_vm,
            va=pred_va,
            flows=flows,
            reduction=inner_reduction,
        )

        violations["pg_mae"] = ACViolation.reduce_violation((pred_pg - test_pg).abs(), reduction=inner_reduction)
        violations["qg_mae"] = ACViolation.reduce_violation((pred_qg - test_qg).abs(), reduction=inner_reduction)
        violations["vm_mae"] = ACViolation.reduce_violation((pred_vm - test_vm).abs(), reduction=inner_reduction)
        violations["va_mae"] = ACViolation.reduce_violation((pred_va - test_va).abs(), reduction=inner_reduction)
        violations["dva_mae"] = ACViolation.reduce_violation((pred_dva - test_dva).abs(), reduction=inner_reduction)

        pred_pf, pred_pt, pred_qf, pred_qt = flows
        violations["pf_mae"] = ACViolation.reduce_violation((pred_pf - test_pf).abs(), reduction=inner_reduction)
        violations["pt_mae"] = ACViolation.reduce_violation((pred_pt - test_pt).abs(), reduction=inner_reduction)
        violations["qf_mae"] = ACViolation.reduce_violation((pred_qf - test_qf).abs(), reduction=inner_reduction)
        violations["qt_mae"] = ACViolation.reduce_violation((pred_qt - test_qt).abs(), reduction=inner_reduction)

        violations["obj_mape"] = ((pred_obj - test_obj) / test_obj).abs()

        return ACViolation.reduce_violations(violations, reduction=reduction, dim=0)


class PerfectACPModel(ACModel):
    """Returns the ground truth, only works with test data."""

    def __init__(self, problem: ACProblem):
        self.problem = problem
        self.violation = problem.violation

    def predict(self, pd: Tensor, qd: Tensor) -> dict[str, Tensor]:
        """Return the ground truth. Only works for `self.problem.test_data`."""
        assert (
            (pd.shape == self.problem.test_data["input/pd"].shape)
            and (qd.shape == self.problem.test_data["input/qd"].shape)
            and (pd == self.problem.test_data["input/pd"]).all()
            and (qd == self.problem.test_data["input/qd"]).all()
        ), "Perfect model inference is only for test data."
        return {
            "pg": self.problem.test_data["primal/pg"],
            "qg": self.problem.test_data["primal/qg"],
            "vm": self.problem.test_data["primal/vm"],
            "va": self.problem.test_data["primal/va"],
        }

    def save_checkpoint(self, path_to_folder: str):
        raise NotImplementedError("Perfect model cannot be saved.")

    @staticmethod
    def load_from_checkpoint(path_to_folder: str, problem: ACProblem):
        raise NotImplementedError("Perfect model cannot be loaded.")
