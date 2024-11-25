from typing import Optional
from abc import abstractmethod, ABC
from torch import Tensor

from ml4opf.formulations.soc.problem import SOCProblem
from ml4opf.formulations.soc.violation import SOCViolation
from ml4opf.formulations.model import OPFModel


class SOCModel(OPFModel, ABC):
    """`OPFModel` for SOCOPF"""

    problem: SOCProblem
    violation: SOCViolation

    @abstractmethod
    def predict(self, pd: Tensor, qd: Tensor) -> dict[str, Tensor]:
        """Predict the SOCOPF primal solution for a given set of loads.

        Args:
            pd (Tensor): Active power demand per load.
            qd (Tensor): Reactive power demand per load.

        Returns:
            dict[str, Tensor]: Dictionary containing the predicted primal solution.

                `pg`: Active power generation per generator or per bus.

                `qg`: Reactive power generation per generator or per bus.

                `w`: Squared voltage magnitude per bus.

                `wr`: Real part of the voltage phasor.

                `wi`: Imaginary part of the voltage phasor.
        """

    def evaluate_model(
        self, reduction: Optional[str] = None, inner_reduction: Optional[str] = None
    ) -> dict[str, Tensor]:
        test_pd = self.problem.test_data["input/pd"]
        test_qd = self.problem.test_data["input/qd"]

        test_pg = self.problem.test_data["primal/pg"]
        test_qg = self.problem.test_data["primal/qg"]
        test_w = self.problem.test_data["primal/w"]
        test_wr = self.problem.test_data["primal/wr"]
        test_wi = self.problem.test_data["primal/wi"]

        test_pf = self.problem.test_data["primal/pf"]
        test_qf = self.problem.test_data["primal/qf"]
        test_pt = self.problem.test_data["primal/pt"]
        test_qt = self.problem.test_data["primal/qt"]

        test_obj = self.violation.objective(test_pg)

        predictions = self.predict(test_pd, test_qd)

        pred_pg = predictions["pg"]
        pred_qg = predictions["qg"]
        pred_w = predictions["w"]
        pred_wr = predictions["wr"]
        pred_wi = predictions["wi"]

        pred_obj = self.violation.objective(pred_pg)

        flows = self.violation.flows_from_voltage(pred_w, pred_wr, pred_wi)

        violations = self.violation.calc_violations(
            pd=test_pd,
            qd=test_qd,
            pg=pred_pg,
            qg=pred_qg,
            w=pred_w,
            wr=pred_wr,
            wi=pred_wi,
            flows=flows,
            reduction=inner_reduction,
        )

        violations["pg_mae"] = SOCViolation.reduce_violation((pred_pg - test_pg).abs(), reduction=inner_reduction)
        violations["qg_mae"] = SOCViolation.reduce_violation((pred_qg - test_qg).abs(), reduction=inner_reduction)
        violations["w_mae"] = SOCViolation.reduce_violation((pred_w - test_w).abs(), reduction=inner_reduction)
        violations["wr_mae"] = SOCViolation.reduce_violation((pred_wr - test_wr).abs(), reduction=inner_reduction)
        violations["wi_mae"] = SOCViolation.reduce_violation((pred_wi - test_wi).abs(), reduction=inner_reduction)

        pred_pf, pred_pt, pred_qf, pred_qt = flows
        violations["pf_mae"] = SOCViolation.reduce_violation((pred_pf - test_pf).abs(), reduction=inner_reduction)
        violations["pt_mae"] = SOCViolation.reduce_violation((pred_pt - test_pt).abs(), reduction=inner_reduction)
        violations["qf_mae"] = SOCViolation.reduce_violation((pred_qf - test_qf).abs(), reduction=inner_reduction)
        violations["qt_mae"] = SOCViolation.reduce_violation((pred_qt - test_qt).abs(), reduction=inner_reduction)

        violations["obj_mape"] = ((pred_obj - test_obj) / test_obj).abs()

        return SOCViolation.reduce_violations(violations, reduction=reduction, dim=0)


class PerfectSOCModel(SOCModel):
    """Returns the ground truth, only works with test data."""

    def __init__(self, problem: SOCProblem):
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
            "w": self.problem.test_data["primal/w"],
            "wr": self.problem.test_data["primal/wr"],
            "wi": self.problem.test_data["primal/wi"],
        }

    def save_checkpoint(self, path_to_folder: str):
        raise NotImplementedError("Perfect model cannot be saved.")

    @staticmethod
    def load_from_checkpoint(path_to_folder: str, problem: SOCProblem):
        raise NotImplementedError("Perfect model cannot be loaded.")
