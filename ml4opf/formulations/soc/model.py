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
        raise NotImplementedError("TODO: implement generic evaluate_model for SOC.")


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
