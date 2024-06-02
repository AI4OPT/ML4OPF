""" Abstract base class for ML models for OPF. 

Each formulation should inherit from `OPFModel` and implement the following methods:

- `evaluate_model`: Evaluate the model on the test data.

User models should inherit from the formulation's `OPFModel` class (e.g., `ACPModel`) and implement the following methods:

- `save_checkpoint`: Save the model's checkpoint to a file.

- `load_from_checkpoint`: Load the model's checkpoint from a file.

- `predict`: Predict the solution for a given set of inputs.
"""

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional

from ml4opf.formulations.problem import OPFProblem


class OPFModel(ABC):
    """An abstract base class for ACOPF models."""

    @abstractmethod
    def save_checkpoint(self, path_to_folder: str):
        """Save the model's checkpoint to a file.

        Args:
            path (str): Path to save the checkpoint.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_checkpoint(self, path_to_folder: str, problem: OPFProblem):
        """Load the model's checkpoint from a file.

        Args:
            path (str): Path to load the checkpoint from.
        """
        pass

    @abstractmethod
    def predict(self, *inputs: Tensor) -> dict[str, Tensor]:
        """Predict the solution for a given set of inputs.

        Args:
            *inputs (Tensor): Input tensors to the model.

        Returns:
            dict[str, Tensor]: Dictionary containing the solution.
        """
        pass

    @abstractmethod
    def evaluate_model(
        self, reduction: Optional[str] = None, inner_reduction: Optional[str] = None
    ) -> dict[str, Tensor]:
        """Evaluate the model on the test data.

        Args:
            reduction (str, optional): Reduction method for the metrics. Defaults to None. Must be one of "mean", "sum", "none".
                                         If specified, each value in the returned dictionary will be a scalar. Otherwise, they are arrays of shape (n_test_samples,)

        Returns:
            dict[str, Tensor]: Dictionary containing Tensor metrics of the model's performance.
        """
        pass
