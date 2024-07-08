import torch.nn as nn
from torch import Tensor

from typing import Optional, Union

from ml4opf import OPFViolation, OPFModel
from ml4opf.loss_functions.objective import ObjectiveLoss
from ml4opf.loss_functions.penalty import PenaltyLoss
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet
from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN


class PenaltyNN(BasicNN):
    violation: OPFViolation
    opfmodel: OPFModel

    def __init__(
        self,
        opfmodel: OPFModel,
        slices: list[slice],
        optimizer: str = "adam",
        loss: str = "mse",
        hidden_sizes: list[int] = [100, 100],
        activation: str = "relu",
        boundrepair: str = "none",
        learning_rate: float = 1e-3,
        exclude_keys: Optional[Union[str, list[str]]] = None,
        multipliers: Optional[Union[float, dict[str, float]]] = None,
    ):
        super().__init__(opfmodel, slices, optimizer, loss, hidden_sizes, activation, boundrepair, learning_rate)

        self.loss = PenaltyLoss(self.violation, exclude_keys, multipliers)

    def set_loss(self, loss):
        loss = loss.lower()
        if loss == "mse":
            self.base_loss = nn.MSELoss(reduction="none")
        elif loss == "l1":
            self.base_loss = nn.L1Loss(reduction="none")
        elif loss == "obj":
            self.base_loss = ObjectiveLoss(self.violation)
        else:
            raise ValueError(f"Loss function {loss} not supported.")

class PenaltyNeuralNet(BasicNeuralNet):
    model: PenaltyNN

    def __init__(self, config, penalty_config, problem):
        super().__init__(dict(**config, **penalty_config), problem)
