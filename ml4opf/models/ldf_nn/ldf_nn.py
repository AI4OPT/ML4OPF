import torch.nn as nn

from ml4opf.formulations import OPFViolation, OPFModel
from ml4opf.loss_functions import LDFLoss, ObjectiveLoss
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet
from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN


class LDFNN(BasicNN):
    """Base class for LDF containing formulation-agnostic methods."""
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
        step_size: float = 1e-5,
        kickin: int = 0,
        update_freq: int = 500,
        divide_by_counter: bool = True,
        exclude_keys: list[str] = [],
        weight_init_seed: int = 42,
    ):
        super().__init__(opfmodel, slices, optimizer, loss, hidden_sizes, activation, boundrepair, learning_rate, weight_init_seed)

        self.loss = LDFLoss(
            self.violation,
            step_size=step_size,
            kickin=kickin,
            update_freq=update_freq,
            divide_by_counter=divide_by_counter,
            exclude_keys=exclude_keys,
        )

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

    def on_train_epoch_start(self):
        self.loss.start_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        self.loss.end_epoch()

class LDFNeuralNet(BasicNeuralNet):
    model: LDFNN
