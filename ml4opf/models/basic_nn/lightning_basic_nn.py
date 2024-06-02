import pytorch_lightning as pl, torch, torch.nn as nn, math

from typing import Optional
from functools import partial
from abc import abstractmethod

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor

from ml4opf import OPFViolation, warn, OPFModel


class BasicNN(pl.LightningModule):
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
        weight_init_seed: int = 42,
    ):
        super().__init__()

        self.opfmodel = opfmodel
        self.violation = opfmodel.violation
        self.save_hyperparameters(ignore=["opfmodel"])

        self.slices = slices
        assert len(slices) == 2, "Got len(slices) != 2. First slice should correspond to the inputs, second slice to the outputs. See BasicNN.make_dataset."
        self.hidden_sizes = hidden_sizes

        self.set_activation(activation)
        self.set_loss(loss)
        self.make_network()
        self.add_boundrepair(boundrepair)
        self.make_optimizer(optimizer, learning_rate)
        self.init_weights(seed=weight_init_seed) # see nn.Linear.reset_parameters and pytorch#57109
        self.to(torch.float32)

    @abstractmethod
    def add_boundrepair(self, boundrepair: str):
        pass

    @property
    def input_size(self) -> int:  # assumes slices[0] is input
        return max(self.slices[0][k].stop for k in self.slices[0])

    @property
    def output_size(self) -> int:  # assumes slices[1] is output
        return max(self.slices[1][k].stop for k in self.slices[1])

    def set_activation(self, activation: str):
        activation = activation.lower()
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            raise ValueError(f"Activation function {activation} not supported.")

    def set_loss(self, loss: str):
        loss = loss.lower()
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "l1":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Loss function {loss} not supported.")
    
    def init_weights(self, weight_initializer=None, seed: Optional[int] = 42):
        if weight_initializer is None:
            weight_initializer = partial(nn.init.kaiming_uniform_, a=math.sqrt(5))

        rng = torch.Generator().manual_seed(seed)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # following nn.Linear.reset_parameters, but with generator
                weight_initializer(layer.weight, generator=rng)
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias, -bound, bound, generator=rng)

    def make_optimizer(self, optimizer: str, learning_rate: float):
        optimizer = optimizer.lower()
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), learning_rate)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not recognized.")


    def make_network(self):
        self.layers = nn.Sequential()

        self.layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        self.layers.append(self.activation())

        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.layers.append(self.activation())

        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

    def forward(self, x: Tensor):
        return self.layers.forward(x)

    def configure_optimizers(self):
        return self.optimizer

    def _step(self, batch: tuple[Tensor, Tensor], stage: str = ""):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], _):
        return self._step(batch, "val")
