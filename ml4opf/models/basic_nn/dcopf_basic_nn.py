import torch

from torch import Tensor

from ml4opf import DCViolation, DCModel
from ml4opf.layers import BoundRepair

from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet


class DCBasicNN(BasicNN):
    opfmodel: DCModel
    violation: DCViolation

    def add_boundrepair(self, boundrepair: str):
        if boundrepair == "none" or boundrepair is None:
            return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = self.violation.pgmin
        upper[self.pg_slice] = self.violation.pgmax

        self.layers.append(BoundRepair(lower, upper, boundrepair))

    @property
    def pd_slice(self):
        return self.opfmodel.slices[0]["input/pd"]

    @property
    def pg_slice(self):
        return self.opfmodel.slices[1]["primal/pg"]

    @property
    def va_slice(self):
        return self.opfmodel.slices[1]["primal/va"]

    def slice_output(self, y: Tensor) -> tuple[Tensor, Tensor]:
        pg = y[:, self.pg_slice]
        va = y[:, self.va_slice]
        return pg, va


class DCBasicNeuralNet(BasicNeuralNet, DCModel):
    model: DCBasicNN

    def predict(self, pd: Tensor) -> dict[str, Tensor]:
        self.model.eval()

        y_hat = self.model.forward(pd)

        pg, va = self.model.slice_output(y_hat)

        ret: dict[str, Tensor] = dict()
        ret["pg"] = pg
        ret["va"] = va

        return ret
