import torch

from torch import Tensor

from ml4opf.formulations.ed import EDViolation, EDModel
from ml4opf.layers import BoundRepair

from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet


class EDBasicNN(BasicNN):
    opfmodel: EDModel
    violation: EDViolation

    def add_boundrepair(self, boundrepair: str):
        if boundrepair == "none" or boundrepair is None:
            return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = self.violation.pmin
        upper[self.pg_slice] = self.violation.pmax

        self.layers.append(BoundRepair(lower, upper, boundrepair))

    @property
    def pd_slice(self):
        return self.opfmodel.slices[0]["input/pd"]

    @property
    def pg_slice(self):
        return self.opfmodel.slices[1]["primal/pg"]

    def slice_output(self, y: Tensor) -> tuple[Tensor, Tensor]:
        pg = y[:, self.pg_slice]
        return pg


class EDBasicNeuralNet(BasicNeuralNet, EDModel):
    model: EDBasicNN

    def predict(self, pd: Tensor) -> dict[str, Tensor]:
        self.model.eval()

        y_hat = self.model.forward(pd)

        pg = self.model.slice_output(y_hat)

        ret: dict[str, Tensor] = dict()
        ret["pg"] = pg

        return ret
