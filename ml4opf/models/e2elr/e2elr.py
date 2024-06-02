import torch
from torch import Tensor

from ml4opf.layers.bound_repair import BoundRepair
from ml4opf.models.basic_nn.ed_basic_nn import EDBasicNN, EDBasicNeuralNet
from ml4opf.layers.hypersimplex_repair import HyperSimplexRepair


class EDE2ELRNN(EDBasicNN):

    def add_boundrepair(self, boundrepair: str):
        if boundrepair == "none" or boundrepair is None:
            return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = self.violation.pmin
        upper[self.pg_slice] = self.violation.pmax

        self.layers.append(BoundRepair(lower, upper, boundrepair))
        self.pb_repair = HyperSimplexRepair(lower, upper)

    def forward(self, pd: torch.Tensor):
        pg = super().forward(pd)
        pg = self.pb_repair.forward(pg, pd.sum(dim=-1))
        return pg


class EDE2ELRNeuralNet(EDBasicNeuralNet):
    model: EDE2ELRNN
    model_cls = EDE2ELRNN

    def predict(self, pd: Tensor) -> dict[str, Tensor]:
        self.model.eval()

        y_hat = self.model.forward(pd)

        pg = self.model.slice_output(y_hat)

        ret: dict[str, Tensor] = dict()
        ret["pg"] = pg

        return ret
