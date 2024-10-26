import torch
from torch import Tensor

from ml4opf import ACViolation, ACModel
from ml4opf.layers import BoundRepair

from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet


class ACBasicNN(BasicNN):
    opfmodel: ACModel
    violation: ACViolation

    def add_boundrepair(self, boundrepair: str):
        if boundrepair == "none" or boundrepair is None:
            return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = self.violation.pgmin
        lower[self.qg_slice] = self.violation.qgmin
        lower[self.vm_slice] = self.violation.vmin

        upper[self.pg_slice] = self.violation.pgmax
        upper[self.qg_slice] = self.violation.qgmax
        upper[self.vm_slice] = self.violation.vmax

        self.layers.append(BoundRepair(lower, upper, boundrepair))

    @property
    def pd_slice(self):
        return self.opfmodel.slices[0]["input/pd"]

    @property
    def qd_slice(self):
        return self.opfmodel.slices[0]["input/qd"]

    @property
    def pg_slice(self):
        return self.opfmodel.slices[1]["primal/pg"]

    @property
    def qg_slice(self):
        return self.opfmodel.slices[1]["primal/qg"]

    @property
    def vm_slice(self):
        return self.opfmodel.slices[1]["primal/vm"]

    @property
    def va_slice(self):
        return self.opfmodel.slices[1]["primal/va"]

    def slice_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pd = x[:, self.pd_slice]
        qd = x[:, self.qd_slice]
        return pd, qd

    def slice_output(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pg = y[:, self.pg_slice]
        qg = y[:, self.qg_slice]
        vm = y[:, self.vm_slice]
        va = y[:, self.va_slice]
        return pg, qg, vm, va


class ACBasicNeuralNet(BasicNeuralNet, ACModel):
    model: ACBasicNN

    def predict(self, pd: Tensor, qd: Tensor) -> dict[str, Tensor]:
        self.model.eval()

        if len(pd.shape) == 1:
            pd = pd.unsqueeze(0)
            qd = qd.unsqueeze(0)

        x = torch.cat([pd, qd], dim=1)

        y_hat = self.model.forward(x)

        pg, qg, vm, va = self.model.slice_output(y_hat)

        ret: dict[str, Tensor] = dict()
        ret["pg"] = pg
        ret["qg"] = qg
        ret["vm"] = vm
        ret["va"] = va

        return ret
