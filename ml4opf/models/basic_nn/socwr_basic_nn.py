import torch
from torch import Tensor

from ml4opf import SOCWRViolation, SOCWRModel
from ml4opf.layers import BoundRepair

from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet


class SOCWRBasicNN(BasicNN):
    opfmodel: SOCWRModel
    violation: SOCWRViolation

    def add_boundrepair(self, boundrepair: str):
        if boundrepair == "none" or boundrepair is None:
            return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = self.violation.pmin
        lower[self.qg_slice] = self.violation.qmin
        lower[self.w_slice] = self.violation.vmin_sq

        upper[self.pg_slice] = self.violation.pmax
        upper[self.qg_slice] = self.violation.qmax
        upper[self.w_slice] = self.violation.vmax_sq

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
    def w_slice(self):
        return self.opfmodel.slices[1]["primal/w"]

    @property
    def wr_slice(self):
        return self.opfmodel.slices[1]["primal/wr"]
    
    @property
    def wi_slice(self):
        return self.opfmodel.slices[1]["primal/wi"]

    def slice_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pd = x[:, self.pd_slice]
        qd = x[:, self.qd_slice]
        return pd, qd

    def slice_output(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pg = y[:, self.pg_slice]
        qg = y[:, self.qg_slice]
        w = y[:, self.w_slice]
        wr = y[:, self.wr_slice]
        wi = y[:, self.wi_slice]

        return pg, qg, w, wr, wi


class SOCWRBasicNeuralNet(BasicNeuralNet, SOCWRModel):
    model: SOCWRBasicNN

    def predict(self, pd: Tensor, qd: Tensor) -> dict[str, Tensor]:
        self.model.eval()

        if len(pd.shape) == 1:
            pd = pd.unsqueeze(0)
            qd = qd.unsqueeze(0)

        x = torch.cat([pd, qd], dim=1)

        y_hat = self.model.forward(x)

        pg, qg, w, wr, wi = self.model.slice_output(y_hat)

        ret: dict[str, Tensor] = dict()
        ret["pg"] = pg
        ret["qg"] = qg
        ret["w"] = w
        ret["wr"] = wr
        ret["wi"] = wi

        return ret
