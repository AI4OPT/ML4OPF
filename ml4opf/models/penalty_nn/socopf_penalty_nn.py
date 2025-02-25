from torch import Tensor

from ml4opf.models.penalty_nn.penalty_nn import PenaltyNN, PenaltyNeuralNet
from ml4opf.models.basic_nn.socopf_basic_nn import SOCBasicNN, SOCBasicNeuralNet
from ml4opf.loss_functions import ObjectiveLoss
from ml4opf import SOCViolation, SOCModel


class SOCPenaltyNN(PenaltyNN, SOCBasicNN):
    violation: SOCViolation
    opfmodel: SOCModel

    def _step(self, batch: tuple[Tensor, Tensor], stage: str = ""):
        x, y = batch
        y_hat = self(x)

        if isinstance(self.base_loss, ObjectiveLoss):
            # uses formulation specific `OPFViolation.objective`
            base_loss = self.base_loss(pg=y_hat[:, self.pg_slice])
        else:
            base_loss = self.base_loss(y_hat, y)

        pd = x[:, self.pd_slice]
        qd = x[:, self.qd_slice]

        pg_pred = y_hat[:, self.pg_slice]
        qg_pred = y_hat[:, self.qg_slice]
        w_pred = y_hat[:, self.w_slice]
        wr_pred = y_hat[:, self.wr_slice]
        wi_pred = y_hat[:, self.wi_slice]

        loss = self.loss.forward(
            base_loss,
            pd=pd,
            qd=qd,
            pg=pg_pred,
            qg=qg_pred,
            w=w_pred,
            wr=wr_pred,
            wi=wi_pred,
        )

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        return loss


class SOCPenaltyNeuralNet(PenaltyNeuralNet, SOCBasicNeuralNet):
    model: SOCPenaltyNN
