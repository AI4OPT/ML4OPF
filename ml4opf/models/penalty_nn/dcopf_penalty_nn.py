from torch import Tensor

from ml4opf.models.penalty_nn.penalty_nn import PenaltyNN, PenaltyNeuralNet
from ml4opf.models.basic_nn.dcopf_basic_nn import DCBasicNN, DCBasicNeuralNet
from ml4opf.loss_functions import ObjectiveLoss
from ml4opf import DCViolation, DCModel


class DCPenaltyNN(PenaltyNN, DCBasicNN):
    violation: DCViolation
    opfmodel: DCModel

    def _step(self, batch: tuple[Tensor, Tensor], stage: str = ""):
        x, y = batch
        y_hat = self(x)

        if isinstance(self.base_loss, ObjectiveLoss):
            # uses formulation specific `OPFViolation.objective`
            base_loss = self.base_loss(pg=y_hat[:, self.pg_slice])
        else:
            base_loss = self.base_loss(y_hat, y)

        pd = x[:, self.pd_slice]

        pg_pred = y_hat[:, self.pg_slice]
        va_pred = y_hat[:, self.va_slice]

        loss = self.loss.forward(
            base_loss,
            pd=pd,
            pg=pg_pred,
            va=va_pred,
        )

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        return loss


class DCPenaltyNeuralNet(PenaltyNeuralNet, DCBasicNeuralNet):
    model: DCPenaltyNN
