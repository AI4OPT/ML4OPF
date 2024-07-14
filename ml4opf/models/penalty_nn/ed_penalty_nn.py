from torch import Tensor

from ml4opf.models.penalty_nn.penalty_nn import PenaltyNN, PenaltyNeuralNet
from ml4opf.models.basic_nn.ed_basic_nn import EDBasicNN, EDBasicNeuralNet
from ml4opf.loss_functions import ObjectiveLoss
from ml4opf.formulations.ed import EDViolation, EDModel


class EDPenaltyNN(PenaltyNN, EDBasicNN):
    violation: EDViolation
    opfmodel: EDModel

    def _step(self, batch: tuple[Tensor, Tensor], stage: str = ""):
        x, y = batch
        y_hat = self(x)

        if isinstance(self.base_loss, ObjectiveLoss):
            # uses formulation specific `OPFViolation.objective`
            base_loss = self.base_loss(pd=x, pg=y_hat[:, self.pg_slice])
        else:
            base_loss = self.base_loss(y_hat, y)

        pd = x[:, self.pd_slice]

        pg_pred = y_hat[:, self.pg_slice]

        loss = self.loss.forward(
            base_loss,
            pd=pd,
            pg=pg_pred,
        )

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        return loss


class EDPenaltyNeuralNet(PenaltyNeuralNet, EDBasicNeuralNet):
    model: EDPenaltyNN
