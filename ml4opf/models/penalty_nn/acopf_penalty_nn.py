from torch import Tensor

from ml4opf.models.penalty_nn.penalty_nn import PenaltyNN, PenaltyNeuralNet
from ml4opf.models.basic_nn.acopf_basic_nn import ACBasicNN, ACBasicNeuralNet
from ml4opf.loss_functions import ObjectiveLoss
from ml4opf import ACViolation, ACModel


class ACPenaltyNN(PenaltyNN, ACBasicNN):
    violation: ACViolation
    opfmodel: ACModel

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
        vm_pred = y_hat[:, self.vm_slice]
        va_pred = y_hat[:, self.va_slice]

        loss = self.loss.forward(
            base_loss,
            pd=pd,
            qd=qd,
            pg=pg_pred,
            qg=qg_pred,
            vm=vm_pred,
            va=va_pred,
        )

        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        return loss


class ACPenaltyNeuralNet(PenaltyNeuralNet, ACBasicNeuralNet):
    model: ACPenaltyNN
