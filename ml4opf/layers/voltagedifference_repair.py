import torch, torch.nn as nn

from typing import Optional
from torch import Tensor

from ml4opf.layers.slackbus_repair import SlackBusRepair


class VoltageDifferenceRepair(nn.Module):
    r"""Repair voltage angle differences to voltage angles using least-squares."""

    def __init__(self, branch_incidence_matrix: Tensor, slackbus_idx: Optional[int] = None):
        """Initialize VoltageDifferenceRepair module.

        Args:
            branch_incidence_matrix (Tensor): Branch incidence matrix. (nbranch, nbus)
            slackbus_idx (Optional[int]): Index of slack bus.
        """
        super().__init__()
        self.register_buffer("branch_incidence_matrix", branch_incidence_matrix, persistent=True)

        if slackbus_idx is not None:
            self.repair_slack = True
            self.sbr = SlackBusRepair(slackbus_idx)
        else:
            self.repair_slack = False

    def forward(self, dva: Tensor, driver: str = "gelsy") -> Tensor:
        r"""Repair voltage angle differences to voltage angles using least-squares.

        Args:
            dva (Tensor): Voltage angle differences. (batch_size, nbranch)
            driver (str): Least-squares driver. Default: "gelsy".

        Returns:
            Tensor: Voltage angles. (batch_size, nbus)
        """
        va = self.repair(dva, self.branch_incidence_matrix, driver=driver)

        if self.repair_slack:
            va = self.sbr(va)

        return va

    @staticmethod
    @torch.jit.script
    def repair(dva: Tensor, branch_incidence_matrix: Tensor, driver: str = "gelsy") -> Tensor:
        r"""Convert angle differences per-branch to per-bus voltage angles using least-squares.

        Solves the problem \( \min \| A \theta - \Delta \|_2^2 \) for \( \theta \), where \( A \) is the branch incidence matrix.

        Args:
            dva (Tensor): Angle differences per branch (\( \Delta \)). (batch_size, nbranch)

        Returns:
            Tensor: Voltage angles per bus (\( \theta \)). (batch_size, nbus)
        """

        Ab = branch_incidence_matrix.swapaxes(0, 1).to_dense().expand(dva.shape[0], -1, -1)

        lsq = torch.linalg.lstsq(Ab, dva, driver=driver)

        return lsq.solution
