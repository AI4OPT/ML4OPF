import torch, torch.nn as nn

from torch import Tensor


class SlackBusRepair(nn.Module):
    r"""Repair voltage angles by setting the slack bus angle to zero."""

    def __init__(self, slackbus_idx: int):
        """Initialize SlackBusRepair module.

        Args:
            slackbus_idx (int): Index of slack bus.
        """
        super().__init__()

        if not isinstance(slackbus_idx, int):
            if (
                isinstance(slackbus_idx, Tensor)
                and slackbus_idx.numel() == 1
                and not slackbus_idx.dtype.is_floating_point
            ):
                slackbus_idx = slackbus_idx.item()
            else:
                raise TypeError(f"slackbus_idx must be an int, not {type(slackbus_idx)}")

        self.slackbus_idx = slackbus_idx

    def forward(self, va: Tensor) -> Tensor:
        r"""Repair voltage angles by setting the slack bus angle to zero.

        \[ \text{va}_i = \text{va}_i - \text{va}_\text{slack}  \]

        Args:
            va (Tensor): Voltage angles. (batch_size, nbus)

        Returns:
            Tensor: Repaired voltage angles. (batch_size, nbus)
        """
        return self.repair(va, self.slackbus_idx)

    @staticmethod
    @torch.jit.script
    def repair(va: Tensor, slackbus_idx: int) -> Tensor:
        r"""Repair voltage angles by setting the slack bus angle to zero.

        \[ \text{va}_i = \text{va}_i - \text{va}_\text{slack}  \]

        Args:
            va (Tensor): Voltage angles. (batch_size, nbus)
            slackbus_idx (int): Index of slack bus.

        Returns:
            Tensor: Repaired voltage angles. (batch_size, nbus)
        """
        return va - va[..., slackbus_idx].unsqueeze(-1)
