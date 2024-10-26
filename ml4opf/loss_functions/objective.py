""" Use the formulation objective as the loss function """

from typing import Optional

from torch import nn, Tensor

from ml4opf.formulations import OPFViolation


class ObjectiveLoss(nn.Module):
    """
    ObjectiveLoss is the original objective of the OPF.

    It takes as input the same arguments as the corresponding formulation's
    `objective` method, and returns the objective value.
    """

    SUPPORTED_REDUCTIONS = ("none", "mean", "sum", "max")

    def __init__(self, v: OPFViolation, reduction: Optional[str] = "mean"):
        """Initialize ObjectiveLoss module.

        Args:
            v (OPFViolation): OPFViolation module.
            reduction (Optional[str]): Reduction operation. Default: "mean".
        """
        super().__init__()

        assert hasattr(v, "objective"), f"{v.__class__.__name__}.objective() must be implemented for ObjectiveLoss"

        self.v = v

        assert reduction in self.SUPPORTED_REDUCTIONS, f"{reduction=} must be one of {self.SUPPORTED_REDUCTIONS}"

        self.reduction = reduction

    def forward(self, *objective_args, **objective_kwargs) -> Tensor:
        """Compute the objective value for a batch of samples."""
        obj = self.v.objective(*objective_args, **objective_kwargs)

        if self.reduction == "mean":
            return obj.mean()
        elif self.reduction == "sum":
            return obj.sum()
        elif self.reduction == "max":
            return obj.max()
        elif self.reduction == "none":
            return obj
        else:
            raise ValueError(f"Invalid {self.reduction=}. Supported: {self.SUPPORTED_REDUCTIONS}")
