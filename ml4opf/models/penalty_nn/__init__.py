""" A basic feed-forward fully-connected neural network, using the `PenaltyLoss` loss function."""

try:
    from ml4opf.models.penalty_nn.acp_penalty_nn import ACPPenaltyNeuralNet
    from ml4opf.models.penalty_nn.dcp_penalty_nn import DCPPenaltyNeuralNet

    __all__ = ["ACPPenaltyNeuralNet", "DCPPenaltyNeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import LDFNeuralNet, probably because pytorch-lightning is not installed.\n To install it, run:\n\t{INSTALL_CMD}"
    ) from e
