""" A basic feed-forward fully-connected neural network, using the `PenaltyLoss` loss function."""

try:
    from ml4opf.models.penalty_nn.acopf_penalty_nn import ACPenaltyNeuralNet
    from ml4opf.models.penalty_nn.dcopf_penalty_nn import DCPenaltyNeuralNet

    __all__ = ["ACPenaltyNeuralNet", "DCPenaltyNeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import PenaltyNeuralNet, probably because pytorch-lightning is not installed.\n To install it, run:\n\t{INSTALL_CMD}"
    ) from e
