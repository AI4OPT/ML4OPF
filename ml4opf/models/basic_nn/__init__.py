""" A basic feed-forward fully-connected neural network."""

try:
    from ml4opf.models.basic_nn.acp_basic_nn import ACPBasicNeuralNet
    from ml4opf.models.basic_nn.dcp_basic_nn import DCPBasicNeuralNet

    __all__ = ["ACPBasicNeuralNet", "DCPBasicNeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import BasicNeuralNet, probably because pytorch-lightning is not installed.\n To install it, run\n\t{INSTALL_CMD}"
    ) from e
