""" A basic feed-forward fully-connected neural network."""

try:
    from ml4opf.models.basic_nn.acopf_basic_nn import ACBasicNeuralNet
    from ml4opf.models.basic_nn.dcopf_basic_nn import DCBasicNeuralNet
    from ml4opf.models.basic_nn.ed_basic_nn import EDBasicNeuralNet
    from ml4opf.models.basic_nn.socopf_basic_nn import SOCBasicNeuralNet

    __all__ = ["ACBasicNeuralNet", "DCBasicNeuralNet", "EDBasicNeuralNet", "SOCBasicNeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import BasicNeuralNet, probably because pytorch-lightning is not installed.\n To install it, run\n\t{INSTALL_CMD}"
    ) from e
