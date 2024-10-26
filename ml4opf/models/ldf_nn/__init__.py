""" A basic feed-forward fully-connected neural network, using the `LDFLoss` loss function."""

try:
    from ml4opf.models.ldf_nn.acopf_ldf_nn import ACLDFNeuralNet
    from ml4opf.models.ldf_nn.dcopf_ldf_nn import DCLDFNeuralNet

    __all__ = ["ACLDFNeuralNet", "DCLDFNeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import LDFNeuralNet, probably because pytorch-lightning is not installed.\n To install it, run:\n\t{INSTALL_CMD}"
    ) from e
