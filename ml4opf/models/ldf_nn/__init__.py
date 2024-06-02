""" A basic feed-forward fully-connected neural network for ACOPF, using the `LDFLoss` loss function."""

try:
    from ml4opf.models.ldf_nn.acp_ldf_nn import ACPLDFNeuralNet
    from ml4opf.models.ldf_nn.dcp_ldf_nn import DCPLDFNeuralNet

    __all__ = ["ACPLDFNeuralNet", "DCPLDFNeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import LDFNeuralNet, probably because pytorch-lightning is not installed.\n To install it, run:\n\t{INSTALL_CMD}"
    ) from e
