""" A feed-forward fully-connected neural network with a pre-computed PCA as the last layer."""

try:
    from ml4opf.models.pca_nn.pca_nn import ACPCANeuralNet, DCPCANeuralNet, EDPCANeuralNet, SOCPCANeuralNet

    __all__ = ["ACPCANeuralNet", "DCPCANeuralNet", "EDPCANeuralNet", "SOCPCANeuralNet"]
except ImportError as e:
    INSTALL_CMD = "pip install lightning"
    raise ImportError(
        f"Could not import PCANeuralNet, probably because pytorch-lightning is not installed.\n To install it, run\n\t{INSTALL_CMD}"
    ) from e
