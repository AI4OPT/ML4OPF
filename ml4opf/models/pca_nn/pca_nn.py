import torch, torch.nn as nn
from torch import Tensor

from tqdm import tqdm
from ml4opf import OPFModel, debug
from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet
from ml4opf.models.basic_nn.lightning_basic_nn import BasicNN
from ml4opf.models.basic_nn.acopf_basic_nn import ACBasicNN, ACBasicNeuralNet
from ml4opf.models.basic_nn.dcopf_basic_nn import DCBasicNN, DCBasicNeuralNet
from ml4opf.models.basic_nn.socopf_basic_nn import SOCBasicNN, SOCBasicNeuralNet
from ml4opf.models.basic_nn.ed_basic_nn import EDBasicNN, EDBasicNeuralNet


class InversePCALayer(nn.Module):
    def __init__(self, pca_w: Tensor, pca_mu: Tensor):
        super().__init__()
        self.register_buffer("pca_w", pca_w)  # n_components x n_features
        self.register_buffer("pca_mu", pca_mu)  # n_features

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size x n_components] @ [n_features x n_components]ᵀ + [n_features] = [batch_size x n_features]
        return x @ self.pca_w.T + self.pca_mu


class PCANN(BasicNN):
    def __init__(
        self,
        opfmodel: OPFModel,
        slices: list[slice],
        pca_w: Tensor,
        pca_mu: Tensor,
        optimizer: str = "adam",
        loss: str = "mse",
        hidden_sizes: list[int] = [100, 100],
        activation: str = "relu",
        boundrepair: str = "none",
        learning_rate: float = 1e-3,
        weight_init_seed: int = 42,
    ):
        super(BasicNN, self).__init__()

        self.opfmodel = opfmodel
        self.violation = opfmodel.violation
        self.save_hyperparameters(ignore=["opfmodel"])

        self.slices = slices
        assert (
            len(slices) == 2
        ), "Got len(slices) != 2. First slice should correspond to the inputs, second slice to the outputs. See BasicNN.make_dataset."
        self.hidden_sizes = hidden_sizes

        self.set_activation(activation)
        self.set_loss(loss)
        self.make_network(pca_w, pca_mu)
        self.add_boundrepair(boundrepair)
        self.make_optimizer(optimizer, learning_rate)
        self.init_weights(seed=weight_init_seed)  # see nn.Linear.reset_parameters and pytorch#57109
        self.to(torch.float32)

    def make_network(self, pca_w: Tensor, pca_mu: Tensor):
        self.layers = nn.Sequential()

        self.layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        self.layers.append(self.activation())

        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.layers.append(self.activation())

        # self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
        self.layers.append(InversePCALayer(pca_w, pca_mu))


class PCANeuralNet(BasicNeuralNet):
    """A feed-forward neural network with a PCA layer at the end.
    The PCA is computed on initialization using the training set."""

    model: PCANN

    def make_training_model(self, force_new_model=False):
        if hasattr(self, "model") and not force_new_model:
            debug("Model already created. Skipping.")
            return

        debug("Computing incremental PCA on training set...")
        from sklearn.decomposition import IncrementalPCA

        ipca = IncrementalPCA(n_components=self.config["hidden_sizes"][-1])
        for batch in tqdm(self.train_loader, desc="Fitting PCA"):
            ipca.partial_fit(batch[1])  # assuming target is batch[1]

        debug("Done computing incremental PCA.")

        self.model = self.model_cls(
            opfmodel=self,
            slices=self.slices,
            pca_w=torch.as_tensor(ipca.components_, dtype=torch.float32).T,
            pca_mu=torch.as_tensor(ipca.mean_, dtype=torch.float32),
            **self.config,
        )
        self.model._compiled = False  # will (optionally) be compiled in train()


class ACPCANN(PCANN, ACBasicNN):
    pass


class DCPCANN(PCANN, DCBasicNN):
    pass


class SOCPCANN(PCANN, SOCBasicNN):
    pass


class EDPCANN(PCANN, EDBasicNN):
    pass


class ACPCANeuralNet(PCANeuralNet, ACBasicNeuralNet):
    model: ACPCANN


class DCPCANeuralNet(PCANeuralNet, DCBasicNeuralNet):
    model: DCPCANN


class SOCPCANeuralNet(PCANeuralNet, SOCBasicNeuralNet):
    model: SOCPCANN


class EDPCANeuralNet(PCANeuralNet, EDBasicNeuralNet):
    model: EDPCANN
