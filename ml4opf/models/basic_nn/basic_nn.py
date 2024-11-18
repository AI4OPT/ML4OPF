import torch, pytorch_lightning as pl, json, yaml

from torch.utils.data import DataLoader, random_split
from pathlib import Path
from abc import abstractmethod, ABC
from typing import Optional, get_type_hints

from ml4opf import OPFProblem, OPFModel, warn, debug


class BasicNeuralNet(OPFModel, ABC):
    """A basic feed-forward neural network.

    Args:
        config (dict): Dictionary containing the model configuration.

        `optimizer` (str): Optimizer. Supported: "adam", "adamw", "sgd".

        `loss` (str): Loss function. Supported: "mse", "l1".

        `hidden_sizes` (list[int]): List of hidden layer sizes.

        `activation` (str): Activation function. Supported: "relu", "tanh", "sigmoid".

        `boundrepair` (str): Bound clipping method. Supported: "none", "relu", "clamp", "sigmoid".

        `learning_rate` (float): Learning rate.

        problem (OPFProblem): The OPFProblem object.
    """

    @property
    def model_cls(self):
        """The LightningModule class to use for training. Must be a subclass of `BasicNN`."""
        return get_type_hints(self)["model"]

    def __init__(self, config: dict, problem: OPFProblem):
        self.init_config = config
        self.problem = problem
        self.violation = self.problem.violation

        expected_keys = get_type_hints(self.model_cls.__init__)
        expected_keys.pop("opfmodel")
        expected_keys.pop("slices")
        
        # make sure there are no extra keys
        if any(key not in expected_keys.keys() for key in config.keys()):
            raise ValueError(f"Invalid config keys: {config.keys()}. Expected: {expected_keys}")

        self.config = config

    def make_training_model(self, force_new_model=False):
        if hasattr(self, "model") and not force_new_model:
            debug("Model already created. Skipping.")
            return

        self.model = self.model_cls(opfmodel=self, slices=self.slices, **self.config)
        self.model._compiled = False # will (optionally) be compiled in train()

    def make_dataset(self, seed: int = 42, dl_kwargs: Optional[dict] = None, **kwargs):
        if hasattr(self, "train_loader"):
            debug("Dataset already created. Skipping.")
            return
        
        if dl_kwargs is None:
            dl_kwargs = {}

        dataset, self.slices = self.problem.make_dataset(**kwargs)

        rng = torch.Generator().manual_seed(seed)
        train, val = random_split(dataset, [0.8, 0.2], generator=rng)

        dl_kwargs.setdefault("batch_size", 32)
        dl_kwargs.setdefault("num_workers", 8)

        if "shuffle" in dl_kwargs:
            warn("Ignoring shuffle DataLoader argument -- shuffle is set to True for training and False for validation. "
                 + f"To change this behavior, override `{self.__class__.__name__}.make_dataset`.")
            dl_kwargs.pop("shuffle")

        self.train_loader = DataLoader(
            train, shuffle=True, generator=rng, **dl_kwargs
        )
        self.val_loader = DataLoader(
            val, shuffle=False, generator=rng, **dl_kwargs
        )

    def make_trainer(self, force_new_trainer=False, **kwargs):
        if hasattr(self, "trainer") and not force_new_trainer:
            debug("Trainer already created. Skipping.")
            return

        self.trainer = pl.Trainer(**kwargs)

    def train(
        self,
        force_new_model: bool = False,
        trainer_kwargs: Optional[dict] = None,
        dataset_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        compile_kwargs: Optional[dict] = None,
    ):
        if trainer_kwargs is None:
            trainer_kwargs = {}
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        if compile_kwargs is None:
            debug(f"Skipping model compilation since no compile_kwargs were given to {self.__class__.__name__}.train(). To use torch.compile with default kwargs, pass " r"compile_kwargs={}.")
            compile_kwargs = {}
            skip_compile = True
        else:
            skip_compile = False

        self.make_dataset(**dataset_kwargs)
        self.make_trainer(**trainer_kwargs)
        self.make_training_model(force_new_model=force_new_model)

        if not self.model._compiled and not skip_compile:
            self.model = torch.compile(self.model, **compile_kwargs)
            self.model._compiled = True

        self.trainer.fit(self.model, self.train_loader, self.val_loader, **fit_kwargs)

    def save_checkpoint(self, path_to_folder: str):
        if not hasattr(self, "trainer"):
            raise ValueError("Trainer never created. Cannot save checkpoint.")

        path = Path(path_to_folder).resolve()
        if not path.exists():
            path.mkdir(parents=True)
        if not path.is_dir(): # pragma: no cover
            raise ValueError(f"Checkpoint path must be a directory. Got: {path}")

        checkpoint = self.trainer._checkpoint_connector.dump_checkpoint(False)
        checkpoint["__cls"] = self.__class__.__name__

        if self.model._compiled:
            warn("Saving checkpoint of model compiled with torch.compile. Loading this checkpoint does not preserve compilation -- see pytorch#101107.")
            checkpoint["__model_cls"] = self.model._orig_mod.__class__.__name__
        else:
            checkpoint["__model_cls"] = self.model.__class__.__name__

        self.trainer.strategy.save_checkpoint(checkpoint, path / "trainer.ckpt", storage_options=None)
        self.trainer.strategy.barrier("Trainer.save_checkpoint")

    @classmethod
    def load_from_checkpoint(cls, path_to_folder: str, problem: OPFProblem):
        path = Path(path_to_folder).resolve()
        if not path.exists(): # pragma: no cover
            raise ValueError(f"Checkpoint path does not exist. Got: {path}")
        if not path.is_dir(): # pragma: no cover
            # if path.as_posix().endswith(".tar.gz"):
            #     checkpoint_name = path.as_posix().split("/")[-1].replace(".tar.gz", "")
            #     with tempfile.TemporaryDirectory() as tmpdir:
            #         with tarfile.open(path, "r:gz") as tar:
            #             (
            #                 tar.extractall(tmpdir, filter="data")
            #                 if hasattr(tarfile, "data_filter")  # python 3.12
            #                 else tar.extractall(tmpdir)
            #             )
            #         return cls.load_from_checkpoint(Path(tmpdir) / checkpoint_name, problem)
            # else:
            raise ValueError(f"Checkpoint path must be a directory! Got: {path}")

        assert (path / "trainer.ckpt").exists(), f"Checkpoint model file not found at {path}/trainer.ckpt"

        with open(path / "trainer.ckpt", "rb") as f:
            d = torch.load(f, weights_only=False)
            config = d['hyper_parameters']
            assert d["__cls"] == cls.__name__, f"Checkpoint class {d['__cls']} does not match {cls.__name__}"
            config__model_cls = d["__model_cls"]
            del d["__cls"]
            del d["__model_cls"]

            slices = config.pop("slices")
            me = cls(config=config, problem=problem)
            me.slices = slices

            assert (
                config__model_cls == me.model_cls.__name__
            ), f"Checkpoint model class {config__model_cls} does not match {me.model_cls.__name__}"

        me.model = me.model_cls.load_from_checkpoint(path / "trainer.ckpt", opfmodel=me)
        me.model._compiled = False
        return me
