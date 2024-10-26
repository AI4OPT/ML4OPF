""" Abstract base class for each OPF problem.

Each formulation should inhereit from `OPFProblem` and implement the following:  

- `_parse_sanity_check`: Use `self.train_data`, `self.test_data`, and `self.json_data` to perform sanity checks making sure they correspond to the same dataset.  

- `feasibility_check`: Dictionary of keys and values to check feasibility of the problem. Each key is checked to have the corresponding value.

- `default_combos`: A dictionary where keys represent elements of the tuple from the TensorDataset and values are keys of the train_data dictionary which are concatenated. Used by `make_dataset`.  

- `default_order`: The order of the keys in the default_combos dictionary.  
 """

import json
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ml4opf.parsers import PGLearnParser
from ml4opf import info, debug


class OPFProblem(ABC):
    """OPF Problem
    =============

    This class parses the JSON/HDF5 files on initialization, providing a standard interface for accessing OPF data.

    OPFProblem also includes methods for creating input/target tensors from the HDF5 data by concatenating keys,
    though more complex datasets (e.g., for graph neural networks) can be created by accessing `train_data` and `json_data` directly.

    By default, initializing OPFProblem will parse the HDF5/JSON files, remove infeasible samples, and set aside 5000 samples for testing.
    The test data can be accessed via `test_data` - `train_data` will only contain the training data. Models should split the training data into
    training/validation sets themselves downstream.

    Initialization Arguments:

    - `data_directory (str)`: Path to the folder containing the problem files

    - `dataset_name (str)`: Name of the problem to use

    - `primal (bool)`: Whether to parse the primal data (default: True)

    - `dual (bool)`: Whether to parse the dual data (default: True)

    - `train_set (bool)`: Whether to parse the training set (default: True)

    - `test_set (bool)`: Whether to parse the test set (default: True)

    - `convert_to_float32 (bool)`: Whether to convert the data to float32 (default: True)

    - `sanity_check (bool)`: Whether to perform a sanity check on the parsed data (default: True)

    Attributes:

    - `path (Path)`: Path to the problem file folder

    - `name (str)`: Name of the problem to use

    - `train_data (dict)`: Dictionary of parsed HDF5 data. If `make_test_set` is True, this is only the training set.

    - `test_data (dict)`: Dictionary of parsed HDF5 data for the test set. If `make_test_set` is False, this is None.

    - `json_data (dict)`: Dictionary of parsed JSON data.

    - `violation (OPFViolation)`: OPFViolation object for computing constraint violations for this problem.

    Methods:

    - `parse`: Parse the JSON and HDF5 files for the problem

    - `make_dataset`: Create input/target tensors by combining keys from the h5 data. Returns the TensorDataset and slices for extracting the original components.

    - `slice_batch`: Extract the original components from a batch of data given the slices.

    - `slice_tensor`: Extract the original components from a tensor given the slices.
    """

    def __init__(self, data_directory: str, dataset_name: str, **parse_kwargs):
        self.path = Path(data_directory).resolve()
        self.dataset_name = dataset_name

        self.parse(**parse_kwargs)

    def parse(
        self,
        primal: bool = True,
        dual: bool = True,
        train_set: bool = True,
        test_set: bool = True,
        convert_to_float32: bool = True,
        sanity_check: bool = True,
    ):
        """Parse the JSON and HDF5 files for the problem"""
        parser = PGLearnParser(self.path)

        self.train_data = (
            parser.parse_h5(self.dataset_name, "train", primal=primal, dual=dual, convert_to_float32=convert_to_float32)
            if train_set
            else None
        )

        self.test_data = (
            parser.parse_h5(self.dataset_name, "test", primal=primal, dual=dual, convert_to_float32=convert_to_float32)
            if test_set
            else None
        )

        self.case_data = parser.parse_json(self.dataset_name)

        if sanity_check:
            self._parse_sanity_check()

    def make_dataset(
        self,
        combos: Optional[dict[str, list[str]]] = None,
        order: Optional[list[str]] = None,
        data: Optional[dict[str, Tensor]] = None,
        test_set: bool = False,
        sanity_check: bool = True,
    ) -> tuple[dict[str, Tensor], list[dict[str, slice]]]:
        """Make a TensorDataset from self.train_data given the keys in combos and the order of the keys in order."""
        if combos is None:
            assert order is None, "Must provide `combos` if `order` is provided."
            debug(f"Using default combos and order. (see `{self.__class__.__name__}.default_combos`)")
            combos = self.default_combos
            order = self.default_order

        if data is None:
            info(f"Making dataset using OPFProblem.{'test' if test_set else 'train'}_data.")
            data = self.test_data if test_set else self.train_data

        if sanity_check:
            assert set(combos.keys()) == set(order), "Keys of `combos` and elements of `order` must be the same."

            for v in combos.values():
                assert set(v).issubset(set(data.keys())), "All keys in `combos` values must be in `self.train_data`."

        d: dict[str, Tensor] = {}
        slices: dict[str, dict[str, slice]] = {}
        for k, v in combos.items():
            dat = [data[i] for i in v]

            assert (
                len(set(i.shape[:-1] for i in dat)) == 1
            ), "All tensors in a single combo must have the same shape except for the last dimension."
            d[k] = torch.cat(dat, dim=-1)

            # slices are used to slice the concatenated tensor back into the original tensors
            # so if input/pd has 201 and input/qd has 201 columns,
            # slices['input'] = {"input/pd": slice(0, 201), "input/qd": slice(201, 402)}
            slices[k] = {}
            start = 0
            for i in v:
                end = start + data[i].shape[-1]
                slices[k][i] = slice(start, end)
                start = end

        tds = TensorDataset(*[d[i] for i in order])
        slices = [slices[i] for i in order]
        return tds, slices

    @staticmethod
    def slice_batch(batch: tuple[Tensor, ...], slices: list[dict[str, slice]]):
        """Slice the batch tensors into the original tensors

        Args:
            batch (tuple[Tensor, ...]): Batch of tensors from the TensorDataset
            slices (list[dict[str, slice]]): List of dictionaries of slices

        Returns:
            tuple[dict[str, Tensor], ...]: Sliced tensors
        """
        assert len(batch) == len(slices), "Length of batch and slices must be the same."

        sliced = []
        for i, j in zip(batch, slices):
            sliced.append(OPFProblem.slice_tensor(i, j))
        return tuple(sliced)

    @staticmethod
    def slice_tensor(tensor: Tensor, slices: dict[str, slice]):
        """Slice the tensor into the original tensors

        Args:
            tensor (Tensor): Tensor to slice
            slices (dict[str, slice]): Dictionary of slices

        Returns:
            `dict[str, Tensor]`: Sliced tensors
        """
        return {k: tensor[..., v] for k, v in slices.items()}

    @property
    @abstractmethod
    def feasibility_check(self) -> dict[str, str]:
        """Dictionary of keys and values to check feasibility of the problem.

        Each key is checked to have the corresponding value. If any of them
        does not match, the sample is removed from the dataset in `OPFGeneratorH5Parser`.
        See ACOPFProblem.feasibility_check for an example.
        """

    @property
    @abstractmethod
    def default_combos(self) -> dict[str, list[str]]:
        """A dictionary where keys represent elements of the tuple
        from the TensorDataset and values are keys of the train_data
        dictionary which are concatenated.  Used by `make_dataset`."""

    @property
    @abstractmethod
    def default_order(self) -> list[str]:
        """The order of the keys in the default_combos dictionary."""

    def _parse_sanity_check(self):
        """Use self.train_data, self.test_data, self.json_data to
        perform sanity checks making sure they correspond to the same dataset."""
        datas = []
        if self.train_data is not None:
            datas.append(self.train_data)
        if self.test_data is not None:
            datas.append(self.test_data)

        for data in datas:
            if "input/meta/seed" in data and "meta/seed" in data:
                assert len(data["input/meta/seed"]) == len(
                    data["meta/seed"]
                ), "Seed lengths do not match from input/meta to data meta."
                assert all(
                    data["input/meta/seed"] == data["meta/seed"]
                ), "Seeds in input/meta and data meta do not match."
            if "input/meta/config" in data and "meta/config" in data:
                assert data["input/meta/config"] == data["meta/config"], "Configs in input and data meta do not match."

            if "input/meta/config" in data and "config" in self.case_data:
                assert (
                    json.loads(data["input/meta/config"]) == self.case_data["config"]
                ), "Config in input and JSON do not match."
