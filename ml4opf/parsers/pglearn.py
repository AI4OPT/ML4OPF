""" Parser for the PGLearn datasets """

import bz2
import gzip
import json
import shutil
from pathlib import Path
from typing import Union, Sequence
from collections import defaultdict

import h5py
import torch
import numpy as np

from ml4opf import debug, warn

ParsedHDF5Dict = dict[str, Union[torch.Tensor, np.ndarray, np.str_]]
H5Object = Union[h5py.Group, h5py.Dataset]


class PGLearnParser:
    """
    Parser for PGLearn dataset.
    """

    def __init__(self, data_path: Union[str, Path]):
        """Initialize the parser by validating and setting the path."""
        self.data_path = self.validate_path(data_path)
        self.json_path = self.data_path / "case.json"

    def validate_path(self, path: Union[str, Path]) -> Path:
        """Validate the path to the HDF5 file."""
        path_obj = Path(path).resolve()

        assert path_obj.exists(), f"PGLearn path {path_obj} does not exist."
        assert path_obj.is_dir(), f"PGLearn path {path_obj} is not a directory."
        assert any(
            (
                (path_obj / "case.json").exists(),
                (path_obj / "case.json.gz").exists(),
                (path_obj / "case.json.bz2").exists(),
            )
        ), f"PGLearn folder does not contain a case data file. Expected case.json, case.json.gz, or case.json.bz2 in {path}."

        return path_obj

    # ┌─────────────────────────────┐
    # │         HDF5 parser         │
    # └─────────────────────────────┘

    def parse_h5(
        self,
        dataset_name: str,
        split: str = "train",  # "train", "test", "infeasible"
        primal: bool = True,
        dual: bool = False,
        convert_to_float32: bool = True,
    ) -> Union[ParsedHDF5Dict, tuple[ParsedHDF5Dict, ParsedHDF5Dict]]:
        """Parse the HDF5 file.

        Args:
            dataset_name (str): The name of the dataset. Typically the formulation ("ACOPF", "DCOPF", etc.).
            split (str, optional): The split to return. Defaults to "train".
            primal (bool, optional): If True, parse the primal file. Defaults to True.
            dual (bool, optional): If True, parse the dual file. Defaults to False.
            convert_to_float32 (bool, optional): If True, convert all float64 data to torch.float32. Defaults to True.

        Returns:
            dict: Flattened dictionary of HDF5 data with PyTorch tensors for numerical data and NumPy arrays for string/object data.

        If `make_test_set` is True, then this function will return a tuple of two dictionaries. The first dictionary is the
        training set and the second dictionary is the test set. The test set is a random 10% sample of the training set.

        This parser will return a single-level dictionary where the keys are in the form
        of `solution/primal/pg` where `solution` is the group, `primal` is the subgroup,
        and `pg` is the dataset from the HDF5 file. The values are PyTorch tensors. This
        parser uses `h5py.File.visititems` to iterate over the HDF5 file quickly.
        """
        dat: ParsedHDF5Dict = dict()

        def read_direct(dataset: h5py.Dataset):
            arr = np.empty(dataset.shape, dtype=dataset.dtype)

            if set(dataset.shape) == {0}:
                pass  # pragma: no cover
            else:
                dataset.read_direct(arr)

            if arr.ndim == 2:
                if arr.shape[0] == 1:
                    debug(
                        f"Converting {dataset.name}: {arr} with shape (1, {arr.shape[1]}) to shape ({arr.shape[1],})."
                    )
                    arr = arr.squeeze(0)
                elif arr.shape[1] == 1:
                    debug(
                        f"Converting {dataset.name}: {arr} with shape ({arr.shape[0]}, 1) to shape ({arr.shape[0],})."
                    )
                    arr = arr.squeeze(1)

            return arr

        def store(name: str, obj: H5Object, mode: str):
            key = mode + "/" + name
            if isinstance(obj, h5py.Group):
                return
            if isinstance(obj, h5py.Dataset):
                if (
                    obj.dtype.kind in "fiucb"
                ):  # floating-point, signed integer, unsigned integer, complex floating-point, boolean
                    dat[key] = torch.from_numpy(read_direct(obj))
                elif obj.dtype.kind in "SO":  # (byte-)string/object
                    dat[key] = read_direct(obj).astype(str)
                    if dat[key].ndim == 0:
                        debug(f"Converting {name}: {dat[key]} with shape () to shape (1,).")
                        dat[key] = dat[key][()]
                else:
                    raise ValueError(f"Unknown dtype: {obj.dtype} in name {name}")  # pragma: no cover
            else:
                raise ValueError(f"Unknown type: {type(obj)} in name {name}")  # pragma: no cover

        if primal:
            debug("Opening HDF5 primal file")
            with MaybeGunzipH5File(self.data_path / split / dataset_name / "primal.h5", "r") as f:
                f.visititems(lambda name, obj: store(name, obj, "primal"))
            debug("Closed HDF5 primal file")

        if dual:
            debug("Opening HDF5 dual file")
            with MaybeGunzipH5File(self.data_path / split / dataset_name / "dual.h5", "r") as f:
                f.visititems(lambda name, obj: store(name, obj, "dual"))
            debug("Closed HDF5 dual file")

        debug("Opening HDF5 meta file")
        with MaybeGunzipH5File(self.data_path / split / dataset_name / "meta.h5", "r") as f:
            f.visititems(lambda name, obj: store(name, obj, "meta"))
        debug("Closed HDF5 meta file")

        debug("Opening HDF5 input file")
        with MaybeGunzipH5File(self.data_path / split / "input.h5", "r") as f:
            f.visititems(lambda name, obj: store(name, obj, "input"))
        debug("Closed HDF5 input file")

        if convert_to_float32:
            self.convert_to_float32(dat)

        return dat

    @staticmethod
    def convert_to_float32(dat: ParsedHDF5Dict):
        """Convert all float64 data to float32 in-place."""
        for k, v in dat.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                debug(f"Converting {k} from float64 to float32.")
                dat[k] = v.to(torch.float32)

    @staticmethod
    def make_tree(dat: ParsedHDF5Dict, delimiter: str = "/"):
        """Convert a flat dictionary to a tree.
        Note that the keys of `dat` must have a tree structure where data is only at the leaves.
        Assumes keys are delimited by "/", i.e. "solution/primal/pg".

        Args:
            dat (dict): Flat dictionary of data.
            delimiter (str, optional): Delimiter to use for splitting keys. Defaults to "/".

        Returns:
            dict: Tree dictionary of data from `dat`.
        """
        tree = dict()

        def r_correct_shape(d: dict, ret: dict):
            for k in list(d.keys()):
                if delimiter in k:
                    k1, k2 = k.split(delimiter, 1)
                    if k1 not in ret:
                        ret[k1] = dict()
                    r_correct_shape({k2: d[k]}, ret[k1])
                    del d[k]
                else:
                    ret[k] = d[k]

        r_correct_shape(dat, tree)

        return tree

    # ┌─────────────────────────────┐
    # │         JSON parser         │
    # └─────────────────────────────┘

    def open_json(self):
        """Open the JSON file, supporting gzip and bz2 compression based on the file suffix."""

        if self.json_path.exists():
            return open(self.json_path, "r", encoding="utf-8")
        elif (gz_path := self.json_path.with_suffix(".json.gz")).exists():
            return gzip.open(gz_path, "rt", encoding="utf-8")
        elif (bz2_path := self.json_path.with_suffix(".json.bz2")).exists():
            return bz2.open(bz2_path, "rt", encoding="utf-8")
        else:
            raise ValueError(f"JSON file not found: {self.json_path}")

    def parse_json(self, model_type: Union[str, Sequence[str]] = None):
        """Parse the JSON file from PGLearn.

        Args:
            model_type (Union[str, Sequence[str]]): The reference solutions to save. Default: [] (no reference solutions saved.)

        Returns:
            dict: Dictionary containing the parsed data.

        In the JSON file, the data is stored by each individual component.
        So to get generator 1's upper bound on active generation, you'd look at:
        raw_json['data']['gen']['1']['pmax'] and get a float.

        In the parsed version, we aggregate each of the components attributes into torch.Tensor arrays.
        So to get generator 1's upper bound on active generation, you'd look at:
        dat['gen']['pmax'][0] and get a float.
        Note that the index is 0-based and an integer, not 1-based and a string.

        To access the reference solution, pass a model_type (or multiple) and then access dat["ref_solutions"][model_type].
        """
        if model_type is None:
            model_type = []

        dat = {}

        with self.open_json() as f:
            raw_json = json.load(f)
            file_data = raw_json["data"]
            dat["config"] = raw_json["config"]

            if isinstance(model_type, str):
                model_type = [model_type]

            dat["ref_solutions"] = {}
            for mtype in model_type:
                ref_solution = raw_json.get(mtype, None)
                if ref_solution is not None:
                    dat["ref_solutions"][mtype] = dict(meta=ref_solution["meta"])
                    dat["ref_solutions"][mtype]["primal"] = {
                        key: torch.as_tensor(value) for key, value in ref_solution["primal"].items()
                    }
                    dat["ref_solutions"][mtype]["dual"] = {
                        key: torch.as_tensor(value) for key, value in ref_solution["dual"].items()
                    }
                else:
                    warn(f"Reference solution for model type {mtype} not found in JSON file.")

        for key, value in file_data.items():
            if key == "case":
                # skip the case string
                continue
            elif isinstance(value, dict) and value.keys() == {"I", "J", "V", "M", "N"}:
                # parse sparse matrices
                I = torch.as_tensor(value["I"], dtype=torch.long)
                J = torch.as_tensor(value["J"], dtype=torch.long)
                V = torch.as_tensor(value["V"], dtype=torch.float)
                dat[key] = torch.sparse_coo_tensor(
                    indices=torch.stack([I, J]),
                    values=V,
                    size=(value["M"], value["N"]),
                ).coalesce()
            elif key in ["bus_arcs_fr", "bus_arcs_to", "bus_gens", "bus_loads"]:
                # convert component indices to 0-based, pad to dense tensors
                padvalue = file_data[PGLearnParser.padval[key]]
                dat[key] = torch.as_tensor(PGLearnParser.pad_to_dense(value, padvalue + 1)) - 1
            elif key in ["bus_fr", "bus_to", "load_bus", "gen_bus"]:
                # convert component indices to 0-based indices
                dat[key] = torch.as_tensor(value) - 1
            else:
                # try to convert to tensor
                dat[key] = torch.as_tensor(value)

        return dat

    padval = {
        "bus_arcs_fr": "E",
        "bus_arcs_to": "E",
        "bus_gens": "G",
        "bus_loads": "L",
    }

    @staticmethod
    def pad_to_dense(array, padval, dtype=int):
        """from https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape"""

        def iterate_nested_array(array, index=()):
            try:
                for idx, row in enumerate(array):
                    yield from iterate_nested_array(row, (*index, idx))
            except TypeError:  # final level
                yield (*index, slice(len(array))), array

        def get_max_shape(array):
            def get_dimensions(array, level=0):
                yield level, len(array)
                try:
                    for row in array:
                        yield from get_dimensions(row, level + 1)
                except TypeError:  # not an iterable
                    pass

            dimensions = defaultdict(int)
            for level, length in get_dimensions(array):
                dimensions[level] = max(dimensions[level], length)
            return [value for _, value in sorted(dimensions.items())]

        dimensions = get_max_shape(array)
        result = np.full(dimensions, padval)
        for index, value in iterate_nested_array(array):
            result[index] = np.array(value)
        return result.astype(dtype)


class MaybeGunzipH5File(h5py.File):
    def __init__(self, name: str, *args, **kwargs):
        namepath = Path(name)
        if not namepath.exists():
            gznamepath = namepath.with_suffix(".h5.gz")
            assert gznamepath.exists(), f"File {name} does not exist and no gzipped version was found."
            warn(f"Unzipping {gznamepath}. This may take a while, but it only runs once; it will delete the original compressed file and replace it with the uncompressed file.")
            with gzip.open(gznamepath, "rb") as f_in:
                with open(name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gznamepath.unlink()
        super().__init__(name, *args, **kwargs)
