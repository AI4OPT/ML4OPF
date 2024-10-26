"""
This file gives an example implementation of how to read an entire HDF5 into a Python dictionary.
Note that if only using a subset of the data, it is better to use the `h5py.File`
interface directly since it avoids reading unnecessary data from disk.

"""
import h5py
import numpy as np


def parse_hdf5(path, preserve_shape=False):
    """Parse an HDF5 file into a dictionary.

    Args:
        path (str): Path to the HDF5 file.
        preserve_shape (bool, optional): Whether to preserve the shape of the data. Defaults to False.

    Returns:
        dict: Dictionary containing the data from the HDF5 file.
    """
    dat = dict()

    def read_direct(dataset: h5py.Dataset):
        arr = np.empty(dataset.shape, dtype=dataset.dtype)

        if set(dataset.shape) == {0}:
            pass  # pragma: no cover
        else:
            dataset.read_direct(arr)

        return arr

    def store(name, obj):
        if isinstance(obj, h5py.Group):
            return
        elif isinstance(obj, h5py.Dataset):
            dat[name] = read_direct(obj)
        else:
            raise ValueError(
                f"Unexcepted type: {type(obj)} under name {name}. Expected h5py.Group or h5py.Dataset."
            )  # pragma: no cover

    with h5py.File(path, "r") as f:
        f.visititems(store)

    if preserve_shape:
        # recursively correct the shape of the dictionary
        ret = dict()

        def r_correct_shape(d: dict, ret: dict):
            for k in list(d.keys()):
                if "/" in k:
                    k1, k2 = k.split("/", 1)
                    if k1 not in ret:
                        ret[k1] = dict()
                    r_correct_shape({k2: d[k]}, ret[k1])
                    del d[k]
                else:
                    ret[k] = d[k]

        r_correct_shape(dat, ret)

        return ret
    else:
        return dat
