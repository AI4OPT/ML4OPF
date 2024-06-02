import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.parsers.parse_h5 import H5Parser


def test_parse_h5():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    hp = H5Parser(data_dir / "300_ieee_input.h5", data_dir / "300_ieee_DCOPF.h5")

    dat = hp.parse(train_set_size=5)

    hp.get_n_samples(dat, sanity_check=True)

    with pytest.raises(ValueError):
        H5Parser.get_n_samples({})

    with pytest.raises(AssertionError):
        dat["input/pd"] = dat["input/pd"][:1, :]
        H5Parser.get_n_samples(dat, sanity_check=True)

    dat = hp.parse(train_set_size=5)
    tree = H5Parser.make_tree(dat)
    assert tree.keys() == {"dual", "primal", "meta", "input"}

    dat2 = hp.parse(train_set_size=50)
    H5Parser.filter_by_total_load(dat2, (None, None))
    assert H5Parser.get_n_samples(dat2) == 50

    ts = H5Parser.extract_test_set(dat2, test_set_size=1, seed=None)
    assert H5Parser.get_n_samples(ts) == 1
    assert H5Parser.get_n_samples(dat2) > 2

    H5Parser.keep_n_samples(dat2, 2, seed=None)
    assert H5Parser.get_n_samples(dat2) == 2

    H5Parser.keep_n_samples(dat2, 10, seed=1)
    assert H5Parser.get_n_samples(dat2) == 2

    train, test = hp.parse(make_test_set=True, test_set_size=10)
    assert test["meta/seed"].flatten().tolist() == [17, 25, 43, 47, 61, 63, 79, 96, 97, 99]
