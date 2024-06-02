import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.parsers.parse_json import JSONParser


def test_parse_json():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    JSONParser(data_dir / "300_ieee.ref.json").parse("ACOPF")
    JSONParser(data_dir / "gz_300_ieee.ref.json").parse("invalid")  # TODO: check warning
    dat = JSONParser(data_dir / "bz_300_ieee.ref.json").parse("DCOPF")

    assert dat.keys() == {
        "meta",
        "DCOPF",
        "basic_network",
        "source_type",
        "name",
        "source_version",
        "baseMVA",
        "per_unit",
        "n_bus",
        "bus",
        "n_gen",
        "gen",
        "n_branch",
        "branch",
        "n_shunt",
        "shunt",
        "n_load",
        "load",
    }

    with pytest.raises(FileNotFoundError):
        JSONParser(data_dir / "invalid.ref.json")
