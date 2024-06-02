import pytest

import sys, importlib
import ml4opf

# NOTE: This test may break other tests if isn't last


def test_no_torch():
    old_torch = sys.modules["torch"]
    sys.modules["torch"] = None
    with pytest.raises(ImportError):
        importlib.reload(ml4opf)
    sys.modules["torch"] = old_torch


def test_no_rich():
    old_rich = sys.modules["rich"]
    old_rich_logging = sys.modules["rich.logging"]
    sys.modules["rich"] = None
    sys.modules["rich.logging"] = None

    importlib.reload(ml4opf)  # no importerror

    sys.modules["rich"] = old_rich
    sys.modules["rich.logging"] = old_rich_logging
