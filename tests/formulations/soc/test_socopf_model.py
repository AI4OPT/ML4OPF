import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.soc.problem import SOCProblem

from ml4opf.formulations.soc.model import PerfectSOCModel

@pytest.mark.skip(reason="Not implemented yet")
def test_soc_model():
    pass