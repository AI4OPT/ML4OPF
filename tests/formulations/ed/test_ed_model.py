import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path
from ml4opf.formulations.ed.problem import EDProblem

from ml4opf.formulations.ed.model import PerfectEDModel

@pytest.mark.skip(reason="Not implemented yet")
def test_ed_model():
    pass