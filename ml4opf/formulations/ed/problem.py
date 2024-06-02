import torch

from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.ed.violation import EDViolation

from ml4opf.parsers.read_hdf5 import parse_hdf5


class EDProblem(OPFProblem):

    def __init__(self, data_directory: str, case_name: str, ptdf_path: str, dataset_name: str = "ED", **parse_kwargs):
        self.ptdf_path = ptdf_path
        super().__init__(data_directory, case_name, dataset_name, **parse_kwargs)

    def parse(self, *args, **kwargs):
        D = parse_hdf5(self.ptdf_path, preserve_shape=True)
        assert D.keys() == {"PTDF"}
        self.ptdf = torch.from_numpy(D["PTDF"]).swapaxes(0, 1)

        super().parse(*args, **kwargs)

    def _parse_sanity_check(self):
        # super()._parse_sanity_check()
        assert (
            len(set((self.json_data["n_gen"], self.train_data["primal/pg"].shape[1]))) == 1
        ), "Number of generators in JSON and HDF5 files do not match."

        assert (
            len(set((self.json_data["n_load"], self.train_data["input/pd"].shape[1]))) == 1
        ), "Number of loads in JSON and HDF5 files do not match."

        assert (
            self.ptdf.shape[0] == self.json_data["n_branch"]
        ), "Number of branches in PTDF and JSON files do not match."

        assert self.ptdf.shape[1] == self.json_data["n_bus"], "Number of buses in PTDF and JSON files do not match."

    @property
    def feasibility_check(self) -> dict[str, str]:
        """Default feasibility check for EconomicDispatch."""
        return {
            "meta/termination_status": "OPTIMAL",
            "meta/primal_status": "FEASIBLE_POINT",
            "meta/dual_status": "FEASIBLE_POINT",
        }

    @property
    def default_combos(self) -> dict[str, list[str]]:
        """Default combos for EconomicDispatch. input: pd, target: pg, va"""
        return {
            "input": ["input/pd"],
            "target": [
                "primal/pg",
            ],
        }

    @property
    def default_order(self) -> list[str]:
        """Default order for EconomicDispatch. input, target"""
        return ["input", "target"]

    @property
    def violation(self) -> EDViolation:
        """OPFViolation object for EconomicDispatch constraint calculations."""
        if not hasattr(self, "_violation"):
            self._violation = EDViolation(self.json_data, self.ptdf)
        return self._violation
