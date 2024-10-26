""" DCOPF Problem data class """

from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.dc.violation import DCViolation


class DCProblem(OPFProblem):
    """`OPFProblem` for DCOPF"""

    def __init__(self, data_directory: str, dataset_name: str = "DCOPF", **parse_kwargs):
        super().__init__(data_directory, dataset_name, **parse_kwargs)

    def _parse_sanity_check(self):
        super()._parse_sanity_check()
        assert (
            len(set((self.case_data["N"].item(), self.train_data["primal/va"].shape[1]))) == 1
        ), "Number of buses in JSON and HDF5 files do not match."

        assert (
            len(set((self.case_data["G"].item(), self.train_data["primal/pg"].shape[1]))) == 1
        ), "Number of generators in JSON and HDF5 files do not match."

        assert (
            len(set((self.case_data["L"].item(), self.train_data["input/pd"].shape[1]))) == 1
        ), "Number of loads in JSON and HDF5 files do not match."

    @property
    def feasibility_check(self) -> dict[str, str]:
        """Default feasibility check for DCOPF:

        - termination_status: "OPTIMAL"

        - primal_status: "FEASIBLE_POINT"

        - dual_status: "FEASIBLE_POINT"
        """
        return {
            "meta/termination_status": "OPTIMAL",
            "meta/primal_status": "FEASIBLE_POINT",
            "meta/dual_status": "FEASIBLE_POINT",
        }

    @property
    def default_combos(self) -> dict[str, list[str]]:
        """Default combos for DCOPF:

        - input: pd

        - target: pg, va
        """
        return {
            "input": ["input/pd"],
            "target": ["primal/pg", "primal/va"],
        }

    @property
    def default_order(self) -> list[str]:
        """Default order for DCOPF. input, target"""
        return ["input", "target"]

    @property
    def violation(self) -> DCViolation:
        """OPFViolation object for DCOPF constraint calculations."""
        if not hasattr(self, "_violation"):
            self._violation = DCViolation(self.case_data)
        return self._violation
