""" SOCOPF Problem data class """

from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.soc.violation import SOCViolation


class SOCProblem(OPFProblem):
    """`OPFProblem` for SOCOPF"""

    def __init__(self, data_directory: str, dataset_name: str = "SOCOPF", **parse_kwargs):
        super().__init__(data_directory, dataset_name, **parse_kwargs)

    def _parse_sanity_check(self):
        super()._parse_sanity_check()
        json_n_bus = self.case_data["N"].item()
        json_n_gen = self.case_data["G"].item()
        json_n_load = self.case_data["L"].item()
        json_n_branch = self.case_data["E"].item()
        train_n_w = self.train_data["primal/w"].shape[1]
        train_n_wr = self.train_data["primal/wr"].shape[1]
        train_n_wi = self.train_data["primal/wi"].shape[1]
        train_n_pg = self.train_data["primal/pg"].shape[1]
        train_n_qg = self.train_data["primal/qg"].shape[1]
        train_n_pd = self.train_data["input/pd"].shape[1]
        train_n_qd = self.train_data["input/qd"].shape[1]

        assert (
            len(set((json_n_bus, train_n_w))) == 1
        ), f"Number of buses in JSON and HDF5 files do not match. Got {json_n_bus=}, {train_n_w=}"

        assert (
            len(set((json_n_gen, train_n_pg, train_n_qg))) == 1
        ), f"Number of generators in JSON and HDF5 files do not match. Got {json_n_gen=}, {train_n_pg=}, {train_n_qg=}"

        assert (
            len(set((json_n_load, train_n_pd, train_n_qd))) == 1
        ), f"Number of loads in JSON and HDF5 files do not match. Got {json_n_load=}, {train_n_pd=}, {train_n_qd=}"

        assert (
            len(set((json_n_branch, train_n_wr, train_n_wi))) == 1
        ), f"Number of branches in JSON and HDF5 files do not match. Got {json_n_branch=}, {train_n_wr=}, {train_n_wi=}"

    @property
    def feasibility_check(self) -> dict[str, str]:
        """Default feasibility check for SOCOPF:

        - termination_status: "LOCALLY_SOLVED"

        - primal_status: "FEASIBLE_POINT"

        - dual_status: "FEASIBLE_POINT"
        """
        return {
            "meta/termination_status": "LOCALLY_SOLVED",
            "meta/primal_status": "FEASIBLE_POINT",
            "meta/dual_status": "FEASIBLE_POINT",
        }

    @property
    def default_combos(self) -> dict[str, list[str]]:
        """Default combos for SOCOPF:

        - input: pd, qd

        - target: pg, qg, w, wr, wi
        """
        return {
            "input": ["input/pd", "input/qd"],
            "target": ["primal/pg", "primal/qg", "primal/w", "primal/wr", "primal/wi"],
        }

    @property
    def default_order(self) -> list[str]:
        """Default order for SOCOPF: input, target"""
        return ["input", "target"]

    @property
    def violation(self) -> SOCViolation:
        """`SOCViolation` object, created upon first access."""
        if not hasattr(self, "_violation"):
            self._violation = SOCViolation(self.case_data)
        return self._violation
