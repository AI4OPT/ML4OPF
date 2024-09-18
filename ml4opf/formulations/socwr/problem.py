from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.socwr.violation import SOCWRViolation


class SOCWRProblem(OPFProblem):
    """`OPFProblem` for SOCWRPowerModel/SOCOPF"""

    def __init__(self, data_directory: str, case_name: str, dataset_name: str = "SOCOPF", **parse_kwargs):
        super().__init__(data_directory, case_name, dataset_name, **parse_kwargs)

    def _parse_sanity_check(self):
        super()._parse_sanity_check()
        json_n_bus = self.json_data["n_bus"]
        json_n_gen = self.json_data["n_gen"]
        json_n_load = self.json_data["n_load"]
        train_n_w = self.train_data["primal/w"].shape[1]
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

    @property
    def feasibility_check(self) -> dict[str, str]:
        """Default feasibility check for SOCOPF:

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
        """Default combos for SOCOPF:

        - input: pd, qd

        - target: pg, qg, vm, va
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
    def violation(self) -> SOCWRViolation:
        """`SOCWRViolation` object, created upon first access."""
        if not hasattr(self, "_violation"):
            self._violation = SOCWRViolation(self.json_data)
        return self._violation
