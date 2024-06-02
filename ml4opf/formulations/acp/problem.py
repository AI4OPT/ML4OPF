import torch

from torch import Tensor

from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.acp.violation import ACPViolation


class ACPProblem(OPFProblem):
    """`OPFProblem` for ACPPowerModel/ACOPF"""

    def __init__(self, data_directory: str, case_name: str, dataset_name: str = "ACOPF", **parse_kwargs):
        super().__init__(data_directory, case_name, dataset_name, **parse_kwargs)

    def _parse_sanity_check(self):
        super()._parse_sanity_check()
        json_n_bus = self.json_data["n_bus"]
        json_n_gen = self.json_data["n_gen"]
        json_n_load = self.json_data["n_load"]
        train_n_vm = self.train_data["primal/vm"].shape[1]
        train_n_va = self.train_data["primal/va"].shape[1]
        train_n_pg = self.train_data["primal/pg"].shape[1]
        train_n_qg = self.train_data["primal/qg"].shape[1]
        train_n_pd = self.train_data["input/pd"].shape[1]
        train_n_qd = self.train_data["input/qd"].shape[1]

        assert (
            len(set((json_n_bus, train_n_vm, train_n_va))) == 1
        ), f"Number of buses in JSON and HDF5 files do not match. Got {json_n_bus=}, {train_n_vm=}, {train_n_va=}"

        assert (
            len(set((json_n_gen, train_n_pg, train_n_qg))) == 1
        ), f"Number of generators in JSON and HDF5 files do not match. Got {json_n_gen=}, {train_n_pg=}, {train_n_qg=}"

        assert (
            len(set((json_n_load, train_n_pd, train_n_qd))) == 1
        ), f"Number of loads in JSON and HDF5 files do not match. Got {json_n_load=}, {train_n_pd=}, {train_n_qd=}"

    @property
    def feasibility_check(self) -> dict[str, str]:
        """Default feasibility check for ACOPF:

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
        """Default combos for ACOPF:

        - input: pd, qd

        - target: pg, qg, vm, va
        """
        return {
            "input": ["input/pd", "input/qd"],
            "target": ["primal/pg", "primal/qg", "primal/vm", "primal/va"],
        }

    @property
    def default_order(self) -> list[str]:
        """Default order for ACOPF: input, target"""
        return ["input", "target"]

    @property
    def violation(self) -> ACPViolation:
        """`ACPViolation` object, created upon first access."""
        if not hasattr(self, "_violation"):
            self._violation = ACPViolation(self.json_data)
        return self._violation
