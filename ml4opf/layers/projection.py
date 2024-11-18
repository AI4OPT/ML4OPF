from uuid import uuid4

from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.ac.problem import ACProblem
from ml4opf.formulations.dc.problem import DCProblem
from ml4opf.formulations.soc.problem import SOCProblem
from ml4opf import warn, __path__ as ml4opf_path


try:
    from juliapkg import add, resolve
except ImportError as e:
    raise ImportError("The projection layer depends on juliacall. Please install it by running \n\tpip install juliacall") from e

# TODO: integrate with DiffOpt+ParametricOptInterface once POI#143 is merged
class Projection:
    """Projection onto the feasible set. Implemented using juliacall and AI4OPT/OPFGenerator."""
    
    initialized: bool = False

    def __init__(self, problem: OPFProblem):
        self.problem = problem
        self.setup()

        [warn("The projection layer is experimental! Proceed with caution.") for _ in range(3)]

    def setup(self):
        from juliacall import Main
        self.Module = Main.seval(f"module ML4OPF_Projection{uuid4()} end")

        if not Projection.initialized:
            add("OPFGenerator", "5d2523b5-5e96-4b1c-8178-da2b93e9175f", url="https://github.com/AI4OPT/OPFGenerator")
            add("MathOptInterface", "b8f27783-ece8-5eb3-8dc8-9495eed66fee")
            add("JuMP", "4076af6c-e467-56ae-b986-b466b2749572")
            add("PowerModels", "c36e90e8-916a-50a6-bd94-075b64ef4655")
            add("Ipopt", "b6b21f68-93f8-5de0-b562-5493be1d77c9")
            add("HiGHS", "87dc4568-4c63-4d18-b0c0-bb2238e4078b")
            add("Clarabel", "61c947e1-3e6d-4ee4-985a-eec8c727bd6e")
            resolve()
            Projection.initialized = True

        self.Module.seval(f'include("{ml4opf_path[0]}/layers/projection.jl")')

        self.Module.config = self.problem.case_data["config"]
        self.Module.seval("base_data = make_base_data(config)")

        if isinstance(self.problem, ACProblem):
            self.project = self.Module.ac_projection
        elif isinstance(self.problem, DCProblem):
            self.project = self.Module.dc_projection
        elif isinstance(self.problem, SOCProblem):
            self.project = self.Module.soc_projection

    def __call__(self, *args):
        return self.project(*args)

