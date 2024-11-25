try:
    from juliapkg import add, resolve
    add("OPFGenerator", "5d2523b5-5e96-4b1c-8178-da2b93e9175f", url="https://github.com/AI4OPT/OPFGenerator")
    add("Ipopt", "b6b21f68-93f8-5de0-b562-5493be1d77c9")
    add("HiGHS", "87dc4568-4c63-4d18-b0c0-bb2238e4078b")
    add("Clarabel", "61c947e1-3e6d-4ee4-985a-eec8c727bd6e")
    add("DLPack", "53c2dc0f-f7d5-43fd-8906-6c0220547083")
    add("Zygote", "e88e6eb3-aa80-5325-afca-941959d7151f")
    resolve()
    from juliafunction import JuliaFunction
    from juliacall import Main
except ImportError as e:
    raise ImportError("The projection layer depends on juliafunction. Please install it by running \n\tpip install git+https://github.com/klamike/juliafunction") from e

from ml4opf.formulations.problem import OPFProblem
from ml4opf.formulations.ac.problem import ACProblem
from ml4opf.formulations.dc.problem import DCProblem
from ml4opf.formulations.soc.problem import SOCProblem
from ml4opf import warn, __path__ as ml4opf_path


def make_projection_function(problem: OPFProblem):
    """Create a projectionn layer for the given OPF problem.
    
    The following configurations are currently supported:

        - ACProblem: input (pd, qd) with outputs (pg, qg, vm, va)

        - DCProblem: input (pd) with outputs (pg, pf, va)

        - SOCProblem: input (pd, qd) with outputs (pg, qg, w, wr, wi)


    Note only one case can be loaded at a time (but different formulations for the same case are okay).

    To parallelize batch samples using Julia threading,
    set `PYTHON_JULIACALL_THREADS` and `PYTHON_JULIACALL_HANDLE_SIGNALS="yes"`.

    Refer to the JuliaCall documentation on multi-threading in Julia for more details.
    """
    warn(
        """The projection layer is an experimental feature.\n"""
        """The following configurations are currently supported: \n"""
        """- ACProblem: input (pd, qd) with outputs (pg, qg, vm, va) \n"""
        """- DCProblem: input (pd) with outputs (pg, pf, va) \n"""
        """- SOCProblem: input (pd, qd) with outputs (pg, qg, w, wr, wi) \n"""
    )
    Main.seval(f'using OPFGenerator; const PM = OPFGenerator.PowerModels')
    Main.seval(f'config = nothing; base_data = nothing')

    if "pglib_case" not in problem.case_data["config"] and "case_file" not in problem.case_data["config"]:
        problem.case_data["config"]["pglib_case"] = problem.case_data["config"]["ref"]

    Main.config = problem.case_data["config"]
    Main.seval("base_data = config |> OPFGenerator._get_case_info |> first |> PM.parse_file |> PM.make_basic_network |> OPFGenerator.OPFData")
    if isinstance(problem, ACProblem):
        return JuliaFunction(
            forward="ac_projection",
            backward="no_backward",
            batch_dims=[0] * 6,
            include=f"{ml4opf_path[0]}/layers/projection.jl",
            setup_code="base_data = Main.base_data",
        )
    elif isinstance(problem, DCProblem):
        return JuliaFunction(
            forward="dc_projection",
            backward="no_backward",
            batch_dims=[0] * 4,
            include=f"{ml4opf_path[0]}/layers/projection.jl",
            setup_code="base_data = Main.base_data",
        )
    elif isinstance(problem, SOCProblem):
        return JuliaFunction(
            forward="soc_projection",
            backward="no_backward",
            batch_dims=[0] * 7,
            include=f"{ml4opf_path[0]}/layers/projection.jl",
            setup_code="base_data = Main.base_data",
        )
    else:
        raise ValueError(f"Unsupported problem type: {type(problem)}")