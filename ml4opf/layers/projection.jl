using OPFGenerator, JuMP, MathOptInterface, PowerModels
using Ipopt, HiGHS, Clarabel, LinearAlgebra

const PM = PowerModels


function projection(
    OPF::Type{<:OPFGenerator.AbstractFormulation},
    data::OPFGenerator.OPFData,
    solution,
    optimizer
) where T
    opf = OPFGenerator.build_opf(OPF, data, optimizer)

    obj = 0.0
    for v in vars_map[OPF]
        if haskey(solution, v)
            # obj += norm(solution[v] .- opf.model[v], p=2)
            obj += sum((solution[v] .- opf.model[v]) .^ 2)
        end
    end

    JuMP.set_objective(opf.model, JuMP.MOI.MIN_SENSE, obj)

    OPFGenerator.solve!(opf)

    d = OPFGenerator.extract_result(opf)

    return d
end


config = nothing
base_data = nothing


vars_map = Dict{Type{<:OPFGenerator.AbstractFormulation}, Vector{Symbol}}(
    OPFGenerator.ACOPF => [:pg, :qg, :pf, :pt, :qf, :qt, :vm, :va],
    OPFGenerator.DCOPF => [:pg, :pf, :va],
    OPFGenerator.SOCOPF => [:pg, :qg, :pf, :pt, :qf, :qt, :w, :wr, :wi],
)


function ac_projection(pd, qd, pg, qg, vm, va)
    data = deepcopy(base_data)
    data.pd .= pd
    data.qd .= qd
    return projection(
        OPFGenerator.ACOPF,
        data,
        Dict(
            :pg => pg,
            :qg => qg,
            :vm => vm,
            :va => va
        ),
        Ipopt.Optimizer
    )
end


function dc_projection(pd, pg, pf, va)
    data = deepcopy(base_data)
    data.pd .= pd
    return projection(
        OPFGenerator.DCOPF,
        data,
        Dict(
            :pg => pg,
            :pf => pf,
            :va => va
        ),
        HiGHS.Optimizer
    )
end


function soc_projection(pd, qd, pg, qg, w, wr, wi)
    data = deepcopy(base_data)
    data.pd .= pd
    data.qd .= qd
    return projection(
        OPFGenerator.SOCOPF,
        data,
        Dict(
            :pg => pg,
            :qg => qg,
            :w => w,
            :wr => wr,
            :wi => wi
        ),
        Clarabel.Optimizer
    )
end


function make_base_data(config)
    # case_file = config |> OPFGenerator._get_case_name |> first  # once OPFGenerator#141 is merged
    case_file = joinpath(OPFGenerator.PGLib.PGLib_opf, OPFGenerator.PGLib.find_pglib_case(config["ref"])[1])
    return case_file |> PM.parse_file |> PM.make_basic_network |> OPFGenerator.OPFData
end
