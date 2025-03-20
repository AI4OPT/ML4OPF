using OPFGenerator
using Ipopt, HiGHS, Clarabel

const PM = OPFGenerator.PowerModels
const JuMP = OPFGenerator.JuMP
const MOI = OPFGenerator.JuMP.MOI


function projection(
    OPF::Type{<:OPFGenerator.AbstractFormulation},
    data::OPFGenerator.OPFData,
    solution,
    optimizer
)
    opf = OPFGenerator.build_opf(OPF, data, optimizer)

    obj = 0.0
    for v in VARS_MAP[OPF]
        if haskey(solution, v)
            obj += sum((solution[v] .- opf.model[v]) .^ 2)
        end
    end

    JuMP.set_objective(opf.model, MOI.MIN_SENSE, obj)

    OPFGenerator.solve!(opf)

    d = OPFGenerator.extract_result(opf)

    return Tuple([
        d["primal"][string(v)] for v in VARS_MAP[OPF]
    ]), nothing
end

VARS_MAP = Dict{Type{<:OPFGenerator.AbstractFormulation}, Vector{Symbol}}(
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


function no_backward(args...)
    throw(ArgumentError("Projection layer does not support backpropagation."))
end