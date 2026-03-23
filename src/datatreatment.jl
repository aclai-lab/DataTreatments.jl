# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
mutable struct DataTreatment
    data::Vector{AbstractDataset}
    target::Union{Nothing,AbstractVector}
    levels::Union{Nothing,AbstractVector}
    treats::Vector{TreatmentGroup}
end

function load_dataset(
    data::Matrix,
    vnames::Vector{String}=["V$i" for i in 1:size(data, 2)],
    target::Union{Nothing,AbstractVector}=nothing,
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    treatment_ds::Bool=true,
    leftover_ds::Bool=true,
    # data_type::Base.Callable=tabular,
    float_type::Type=Float64
)
    datastruct = _inspecting(data)

    ctarget, clevels = if !isnothing(target) && !(eltype(target) <: AbstractFloat)
        _discrete_encode(target)
    else
        target, nothing
    end

    treats = [treat(datastruct, vnames) for treat in treatments]

    ds = AbstractDataset[]

    if treatment_ds
        ds_td, ds_tc, ds_md = _treatments_ds(data, vnames, datastruct, treats, float_type)
        !isnothing(ds_td) && append!(ds, ds_td)
        !isnothing(ds_tc) && append!(ds, ds_tc)
        !isnothing(ds_md) && append!(ds, ds_md)
    end

    if leftover_ds
        ds_td, ds_tc, ds_md = _leftovers_ds(data, vnames, datastruct, treats, float_type)
        !isnothing(ds_td) && append!(ds, ds_td)
        !isnothing(ds_tc) && append!(ds, ds_tc)
        !isnothing(ds_md) && append!(ds, ds_md)
    end

    # data, treats = data_type(dt, args...; kwargs...)

    # data, treats = get_dataset(dt::DataTreatment, args...; kwargs...)
    
    # return vcat(get_discrete(data), get_continuous(data), get_aggregated(data)), treats

    return DataTreatment(ds, ctarget, clevels, treats)
end

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), args...; kwargs...)