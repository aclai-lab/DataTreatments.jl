# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
mutable struct DataTreatment{T}
    data::Vector{AbstractDataset}
    target::AbstractVector
    treats::Vector{TreatmentGroup}
end

nrows(dt::DataTreatment) = size(first(dt.data).data, 1)

get_target(dt::DataTreatment) = dt.target

function get_discrete(
    dt::DataTreatment
)
    ds = collect(filter(d -> d isa DiscreteDataset, dt.data))
    return if isempty(ds)
        Matrix{Union{Missing,CategoricalValue}}(undef, 0, 0), String[]
    else
        data = get_data(ds)
        Matrix{eltype(data)}(data),
        reduce(vcat, get_vnames.(ds))
    end
end

function get_continuous(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = collect(filter(d -> d isa ContinuousDataset{T}, dt.data))
    return if isempty(ds)
        Matrix{Union{Missing,T}}(undef, 0, 0), String[]
    else
        Matrix{Union{Missing,T}}(get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

function get_aggregated(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa AggregateFeat{T}, get_info(d)), dt.data)
    return if isempty(ds)
        Matrix{Union{Missing,T}}(undef, 0, 0), String[]
    else
        Matrix{Union{Missing,T}}(get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

function get_reduced(
    dt::DataTreatment{T};
    force_type::Bool=false,
) where {T<:Float}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa ReduceFeat, get_info(d)), dt.data)
    return if isempty(ds)
        force_type ?
        (Matrix{VecOrMat{T}}(undef, 0, 0), String[]) :
        (Matrix{Union{Missing,T,VecOrMat{T}}}(undef, 0, 0), String[])
    else
        force_type ?
        (Matrix{VecOrMat{T}}(get_data(ds)),
            reduce(vcat, get_vnames.(ds))) :
        (Matrix{Union{Missing,T,VecOrMat{T}}}(get_data(ds)),
            reduce(vcat, get_vnames.(ds)))
    end
end

is_tabular(dt::DataTreatment) = all(is_tabular.(dt.data))
is_multidim(dt::DataTreatment) = all(is_multidim.(dt.data))

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
function load_dataset(
    data::Matrix,
    vnames::Vector{String}=["V$i" for i in 1:size(data, 2)],
    target::Union{Nothing,AbstractVector}=nothing,
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    treatment_ds::Bool=true,
    leftover_ds::Bool=false,
    float_type::Type=Float64
)
    datastruct = _inspecting(data)

    if isnothing(target)
        target = CategoricalVector[]
    elseif !isnothing(target) && !(eltype(target) <: AbstractFloat)
        target = _discrete_encode(target)
    end

    treats = [treat(datastruct, vnames) for treat in treatments]

    ds = AbstractDataset[]

    if treatment_ds
        ds_td, ds_tc, ds_md = _treatments_ds(
            data,
            vnames,
            datastruct,
            treats, 
            float_type
        )
        !isempty(ds_td) && append!(ds, ds_td)
        !isempty(ds_tc) && append!(ds, ds_tc)
        !isempty(ds_md) && append!(ds, ds_md)
    end

    if leftover_ds
        ds_td, ds_tc, ds_md = _leftovers_ds(
            data,
            vnames,
            datastruct,
            treats,
            float_type
        )
        !isempty(ds_td) && append!(ds, ds_td)
        !isempty(ds_tc) && append!(ds, ds_tc)
        !isempty(ds_md) && append!(ds, ds_md)
    end

    # return DataTreatment{float_type}(ds, ctarget, clevels, treats)
    return DataTreatment{float_type}(ds, target, treats)
end

load_dataset(df::DataFrame, target::AbstractVector, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), target, args...; kwargs...)

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), CategoricalVector[], args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                             get tabular method                               #
# ---------------------------------------------------------------------------- #
"""
    get_tabular(dt::DataTreatment)

Convenience function to collect all tabular-like datasets from a `DataTreatment` 
object, including discrete, continuous, and aggregated multidimensional data.
"""
@inline function get_tabular(
    dt::DataTreatment{T};
    force_type::Bool=false
) where {T<:Float}
    mats = get_discrete(dt), get_continuous(dt), get_aggregated(dt)
    idxs = findall(x -> !(isempty(x)), map(first, mats))

    isempty(idxs) && return force_type ?
        (Matrix{T}(undef, 0,0), String[]) :
        (Matrix{Union{Missing,T}}(undef, 0,0), String[])
    not_empty = collect(zip(mats[idxs]...))

    X = if force_type
        Matrix{Union{CategoricalValue,Int,T}}(reduce(hcat, not_empty[1]))
    else
        Matrix{Union{Missing,CategoricalValue,Int,T}}(reduce(hcat, not_empty[1]))
    end

    return X, reduce(vcat, not_empty[2])
end

# ---------------------------------------------------------------------------- #
#                            get multidim method                               #
# ---------------------------------------------------------------------------- #
"""
    get_multidim(dt::DataTreatment)

Convenience function to collect all reduced multidimensional datasets 
from a `DataTreatment` object.
"""
@inline function get_multidim(
    dt::DataTreatment{T};
    kwargs...
) where {T<:Float}
    return get_reduced(dt; kwargs...)
end
