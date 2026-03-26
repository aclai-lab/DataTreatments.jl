# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
mutable struct DataTreatment{T}
    data::Vector{AbstractDataset}
    target::Union{Nothing,AbstractVector}
    levels::Union{Nothing,AbstractVector}
    treats::Vector{TreatmentGroup}
end

nrows(dt::DataTreatment) = size(first(dt.data).data, 1)

get_levels(dt::DataTreatment) = dt.levels
get_target(dt::DataTreatment) = dt.target

function get_discrete(
    dt::DataTreatment
)::Tuple{Matrix{Union{Missing,Int64}}, Vector{String}}
    ds = collect(filter(d -> d isa DiscreteDataset, dt.data))
    return if isempty(ds)
        Matrix{Union{Missing,Int64}}(undef, 0, 0), String[]
    else
        Matrix{Union{Missing,Int64}}(get_data(ds)),
        reduce(vcat, get_vnames.(ds))
    end
end

function get_continuous(
    dt::DataTreatment{T}
)::Tuple{Matrix{Union{Missing,T}}, Vector{String}} where {T<:AbstractFloat}
    ds = collect(filter(d -> d isa ContinuousDataset{T}, dt.data))
    return if isempty(ds)
        Matrix{Union{Missing,T}}(undef, 0, 0), String[]
    else
        Matrix{Union{Missing,T}}(get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

function get_aggregated(
    dt::DataTreatment{T}
)::Tuple{Matrix{Union{Missing,T}}, Vector{String}} where {T<:AbstractFloat}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa AggregateFeat{T}, get_info(d)), dt.data)
    return if isempty(ds)
        Matrix{Union{Missing,T}}(undef, 0, 0), String[]
    else
        Matrix{Union{Missing,T}}(get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

function get_reduced(
    dt::DataTreatment{T}
)::Tuple{Matrix{Union{Missing,T,AbstractArray{T}}}, Vector{String}} where {T<:AbstractFloat}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa ReduceFeat, get_info(d)), dt.data)
    return if isempty(ds)
        Matrix{Union{Missing,T,AbstractArray{T}}}(undef, 0, 0), String[]
    else
        Matrix{Union{Missing,T,AbstractArray{T}}}(get_data(ds)), reduce(vcat, get_vnames.(ds))
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
    leftover_ds::Bool=true,
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

    return DataTreatment{float_type}(ds, ctarget, clevels, treats)
end

load_dataset(df::DataFrame, target::AbstractVector, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), target, args...; kwargs...)

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), nothing, args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                             get tabular method                               #
# ---------------------------------------------------------------------------- #
"""
    get_tabular(dt::DataTreatment)

Convenience function to collect all tabular-like datasets from a `DataTreatment` 
object, including discrete, continuous, and aggregated multidimensional data.
"""
# @inline get_tabular(dt::DataTreatment)::Matrix{Union{Missing, Float64}} = begin
#     mats = [get_discrete(dt), get_continuous(dt), get_aggregated(dt)]
#     nonempty = filter(x -> !(isempty(x) || size(x,2) == 0), mats)
#     isempty(nonempty) ? Matrix{Union{Missing, Float64}}(undef, nrows(dt), 0) : hcat(nonempty...)
# end

# @inline function get_tabular(dt::DataTreatment)
#     mats = [get_discrete(dt), get_continuous(dt), get_aggregated(dt)]
#     (firsts, seconds) = (map(first, mats), map(last, mats))
#     nonempty = filter(x -> !(isempty(x) || size(x,2) == 0), firsts)
#     isempty(nonempty) ? Matrix{Union{Missing, Float64}}(undef, nrows(dt), 0) : hcat(nonempty...)
# end
@inline function get_tabular(
    dt::DataTreatment{T}
)::Tuple{Matrix{Union{Missing,T}}, Vector{String}} where {T<:AbstractFloat}
    mats = get_discrete(dt), get_continuous(dt), get_aggregated(dt)
    idxs = findall(x -> !(isempty(x)), map(first, mats))
    not_empty = collect(zip(mats[idxs]...))

    return reduce(hcat, not_empty[1]), reduce(vcat, not_empty[2])
end

# @inline function get_tabular(
#     dt::DataTreatment{T}
# )::Tuple{Matrix{Union{Missing,T}}, Vector{String}} where {T<:AbstractFloat}
#     disc_mat, disc_names = get_discrete(dt)
#     cont_mat, cont_names = get_continuous(dt)
#     aggr_mat, aggr_names = get_aggregated(dt)

#     mats   = Matrix{Union{Missing,T}}[]
#     vnames = String[]

#     !isempty(disc_mat) && (push!(mats, Matrix{Union{Missing,T}}(disc_mat)); append!(vnames, disc_names))
#     !isempty(cont_mat) && (push!(mats, cont_mat); append!(vnames, cont_names))
#     !isempty(aggr_mat) && (push!(mats, aggr_mat); append!(vnames, aggr_names))

#     isempty(mats) && return Matrix{Union{Missing,T}}(undef, nrows(dt), 0), String[]
#     return reduce(hcat, mats), vnames
# end
# ---------------------------------------------------------------------------- #
#                            get multidim method                               #
# ---------------------------------------------------------------------------- #
"""
    get_multidim(dt::DataTreatment)

Convenience function to collect all reduced multidimensional datasets 
from a `DataTreatment` object.
"""
@inline function get_multidim(
    dt::DataTreatment{T}
)::Tuple{Matrix{Union{Missing,AbstractArray{T}}}, Vector{String}} where {T<:AbstractFloat}
    return get_reduced(dt)
end
