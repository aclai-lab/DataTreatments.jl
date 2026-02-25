module DataTreatments

using Reexport

using InMemoryDatasets

using Statistics
using StatsBase
using LinearAlgebra
using DataFrames
using Catch22

using Normalization
@reexport using Normalization: AbstractNormalization, fit!, fit, normalize!, normalize

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataTreatment end
abstract type AbstractDataFeature end
abstract type AbstractMetaData end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const NameTypes = Union{Symbol, String}

# ---------------------------------------------------------------------------- #
#                                   files                                      #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
       trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
       forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
       stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq,
       transition_variance, periodicity, base_set, catch9, catch22_set, complete_set
include("featureset.jl")

export movingwindow, wholewindow, splitwindow, adaptivewindow
export @evalwindow
include("windowing.jl")

export is_multidim_dataset, nvals, convert
export has_uniform_element_size
include("treatment.jl")

include("ds_builder.jl")

# using Normalization: HalfZScore, MinMax, halfstd, zscore, center
using Normalization: dimparams, negdims, forward, estimators, normalization
import Normalization: @_Normalization, ZScore, Center
import Normalization: fit!, fit, normalize!, normalize, __mapdims!

using Statistics: mean, median, std
using StatsBase: mad, iqr
using LinearAlgebra: norm

export ZScore, MinMax, Scale, Center, Sigmoid, UnitEnergy, UnitPower, PNorm
include("normalize.jl")

# ---------------------------------------------------------------------------- #
#                                DataFeature                                   #
# ---------------------------------------------------------------------------- #
struct TabularFeat{T} <: AbstractDataFeature
    type::DataType
    vname::Symbol
end

struct AggregateFeat{T} <: AbstractDataFeature
    type::DataType
    vname::Symbol
    feat::Base.Callable
    nwin::Int64
end

struct ReduceFeat{T} <: AbstractDataFeature
    type::DataType
    vname::Symbol
    reducefunc::Base.Callable
end

# getters
get_type(f::AbstractDataFeature) = f.type
get_vname(f::AbstractDataFeature) = f.vname

get_feat(f::AggregateFeat)  = f.feat
get_nwin(f::AggregateFeat)  = f.nwin

get_reducefunc(f::ReduceFeat) = f.reducefunc

# ---------------------------------------------------------------------------- #
#                                  MetaData                                    #
# # ---------------------------------------------------------------------------- #
struct MetaData <: AbstractMetaData
    groups::Vector{Vector{Int64}}
    # method::Vector{Symbol}
end

# ---------------------------------------------------------------------------- #
#                                DataTreatment                                 #
# ---------------------------------------------------------------------------- #
struct DataTreatment <: AbstractDataTreatment
    X::Dataset
    y::Vector
    datafeature::Vector{<:AbstractDataFeature}
    metadata::MetaData

    function DataTreatment(
        X::Dataset,
        y::Union{AbstractVector,Nothing}=nothing;
        vnames::Union{Vector{String},Vector{Symbol},Nothing}=nothing,
        norm::Union{NormSpec,Type{<:AbstractNormalization},Nothing}=nothing,
        groups::Union{Symbol, Tuple{Vararg{Symbol}}, Vector{Vector{Symbol}}}=:vname,
        kwargs...
    )
        # se non è nothing, verifica la lunghezza di y
        isnothing(y) ? (y = Vector{Nothing}(nothing, size(X, 1))) : size(X, 1) != length(y) &&
            throw(DimensionMismatch("y length ($(length(y))) must match X rows ($(size(X, 1)))"))

        # verifica il contenuto di vnames
        isnothing(vnames) && (vnames = [Symbol("V$i") for i in 1:size(X, 2)])
        vnames isa Vector{String} && (vnames = Symbol.(vnames))

        X, features = build_dataset(X; vnames, kwargs...)

        # se non è specificato un raggruppamento, allora verrà utilizzato di default
        # il raggruppamento per nome
        # che in caso tabulare, significa che la normalizzazione avverrà columnwise
        # gestire il caso in cui la normalizzazione debba essere:
        # - tutto il dataset :all
        # - speciale: bitvector, oppure scelta manuale colonne
        fields = groups isa Symbol ? [groups] : collect(groups)
        groupidxs, _ = _groupby(features, fields)

        # if !isnothing(norm)
        #     norm isa Type{<:AbstractNormalization} && (norm = norm())

        #     Threads.@threads for g in groupidxs
        #         X[:, g] =
        #             normalize(X[:, g], norm)
        #     end
        # end

        metadata = MetaData(groupidxs)

        new{eltype(X),eltype(y)}(X, y, features, metadata)
    end

    DataTreatment(X::AbstractDataFrame, args...; kwargs...) =
        DataTreatment(Dataset(X), args...; vnames=propertynames(X), kwargs...)
end

# value access methods
Base.getproperty(dt::DataTreatment, s::Symbol) = getfield(dt, s)
Base.propertynames(::DataTreatment) =
    (:dataset, :datafeature)

get_dataset(dt::DataTreatment) = dt.dataset
get_datafeature(dt::DataTreatment) = dt.datafeature
get_reducefunc(dt::DataTreatment) = dt.reducefunc
get_aggrtype(dt::DataTreatment) = dt.aggrtype
get_groups(dt::DataTreatment) = dt.groups
get_norm(dt::DataTreatment) = dt.norm

# Convenience methods for common operations
get_vnames(dt::DataTreatment) = unique(get_vname.(dt.datafeature))
get_features(dt::DataTreatment) = unique(get_feat.(dt.datafeature))
get_nwindows(dt::DataTreatment) = maximum(get_nwin.(dt.datafeature))
get_reducefuncs(dt::DataTreatment) = unique(get_reducefunc.(dt.datafeature))

# Size and iteration methods
Base.size(dt::DataTreatment) = size(dt.dataset)
Base.size(dt::DataTreatment, dim::Int) = size(dt.dataset, dim)
Base.length(dt::DataTreatment) = length(dt.datafeature)
Base.eltype(dt::DataTreatment) = eltype(dt.dataset)

# Indexing methods
Base.getindex(dt::DataTreatment, i::Int) = dt.dataset[:, i]
Base.getindex(dt::DataTreatment, i::Int, j::Int) = dt.dataset[i, j]
Base.getindex(dt::DataTreatment, ::Colon, j::Int) = dt.dataset[:, j]
Base.getindex(dt::DataTreatment, i::Int, ::Colon) = dt.dataset[i, :]
Base.getindex(dt::DataTreatment, I...) = dt.dataset[I...]

export DataTreatment
export get_vname, get_feat, get_nwin
export get_vnames, get_features, get_nwindows, get_reducefuncs
export get_dataset, get_datafeature, get_reducefunc, get_aggrtype
export get_groups, get_norm, get_normdims

export groupby
include("groupby.jl")

end
