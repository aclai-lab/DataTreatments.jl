module DataTreatments

using Reexport

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
const Groups = Union{BitVector,Symbol,Vector{Symbol},Vector{Vector{Symbol}}}

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
    id::Int
    type::Type
    vname::Symbol
end

struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    type::Type
    vname::Symbol
    feat::Base.Callable
    nwin::Int
end

struct ReduceFeat{T} <: AbstractDataFeature
    id::Int
    type::Type
    vname::Symbol
    reducefunc::Base.Callable
end

# getters
get_id(f::AbstractDataFeature) = f.id
get_type(f::AbstractDataFeature) = f.type
get_vname(f::AbstractDataFeature) = f.vname

get_feat(f::AggregateFeat) = f.feat
get_nwin(f::AggregateFeat) = f.nwin

get_reducefunc(f::ReduceFeat) = f.reducefunc

# ---------------------------------------------------------------------------- #
#                                  MetaData                                    #
# # ---------------------------------------------------------------------------- #
struct MetaData <: AbstractMetaData
    norm::Union{NormSpec,Type{<:AbstractNormalization},Nothing}
    groupmethod::Groups
    groups::Vector{Base.Generator}
    # method::Vector{Symbol}
end

# ---------------------------------------------------------------------------- #
#                                DataTreatment                                 #
# ---------------------------------------------------------------------------- #
struct DataTreatment{T,S} <: AbstractDataTreatment
    X::Matrix{T}
    y::Vector{S}
    datafeature::Vector{<:AbstractDataFeature}
    metadata::MetaData

    function DataTreatment(
        X::Matrix{T},
        y::Union{AbstractVector,Nothing}=nothing;
        vnames::Union{Vector{String},Vector{Symbol},Nothing}=nothing,
        norm::Union{NormSpec,Type{<:AbstractNormalization},Nothing}=nothing,
        groups::Groups=:vname,
        kwargs...
    ) where T
        isnothing(y) ? (y = Vector{Nothing}(nothing, size(X, 1))) : size(X, 1) != length(y) &&
            throw(DimensionMismatch("y length ($(length(y))) must match X rows ($(size(X, 1)))"))

        isnothing(vnames) && (vnames = [Symbol("V$i") for i in 1:size(X, 2)])
        vnames isa Vector{String} && (vnames = Symbol.(vnames))

        X, features = build_dataset(X; vnames, kwargs...)

        # se non è specificato un raggruppamento, allora verrà utilizzato di default
        # il raggruppamento per nome
        # che in caso tabulare, significa che la normalizzazione avverrà columnwise
        # gestire il caso in cui la normalizzazione debba essere:
        # - tutto il dataset :all
        # - speciale: bitvector, oppure scelta manuale colonne
        # fields = groups isa Symbol ? [groups] : collect(groups)
        groupidxs = groupby(features, groups)

        if !isnothing(norm)
            flat_groups = [reduce(vcat, g) for g in groupidxs]
            Threads.@threads for g in flat_groups
                normalize!(@view(X[:, g]), norm)
            end
        end

        metadata = MetaData(norm, groups, groupidxs isa Base.Generator ? [groupidxs] : groupidxs)

        new{eltype(X),eltype(y)}(X, y, features, metadata)
    end

    DataTreatment(X::AbstractDataFrame, args...; kwargs...) =
        DataTreatment(reduce(hcat, vec.(eachcol(X))), args...; vnames=propertynames(X), kwargs...)
end

# ---------------------------------------------------------------------------- #
#                            DataTreatment methods                             #
# ---------------------------------------------------------------------------- #
# value access methods
Base.getproperty(dt::DataTreatment, s::Symbol) = getfield(dt, s)
Base.propertynames(::DataTreatment) =
    (:X, :y, :datafeature, :metadata)

get_X(dt::DataTreatment) = dt.X
get_y(dt::DataTreatment) = dt.y
get_datafeature(dt::DataTreatment) = dt.datafeature
get_metadata(dt::DataTreatment) = dt.metadata

# metadatas
get_groups(dt::DataTreatment) = reduce(vcat, collect.(dt.metadata.groups))
get_groupmethod(dt::DataTreatment) = dt.metadata.groupmethod
get_norm(dt::DataTreatment) = dt.metadata.norm

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

# ---------------------------------------------------------------------------- #
#                              DataTreatment show                              #
# ---------------------------------------------------------------------------- #
function _show_datatreatment(io::IO, dt::DataTreatment{T,S}) where {T,S}
    nrows, ncols = size(dt.X)

    feat_kinds = map(dt.datafeature) do f
        if f isa TabularFeat       "tabular"
        elseif f isa AggregateFeat "aggregate"
        elseif f isa ReduceFeat    "reduce"
        else string(typeof(f))
        end
    end
    feat_kind = length(unique(feat_kinds)) == 1 ? first(feat_kinds) : join(unique(feat_kinds), " + ")

    supervised = !(S <: Nothing)
    norm_str = isnothing(dt.metadata.norm) ? "none" : string(dt.metadata.norm)

    gm = dt.metadata.groupmethod
    gm_str = gm isa Symbol ? string(gm) :
             gm isa Vector{Symbol} ? join(string.(gm), ", ") :
             string(gm)

    green  = "\e[32m"
    yellow = "\e[33m"
    white  = "\e[37m"
    reset  = "\e[0m"

    # align '::' by padding the left part to the same width
    x_left = "X         $(nrows) × $(ncols)  "
    y_left = supervised ? "y         supervised  " : "y         unsupervised  "
    pad_width = max(length(x_left), length(y_left))
    x_line = rpad(x_left, pad_width) * "::  $(T)"
    y_line = supervised ? rpad(y_left, pad_width) * "::  $(S)" : rpad("y         unsupervised", pad_width)

    lines = Tuple{String,String}[
        (" ",  "$(yellow)DataTreatment$(reset)"),
        ("  ", "$(white)$(x_line)$(reset)"),
        ("  ", "$(white)$(y_line)$(reset)"),
        ("  ", "$(white)feature   $(feat_kind)$(reset)"),
        ("  ", "$(white)norm      $(norm_str)$(reset)"),
        ("  ", "$(white)group     $(gm_str)$(reset)"),
    ]

    visible_length(s) = length(replace(s, r"\e\[[0-9;]*m" => ""))

    full_lines = ["│$(prefix)$(content)" for (prefix, content) in lines]
    seps  = Set([1, 3])
    width = maximum(visible_length, full_lines) + 1

    hline(left, right) = green * left * "─"^(width - 1) * right * reset

    println(io, hline("┌", "┐"))
    for (i, line) in enumerate(full_lines)
        pad = width - visible_length(line)
        println(io, green * "│" * reset * line[4:end] * " "^pad * green * "│" * reset)
        i in seps && println(io, hline("├", "┤"))
    end
    print(io, hline("└", "┘"))
end

Base.show(io::IO, dt::DataTreatment) = _show_datatreatment(io, dt)
Base.show(io::IO, ::MIME"text/plain", dt::DataTreatment) = _show_datatreatment(io, dt)

export DataTreatment
export get_id, get_type, get_vname, get_feat, get_nwin
export get_X, get_y, get_datafeature, get_metadata
export get_vnames, get_features, get_nwindows, get_reducefuncs
export get_groups, get_groupmethod, get_norm

export groupby
include("groupby.jl")

end
