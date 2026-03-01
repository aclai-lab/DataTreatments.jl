module DataTreatments

using DataFrames
using Catch22

using Statistics: mean, median, std, cov
using StatsBase: mad
using LinearAlgebra: norm

using CategoricalArrays
using Normalization

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataTreatment end
abstract type AbstractDataFeature end
abstract type AbstractMetaData end

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

import Normalization: @_Normalization
export ZScore, MinMax, Center, Sigmoid, UnitEnergy, UnitPower
export Scale, ScaleMad, ScaleFirst, PNorm1, PNorm, PNormInf
export Robust
include("normalize.jl")

include("ds_builder.jl")

# ---------------------------------------------------------------------------- #
#                                DataFeature                                   #
# ---------------------------------------------------------------------------- #
# struct DiscreteFeat{T} <: AbstractDataFeature
mutable struct DiscreteFeat{T} <: AbstractDataFeature
    # X::AbstractArray{T}
    id::Int
    type::Type
    vname::Symbol
    hasmissing::Bool
    hasnan::Bool
end

struct ScalarFeat{T} <: AbstractDataFeature
# mutable struct ScalarFeat{T} <: AbstractDataFeature
    # X::AbstractArray{T}
    id::Int
    type::Type
    vname::Symbol
    hasmissing::Bool
    hasnan::Bool
end

# struct AggregateFeat{T} <: AbstractDataFeature
mutable struct AggregateFeat{T} <: AbstractDataFeature
    # X::AbstractArray{T}
    id::Int
    type::Type
    vname::Symbol
    feat::Base.Callable
    nwin::Int
    hasmissing::Bool
    hasnan::Bool
end

# struct ReduceFeat{T} <: AbstractDataFeature
mutable struct ReduceFeat{T} <: AbstractDataFeature
    # X::AbstractArray{T}
    id::Int
    type::Type
    vname::Symbol
    reducefunc::Base.Callable
    hasmissing::Bool
    hasnan::Bool
end

# getters
# get_X(f::AbstractDataFeature) = f.X
get_id(f::AbstractDataFeature) = f.id
get_type(f::AbstractDataFeature) = f.type
get_vname(f::AbstractDataFeature) = f.vname
get_hasmissing(f::AbstractDataFeature) = f.has_missing
get_hasnan(f::AbstractDataFeature) = f.has_nan

get_feat(f::AggregateFeat) = f.feat
get_nwin(f::AggregateFeat) = f.nwin

get_reducefunc(f::ReduceFeat) = f.reducefunc

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const MultiDimFeats = Union{AggregateFeat, ReduceFeat}
const Discrete = Union{AbstractString, Symbol, CategoricalValue, UInt32, Int}
const Groups = Union{BitVector,Symbol,Vector{Symbol},Vector{Vector{Symbol}}}

# ---------------------------------------------------------------------------- #
#                                  MetaData                                    #
# # ---------------------------------------------------------------------------- #
struct MetaData <: AbstractMetaData
    norm_tc::Union{Nothing,Type{<:AbstractNormalization}}
    norm_md::Union{Nothing,Type{<:AbstractNormalization}}
    group_td::Groups
    group_tc::Groups
    group_md::Groups
    groupidxs_td::Union{Nothing,Vector{Base.Generator}}
    groupidxs_tc::Union{Nothing,Vector{Base.Generator}}
    groupidxs_md::Union{Nothing,Vector{Base.Generator}}
end

# ---------------------------------------------------------------------------- #
#                                DataTreatment                                 #
# ---------------------------------------------------------------------------- #
struct DataTreatment{T,S} <: AbstractDataTreatment
    Xtd::Union{Nothing, Matrix}
    Xtc::Union{Nothing, Matrix}
    Xmd::Union{Nothing, Matrix}
    td_feats::Union{Nothing, Vector{<:DiscreteFeat}}
    tc_feats::Union{Nothing, Vector{<:ScalarFeat}}
    md_feats::Union{Nothing, Vector{<:AbstractDataFeature}}
    y::Vector{S}
    metadata::MetaData

    function DataTreatment(
        X::Matrix{T},
        y::Union{AbstractVector,Nothing}=nothing;
        norm_tc::Union{Type{<:AbstractNormalization},Nothing}=nothing,
        norm_md::Union{Type{<:AbstractNormalization},Nothing}=nothing,
        group_td::Groups=:vname,
        group_tc::Groups=:vname,
        group_md::Groups=:vname,
        kwargs...
    ) where T
        isnothing(y) ? (y = Vector{Nothing}(nothing, size(X, 1))) : size(X, 1) != length(y) &&
            throw(DimensionMismatch("y length ($(length(y))) must match X rows ($(size(X, 1)))"))

        Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats = build_datasets(X; kwargs...)

        groupidxs_td = groupidxs_tc = groupidxs_md = nothing

        !isnothing(td_feats) && begin
            groupidxs_td = groupby(td_feats, group_td)
        end

        !isnothing(tc_feats) && begin
            groupidxs_tc = groupby(tc_feats, group_tc)
            if !isnothing(norm_tc)
                flat_groups = [reduce(vcat, g) for g in groupidxs_tc]
                Threads.@threads for g in flat_groups
                    normalize!(@view(Xtc[:, g]), norm_tc)
                end
            end
        end

        !isnothing(md_feats) && begin
            groupidxs_md = groupby(md_feats, group_md)
            if !isnothing(norm_md)
                flat_groups = [reduce(vcat, g) for g in groupidxs_md]
                Threads.@threads for g in flat_groups
                    normalize!(@view(Xmd[:, g]), norm_md)
                end
            end
        end

        metadata = MetaData(
            norm_tc,
            norm_md,
            group_td,
            group_tc,
            group_md,
            groupidxs_td isa Base.Generator ? [groupidxs_td] : groupidxs_td,
            groupidxs_tc isa Base.Generator ? [groupidxs_tc] : groupidxs_tc,
            groupidxs_md isa Base.Generator ? [groupidxs_md] : groupidxs_md
        )

        new{eltype(Xtc),eltype(y)}(Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats, y, metadata)
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
    (:Xtd, :Xtc, :Xmd, :td_feats, :tc_feats, :md_feats, :y, :metadata)

get_X(dt::DataTreatment) = dt.X
get_y(dt::DataTreatment) = dt.y
get_datafeature(dt::DataTreatment) = dt.datafeature
get_metadata(dt::DataTreatment) = dt.metadata

# metadata
get_groups(dt::DataTreatment) = reduce(vcat, collect.(dt.metadata.groups))
get_groupmethod(dt::DataTreatment) = dt.metadata.groupmethod
get_norm(dt::DataTreatment) = dt.metadata.norm

# Convenience methods for common operations
get_vnames(dt::DataTreatment) = unique(get_vname.(dt.datafeature))
get_features(dt::DataTreatment) = unique(get_feat.(dt.datafeature))
get_nwindows(dt::DataTreatment) = maximum(get_nwin.(dt.datafeature))
get_reducefuncs(dt::DataTreatment) = unique(get_reducefunc.(dt.datafeature))

# Size and iteration methods
Base.size(dt::DataTreatment) = size(dt.X)
Base.size(dt::DataTreatment, dim::Int) = size(dt.X, dim)
Base.length(dt::DataTreatment) = length(dt.datafeature)
Base.eltype(dt::DataTreatment) = eltype(dt.X)

# Indexing methods
Base.getindex(dt::DataTreatment, i::Int) = dt.X[:, i]
Base.getindex(dt::DataTreatment, i::Int, j::Int) = dt.X[i, j]
Base.getindex(dt::DataTreatment, ::Colon, j::Int) = dt.X[:, j]
Base.getindex(dt::DataTreatment, i::Int, ::Colon) = dt.X[i, :]
Base.getindex(dt::DataTreatment, I...) = dt.X[I...]

# ---------------------------------------------------------------------------- #
#                              DataTreatment show                              #
# ---------------------------------------------------------------------------- #
function _show_datatreatment(io::IO, dt::DataTreatment{T,S}) where {T,S}
    green  = "\e[32m"
    yellow = "\e[33m"
    white  = "\e[37m"
    reset  = "\e[0m"

    supervised = !(S <: Nothing)

    # build dataset info strings
    td_str = isnothing(dt.Xtd) ? nothing : "$(size(dt.Xtd, 1)) × $(size(dt.Xtd, 2))  ::  $(eltype(dt.Xtd))"
    tc_str = isnothing(dt.Xtc) ? nothing : "$(size(dt.Xtc, 1)) × $(size(dt.Xtc, 2))  ::  $(T)"
    md_str = isnothing(dt.Xmd) ? nothing : "$(size(dt.Xmd, 1)) × $(size(dt.Xmd, 2))  ::  $(eltype(dt.Xmd))"

    norm_tc_str = isnothing(dt.metadata.norm_tc) ? "none" : string(dt.metadata.norm_tc)
    norm_md_str = isnothing(dt.metadata.norm_md) ? "none" : string(dt.metadata.norm_md)

    gm_str(gm) = gm isa Symbol ? string(gm) :
                 gm isa Vector{Symbol} ? join(string.(gm), ", ") :
                 string(gm)

    lines = Tuple{String,String}[
        (" ",  "$(yellow)DataTreatment$(reset)"),
        ("  ", "$(white)y    $(supervised ? "supervised  ::  $S" : "unsupervised")$(reset)"),
    ]

    !isnothing(td_str) && push!(lines, ("  ", "$(white)Xtd  $(td_str)$(reset)"))
    !isnothing(tc_str) && push!(lines, ("  ", "$(white)Xtc  $(tc_str)$(reset)"))
    !isnothing(md_str) && push!(lines, ("  ", "$(white)Xmd  $(md_str)$(reset)"))

    push!(lines, ("  ", "$(white)norm_tc   $(norm_tc_str)$(reset)"))
    push!(lines, ("  ", "$(white)norm_md   $(norm_md_str)$(reset)"))

    !isnothing(dt.metadata.group_td) && push!(lines,
        ("  ", "$(white)group_td  $(gm_str(dt.metadata.group_td))$(reset)"))
    !isnothing(dt.metadata.group_tc) && push!(lines,
        ("  ", "$(white)group_tc  $(gm_str(dt.metadata.group_tc))$(reset)"))
    !isnothing(dt.metadata.group_md) && push!(lines,
        ("  ", "$(white)group_md  $(gm_str(dt.metadata.group_md))$(reset)"))

    visible_length(s) = length(replace(s, r"\e\[[0-9;]*m" => ""))

    full_lines = ["│$(prefix)$(content)" for (prefix, content) in lines]
    seps  = Set([1, 2])
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
export get_id, get_type, get_vname, get_feat, get_nwin, get_reducefunc
export get_X, get_y, get_datafeature, get_metadata
export get_vnames, get_features, get_nwindows, get_reducefuncs
export get_groups, get_groupmethod, get_norm

export groupby
include("groupby.jl")

end
