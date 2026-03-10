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
abstract type AbstractDataset end
abstract type AbstractDataFeature end
abstract type AbstractMetaData end

# ---------------------------------------------------------------------------- #
#                               internal utils                                 #
# ---------------------------------------------------------------------------- #
include("errors.jl")

# ---------------------------------------------------------------------------- #
#                                  structs                                     #
# ---------------------------------------------------------------------------- #
export DatasetStructure
export get_vnames, get_datatype, get_dims
export get_valididxs, get_missingidxs, get_nanidxs
export get_hasmissing, get_hasnans
export get_dataset_structure
include("structs/dataset_structure.jl")

export TreatmentGroup
export get_idxs, get_dims, get_vnames, get_aggrfunc, get_groupby
include("structs/treatment_group.jl")

export DiscreteFeat, ContinuousFeat, AggregateFeat, ReduceFeat
export get_id, get_vname, get_valididxs
export get_missingidxs, get_nanidxs, get_hasmissing, get_hasnans
export get_levels, get_feat, get_nwin, get_reducefunc
include("structs/metadata.jl")

export DiscreteDataset, ContinuousDataset, MultidimDataset
export discrete_encode
include("structs/dataset.jl")

export DataTreatment
export get_dataset, get_ds_struct, get_t_groups, get_float_type
export get_nrows, get_ncols
export get_datasets
include("structs/datatreatment.jl")

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

export aggregate, reducesize
export is_multidim_dataset, has_uniform_element_size, safe_feat
include("treatment.jl")

# import Normalization: @_Normalization
# export ZScore, MinMax, Center, Sigmoid, UnitEnergy, UnitPower
# export Scale, ScaleMad, ScaleFirst, PNorm1, PNorm, PNormInf
# export Robust
# include("normalize.jl")

# include("ds_builder.jl")

# # ---------------------------------------------------------------------------- #
# #                                DataFeature                                   #
# # ---------------------------------------------------------------------------- #
# struct DiscreteFeat <: AbstractDataFeature
#     id::Int
#     vname::Symbol
#     values::Vector{String}
#     hasmissing::Bool
# end

# struct ScalarFeat{T} <: AbstractDataFeature
#     id::Int
#     vname::Symbol
#     hasmissing::Bool
#     hasnan::Bool
# end

# struct AggregateFeat{T} <: AbstractDataFeature
#     id::Int
#     vname::Symbol
#     feat::Base.Callable
#     nwin::Int
#     hasmissing::Bool
#     hasnan::Bool
# end

# struct ReduceFeat{T} <: AbstractDataFeature
#     id::Int
#     vname::Symbol
#     reducefunc::Base.Callable
#     hasmissing::Bool
#     hasnan::Bool
# end

# # getters
# # get_X(f::AbstractDataFeature) = f.X
# get_id(f::AbstractDataFeature) = f.id
# get_type(f::AbstractDataFeature) = f.type
# get_vname(f::AbstractDataFeature) = f.vname
# get_hasmissing(f::AbstractDataFeature) = f.hasmissing

# get_hasnan(f::Union{<:ScalarFeat,AggregateFeat,ReduceFeat}) = f.hasnan

# get_feat(f::AggregateFeat) = f.feat
# get_nwin(f::AggregateFeat) = f.nwin

# get_reducefunc(f::ReduceFeat) = f.reducefunc

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# const MultiDimFeats = Union{AggregateFeat, ReduceFeat}
# const Discrete = Union{AbstractString, Symbol, CategoricalValue, UInt32, Int}
# const Groups = Union{BitVector,Symbol,Vector{Symbol},Vector{Vector{Symbol}}}

# # ---------------------------------------------------------------------------- #
# #                                  MetaData                                    #
# # # ---------------------------------------------------------------------------- #
# struct MetaData <: AbstractMetaData
#     norm_tc::Union{Nothing,Type{<:AbstractNormalization}}
#     norm_md::Union{Nothing,Type{<:AbstractNormalization}}
#     group_td::Groups
#     group_tc::Groups
#     group_md::Groups
#     groupidxs_td::Union{Nothing,Vector{Base.Generator}}
#     groupidxs_tc::Union{Nothing,Vector{Base.Generator}}
#     groupidxs_md::Union{Nothing,Vector{Base.Generator}}
# end

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
# function _normalize_groups!(
#     X::Matrix,
#     feats::Vector,
#     norm::Union{Type{<:AbstractNormalization},Nothing},
#     group::Groups
# )
#     if !isnothing(norm)
#         groupidxs = groupby(feats, group)
#         flat_groups = [reduce(vcat, g) for g in groupidxs]
#         Threads.@threads for g in flat_groups
#             normalize!(@view(X[:, g]), norm)
#         end
#     end
# end

# ---------------------------------------------------------------------------- #
#                                DataTreatment                                 #
# ---------------------------------------------------------------------------- #
# struct DataTreatment{T,S} <: AbstractDataTreatment
#     Xtd::Union{Nothing, Matrix}
#     Xtc::Union{Nothing, Matrix}
#     Xmd::Union{Nothing, Matrix}
#     td_feats::Union{Nothing, Vector{<:DiscreteFeat}}
#     tc_feats::Union{Nothing, Vector{<:ScalarFeat}}
#     md_feats::Union{Nothing, Vector{<:AbstractDataFeature}}
#     y::Vector{S}
#     metadata::MetaData

#     function DataTreatment(
#         X::Matrix{T},
#         y::Union{AbstractVector,Nothing}=nothing;
#         norm_tc::Union{Type{<:AbstractNormalization},Nothing}=nothing,
#         norm_md::Union{Type{<:AbstractNormalization},Nothing}=nothing,
#         group_td::Groups=:vname,
#         group_tc::Groups=:vname,
#         group_md::Groups=:vname,
#         kwargs...
#     ) where T
#         isnothing(y) ? (y = Vector{Nothing}(nothing, size(X, 1))) : size(X, 1) != length(y) &&
#             throw(DimensionMismatch("y length ($(length(y))) must match X rows ($(size(X, 1)))"))

#         Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats = build_datasets(X; kwargs...)

#         groupidxs_td = groupidxs_tc = groupidxs_md = nothing

#         !isnothing(td_feats) && begin
#             groupidxs_td = groupby(td_feats, group_td)
#         end

#         !isnothing(tc_feats) && begin
#             groupidxs_tc = groupby(tc_feats, group_tc)
#             _normalize_groups!(Xtc, tc_feats, norm_tc, group_tc)
#         end

#         !isnothing(md_feats) && begin
#             groupidxs_md = groupby(md_feats, group_md)
#             _normalize_groups!(Xmd, md_feats, norm_md, group_md)
#         end

#         metadata = MetaData(
#             norm_tc,
#             norm_md,
#             group_td,
#             group_tc,
#             group_md,
#             groupidxs_td isa Base.Generator ? [groupidxs_td] : groupidxs_td,
#             groupidxs_tc isa Base.Generator ? [groupidxs_tc] : groupidxs_tc,
#             groupidxs_md isa Base.Generator ? [groupidxs_md] : groupidxs_md
#         )

#         new{eltype(Xtc),eltype(y)}(Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats, y, metadata)
#     end

#     DataTreatment(X::AbstractDataFrame, args...; kwargs...) =
#         DataTreatment(Matrix(X), args...; vnames=propertynames(X), kwargs...)
# end

# ---------------------------------------------------------------------------- #
#                            DataTreatment methods                             #
# ---------------------------------------------------------------------------- #

# # value access methods
# Base.getproperty(dt::DataTreatment, s::Symbol) = getfield(dt, s)
# Base.propertynames(::DataTreatment) =
#     (:Xtd, :Xtc, :Xmd, :td_feats, :tc_feats, :md_feats, :y, :metadata)

# function get_X(dt::DataTreatment, type::Symbol=:all)
#     if type === :all
#         parts = []
#         !isnothing(dt.Xtd) && push!(parts, dt.Xtd)
#         !isnothing(dt.Xtc) && push!(parts, dt.Xtc)
#         !isnothing(dt.Xmd) && push!(parts, dt.Xmd)
#         return isempty(parts) ? nothing : reduce(hcat, parts)
#     elseif type === :discrete
#         return dt.Xtd
#     elseif type === :scalar
#         return dt.Xtc
#     elseif type === :multivariate
#         return dt.Xmd
#     else
#         throw(ArgumentError("type must be :all, :discrete, :scalar, or :multivariate"))
#     end
# end

# get_y(dt::DataTreatment) = dt.y

# function get_datafeature(dt::DataTreatment)
#     vcat(filter(!isnothing, [dt.td_feats, dt.tc_feats, dt.md_feats])...)
# end

# get_metadata(dt::DataTreatment) = dt.metadata

# # metadata
# get_groups(dt::DataTreatment) = reduce(vcat, collect.(dt.metadata.groups))
# get_groupmethod(dt::DataTreatment) = dt.metadata.groupmethod
# get_norm(dt::DataTreatment, type::Symbol=:all) = 
#     type === :all ? (dt.metadata.norm_tc, dt.metadata.norm_md) :
#     type === :scalar ? dt.metadata.norm_tc :
#     type === :multivariate ? dt.metadata.norm_md :
#     throw(ArgumentError("type must be :all, :scalar, or :multivariate"))

# # Convenience methods for common operations
# function get_vnames(dt::DataTreatment, type::Symbol=:all)
#     feat_sources = [
#         (type ∈ (:all, :discrete)) ? dt.td_feats : nothing,
#         (type ∈ (:all, :scalar)) ? dt.tc_feats : nothing,
#         (type ∈ (:all, :multivariate)) ? dt.md_feats : nothing,
#     ]
#     feats = vcat(filter(!isnothing, feat_sources)...)
#     return unique(get_vname.(feats))
# end

# function get_features(dt::DataTreatment, type::Symbol=:all)
#     feats = (type ∈ (:all, :multivariate)) ? dt.md_feats : nothing
#     isnothing(feats) && return Symbol[]
#     return unique(get_feat.(filter(f -> f isa AggregateFeat, feats)))
# end

# function get_nwindows(dt::DataTreatment, type::Symbol=:multivariate)
#     type != :multivariate && throw(ArgumentError("get_nwindows only applies to :multivariate type"))
#     isnothing(dt.md_feats) && return 0
#     return maximum(get_nwin.(filter(f -> f isa AggregateFeat, dt.md_feats)))
# end

# function get_reducefuncs(dt::DataTreatment, type::Symbol=:multivariate)
#     type != :multivariate && throw(ArgumentError("get_reducefuncs only applies to :multivariate type"))
#     isnothing(dt.md_feats) && return []
#     return unique(get_reducefunc.(filter(f -> f isa ReduceFeat, dt.md_feats)))
# end

# # Size and iteration methods
# Base.size(dt::DataTreatment) = size(get_X(dt, :all))
# Base.size(dt::DataTreatment, dim::Int) = size(get_X(dt, :all), dim)
# function Base.length(dt::DataTreatment, type::Symbol=:all)
#     count = 0
#     type ∈ (:all, :discrete) && !isnothing(dt.td_feats) && (count += length(dt.td_feats))
#     type ∈ (:all, :scalar) && !isnothing(dt.tc_feats) && (count += length(dt.tc_feats))
#     type ∈ (:all, :multivariate) && !isnothing(dt.md_feats) && (count += length(dt.md_feats))
#     return count
# end
# Base.eltype(dt::DataTreatment) = Union{eltype(get_X(dt, :discrete)), eltype(get_X(dt, :scalar)), eltype(get_X(dt, :multivariate))}

# # Indexing methods
# Base.getindex(dt::DataTreatment, i::Int) = get_X(dt, :all)[:, i]
# Base.getindex(dt::DataTreatment, i::Int, j::Int) = get_X(dt, :all)[i, j]
# Base.getindex(dt::DataTreatment, ::Colon, j::Int) = get_X(dt, :all)[:, j]
# Base.getindex(dt::DataTreatment, i::Int, ::Colon) = get_X(dt, :all)[i, :]
# Base.getindex(dt::DataTreatment, I...) = get_X(dt, :all)[I...]

# export DataTreatment
# export get_id, get_type, get_vname, get_feat, get_nwin, get_reducefunc
# export get_hasmissing, get_hasnan
# export get_X, get_y, get_datafeature, get_metadata
# export get_vnames, get_features, get_nwindows, get_reducefuncs
# export get_groups, get_groupmethod, get_norm

# # ---------------------------------------------------------------------------- #
# #                              DataTreatment show                              #
# # ---------------------------------------------------------------------------- #
# function _show_datatreatment(io::IO, dt::DataTreatment{T,S}) where {T,S}
#     yellow = "\e[33m"
#     white  = "\e[37m"
#     cyan   = "\e[36m"
#     reset  = "\e[0m"

#     supervised = !(S <: Nothing)

#     # build dataset info strings
#     td_str = isnothing(dt.Xtd) ? nothing : "$(size(dt.Xtd, 1))×$(size(dt.Xtd, 2))  discrete"
#     tc_str = isnothing(dt.Xtc) ? nothing : "$(size(dt.Xtc, 1))×$(size(dt.Xtc, 2))  scalar"
#     md_str = isnothing(dt.Xmd) ? nothing : "$(size(dt.Xmd, 1))×$(size(dt.Xmd, 2))  multivariate"

#     norm_tc_str = isnothing(dt.metadata.norm_tc) ? nothing : string(dt.metadata.norm_tc)
#     norm_md_str = isnothing(dt.metadata.norm_md) ? nothing : string(dt.metadata.norm_md)

#     gm_str(gm) = gm isa Symbol ? string(gm) :
#                  gm isa Vector{Symbol} ? join(string.(gm), ", ") :
#                  string(gm)

#     lines = Tuple{String,String}[
#         (" ",  "$(yellow)DataTreatment$(reset)"),
#     ]

#     !isnothing(td_str) && push!(lines, ("  ", "$(cyan)Xtd  $(reset)$(white)$(td_str)$(reset)"))
#     !isnothing(tc_str) && push!(lines, ("  ", "$(cyan)Xtc  $(reset)$(white)$(tc_str)$(reset)"))
#     !isnothing(md_str) && push!(lines, ("  ", "$(cyan)Xmd  $(reset)$(white)$(md_str)$(reset)"))

#     push!(lines, ("  ", "$(white)y    $(supervised ? "supervised  ::  $S" : "unsupervised")$(reset)"))

#     !isnothing(norm_tc_str) && push!(lines, ("  ", "$(white)norm_tc   $(norm_tc_str)$(reset)"))
#     !isnothing(norm_md_str) && push!(lines, ("  ", "$(white)norm_md   $(norm_md_str)$(reset)"))

#     !isnothing(dt.metadata.group_td) && push!(lines,
#         ("  ", "$(white)group_td  $(gm_str(dt.metadata.group_td))$(reset)"))
#     !isnothing(dt.metadata.group_tc) && push!(lines,
#         ("  ", "$(white)group_tc  $(gm_str(dt.metadata.group_tc))$(reset)"))
#     !isnothing(dt.metadata.group_md) && push!(lines,
#         ("  ", "$(white)group_md  $(gm_str(dt.metadata.group_md))$(reset)"))

#     visible_length(s) = length(replace(s, r"\e\[[0-9;]*m" => ""))

#     full_lines = ["$(prefix)$(content)" for (prefix, content) in lines]
    
#     for line in full_lines
#         println(io, line)
#     end
# end

# Base.show(io::IO, dt::DataTreatment) = _show_datatreatment(io, dt)
# Base.show(io::IO, ::MIME"text/plain", dt::DataTreatment) = _show_datatreatment(io, dt)

export groupby
include("groupby.jl")

end
