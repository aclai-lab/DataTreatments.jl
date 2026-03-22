abstract type AbstractDataFeature end
abstract type AbstractDataset end

# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    levels::CategoricalArrays.CategoricalVector
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
end

struct ContinuousFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
end

struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    subid::Int
    vname::String
    dims::Int
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}
end

struct ReduceFeat{T} <: AbstractDataFeature
    id::Vector
    vname::String
    dims::Int
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}
end

get_dims(f::Union{AggregateFeat,ReduceFeat}) = f.dims

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
_get_features(a::Base.Callable) = a.features
_get_reducefunc(r::Base.Callable) = r.reducefunc

function _reindex_groups(groups::Vector{Vector{Int}}, idxs::AbstractVector{Int})
    idx_set = Set(idxs)
    # Build reverse mapping: old index -> new index
    old_to_new = Dict(old => new for (new, old) in enumerate(idxs))
    
    new_groups = Vector{Vector{Int}}()

    for grp in groups
        new_grp = [old_to_new[i] for i in grp if i in idx_set]
        if !isempty(new_grp)
            push!(new_groups, new_grp)
        end
    end
    
    return new_groups
end

_reindex_groups(groups::Nothing, idxs::AbstractVector{Int}) = Vector{Vector{Int}}()

mutable struct DiscreteDataset <: AbstractDataset
    data::AbstractMatrix
    info::Vector{<:DiscreteFeat}

    DiscreteDataset(data::AbstractMatrix, info::Vector{<:DiscreteFeat}) = new(data, info)
    
    function DiscreteDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple
    )
        T = datastruct.datatype[ids]
        vnames = vnames[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        codes, levels = discrete_encode(data[:, ids])

        return new(
            stack(codes),
            [DiscreteFeat{T[i]}(ids[i], vnames[i], levels[i], valid[i], miss[i])
                for i in eachindex(ids)]
        )
    end
end

mutable struct ContinuousDataset{T} <: AbstractDataset
    data::AbstractMatrix
    info::Vector{<:ContinuousFeat}

    ContinuousDataset(data::AbstractMatrix, info::Vector{<:ContinuousFeat{T}}) where T =
        new{T}(data, info)

    function ContinuousDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        float_type::Type
    )
        vnames = vnames[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        nan = datastruct.nanidxs[ids]

        return new{float_type}(
            reduce(hcat, [map(x -> ismissing(x) ? missing : float_type(x), @view data[:, id])
                for id in ids]),
            [ContinuousFeat{float_type}(ids[i], vnames[i], valid[i], miss[i], nan[i])
                for i in eachindex(ids)]
        )
    end
end

mutable struct MultidimDataset{T} <: AbstractDataset
    data::AbstractArray
    info::Vector{<:Union{AggregateFeat,ReduceFeat}}
    groups::Union{Nothing,Vector{Vector{Int}}}

    MultidimDataset(
        data::AbstractArray,
        info::Vector{<:AggregateFeat{T}},
        groups::Union{Nothing,Vector{Vector{Int}}}
    ) where T = new{AggregateFeat{T}}(data, info, groups)

    MultidimDataset(
        data::AbstractArray,
        info::Vector{<:ReduceFeat{T}},
        groups::Union{Nothing,Vector{Vector{Int}}}=nothing
    ) where T = new{ReduceFeat{T}}(data, info, groups)

    function MultidimDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        aggrfunc::Base.Callable,
        float_type::Type,
        groups::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}
    )
        data = @view data[:, ids]
        vnames = vnames[ids]
        dims = datastruct.dims[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        nan = datastruct.nanidxs[ids]
        hasmiss = datastruct.hasmissing[ids]
        hasnan = datastruct.hasnans[ids]

        md, nwindows = aggrfunc(data, valid, float_type)

        md_feats = if hasfield(typeof(aggrfunc), :features)
            tuples = Iterators.flatten((
                ((c, f, n) for f in _get_features(aggrfunc) for n in 1:nwindows[c])
                for c in eachindex(ids)
            ))

            [AggregateFeat{float_type}(
                ids[c],
                j,
                vnames[c],
                dims[c],
                f,
                n,
                valid[c],
                miss[c],
                nan[c],
                hasmiss[c],
                hasnan[c]
            ) for (j, (c, f, n)) in enumerate(tuples)]
        else
            [ReduceFeat{AbstractArray{float_type}}(
                ids[i],
                vnames[c],
                dims[c],
                _get_reducefunc(aggrfunc),
                valid[c],
                miss[c],
                nan[c],
                hasmiss[c],hasnan[c])
                for (i, c) in enumerate(axes(md,2))]
        end

        grouped = isnothing(groups) ? nothing : _groupby(md_feats, groups)

        new{eltype(md_feats)}(md, md_feats, grouped)
    end
end

Base.getindex(ds::MultidimDataset, idxs::AbstractVector{Int}) =
    MultidimDataset(@view(ds.data[:, idxs]), ds.info[idxs], _reindex_groups(ds.groups, idxs))

get_dims(d::MultidimDataset) = [get_dims(f) for f in d.info]