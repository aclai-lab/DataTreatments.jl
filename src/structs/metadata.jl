# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Vector
    vname::String
    levels::CategoricalArrays.CategoricalVector
    valididxs::Vector{Int}
    missingidxs::Vector{Int}

    function DiscreteFeat{T}(
        id::Vector,
        vname::String,
        levels::CategoricalArrays.CategoricalVector,
        valididxs::Vector{Int},
        missingidxs::Vector{Int}
    ) where T
        new{T}(id, vname, levels, valididxs, missingidxs)
    end
end

struct ContinuousFeat{T} <: AbstractDataFeature
    id::Vector
    vname::String
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}

    function ContinuousFeat{T}(
        id::Vector,
        vname::String,
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int}
    ) where T
        new{T}(id, vname, valididxs, missingidxs, nanidxs)
    end
end

struct AggregateFeat{T} <: AbstractDataFeature
    id::Vector
    vname::String
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}

    function AggregateFeat{T}(
        id::Vector,
        vname::String,
        feat::Base.Callable,
        nwin::Int,
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int},
        hasmissing::Vector{Int},
        hasnans::Vector{Int}
    ) where T
        new{T}(id, vname, feat, nwin, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end
end

struct ReduceFeat{T} <: AbstractDataFeature
    id::Vector
    vname::String
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}

    function ReduceFeat{T}(
        id::Vector,
        vname::String,
        reducefunc::Base.Callable,
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int},
        hasmissing::Vector{Int},
        hasnans::Vector{Int}
    ) where T
        new{T}(id, vname, reducefunc, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end
end

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_id(f::AbstractDataFeature) -> Int

Returns the column id of the feature.
"""
get_id(f::AbstractDataFeature) = f.id

"""
    get_vname(f::AbstractDataFeature) -> String

Returns the variable name of the feature.
"""
get_vname(f::AbstractDataFeature) = f.vname

"""
    get_valididxs(f::AbstractDataFeature) -> Vector{Int}

Returns the indices of valid values for the feature.
"""
get_valididxs(f::AbstractDataFeature) = f.valididxs

"""
    get_missingidxs(f::AbstractDataFeature) -> Vector{Int}

Returns the indices of `missing` values for the feature.
"""
get_missingidxs(f::AbstractDataFeature) = f.missingidxs

"""
    get_nanidxs(f::Union{ContinuousFeat,AggregateFeat,ReduceFeat}) -> Vector{Int}

Returns the indices of `NaN` values for the feature.
"""
get_nanidxs(f::Union{ContinuousFeat,AggregateFeat,ReduceFeat}) = f.nanidxs

"""
    get_hasmissing(f::Union{AggregateFeat,ReduceFeat}) -> Vector{Bool}

Returns the per-element flags indicating presence of `missing` values internally.
"""
get_hasmissing(f::Union{AggregateFeat,ReduceFeat}) = f.hasmissing

"""
    get_hasnans(f::Union{AggregateFeat,ReduceFeat}) -> Vector{Bool}

Returns the per-element flags indicating presence of `NaN` values internally.
"""
get_hasnans(f::Union{AggregateFeat,ReduceFeat}) = f.hasnans

"""
    get_levels(f::DiscreteFeat) -> CategoricalArrays.CategoricalVector

Returns the categorical levels of the discrete feature.
"""
get_levels(f::DiscreteFeat) = f.levels

"""
    get_feat(f::AggregateFeat) -> Base.Callable

Returns the aggregation function of the feature.
"""
get_feat(f::AggregateFeat) = f.feat

"""
    get_nwin(f::AggregateFeat) -> Int

Returns the number of windows of the aggregate feature.
"""
get_nwin(f::AggregateFeat) = f.nwin

"""
    get_reducefunc(f::ReduceFeat) -> Base.Callable

Returns the reduce function of the feature.
"""
get_reducefunc(f::ReduceFeat) = f.reducefunc