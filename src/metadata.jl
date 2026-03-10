# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
"""
    DiscreteFeat{T} <: AbstractDataFeature

Metadata for a **discrete (categorical)** column in a `DataTreatment`.

Stores the categorical levels alongside validity information. Used for columns
whose values belong to a finite set of categories (e.g., labels, classes, 
ordinal codes).

# Type Parameter
- `T`: the element type of the column (e.g., `String`, `Int`).

# Fields
- `id::Vector`: column index identifier within the original dataset.
- `vname::String`: column name.
- `levels::CategoricalArrays.CategoricalVector`: the ordered set of categorical levels.
- `valididxs::Vector{Int}`: row indices with valid (non-missing) values.
- `missingidxs::Vector{Int}`: row indices containing `missing`.

# Example
```julia
DiscreteFeat{String}([1], "color", categorical(["red", "blue", "green"]), [1,2,3], Int[])
```
"""
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

"""
    ContinuousFeat{T} <: AbstractDataFeature

Metadata for a **continuous (numeric scalar)** column in a `DataTreatment`.

Tracks validity, missingness, and `NaN` indices for scalar numeric columns
(e.g., measurements, sensor readings, computed statistics).

# Type Parameter
- `T`: the numeric element type (e.g., `Float64`, `Int`).

# Fields
- `id::Vector`: column index identifier within the original dataset.
- `vname::String`: column name.
- `valididxs::Vector{Int}`: row indices with valid (non-missing, non-NaN) values.
- `missingidxs::Vector{Int}`: row indices containing `missing`.
- `nanidxs::Vector{Int}`: row indices containing `NaN`.

# Example
```julia
ContinuousFeat{Float64}([2], "temperature", [1,2,4,5], [3], Int[])
```
"""
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

"""
    AggregateFeat{T} <: AbstractDataFeature

Metadata for a **multidimensional column processed via aggregation** in a `DataTreatment`.

When a column contains multidimensional elements (e.g., time series, spectrograms),
the `aggregate` strategy flattens them into tabular form by applying a set of
feature functions over one or more sliding windows. Each original element is
splatted into multiple scalar columns — one per (window, feature) combination.

This struct stores the metadata needed to reconstruct and validate that process.

# Type Parameter
- `T`: the element type of the inner arrays (e.g., `Float64`).

# Fields
- `id::Vector`: column index identifier within the original dataset.
- `vname::String`: column name.
- `feat::Base.Callable`: the feature function applied within each window
  (e.g., `maximum`, `mean`).
- `nwin::Int`: the number of windows used in the aggregation.
- `valididxs::Vector{Int}`: row indices with valid (non-missing, non-NaN) elements.
- `missingidxs::Vector{Int}`: row indices where the element is `missing`.
- `nanidxs::Vector{Int}`: row indices where the element is `NaN`.
- `hasmissing::Vector{Int}`: row indices where the element is a valid array but
  contains `missing` values internally.
- `hasnans::Vector{Int}`: row indices where the element is a valid array but
  contains `NaN` values internally.

# Example
```julia
AggregateFeat{Float64}([3], "audio_signal", maximum, 4, [1,2], Int[], Int[], Int[], [2])
```

See also: [`ReduceFeat`](@ref), [`aggregate`](@ref)
"""
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

"""
    ReduceFeat{T} <: AbstractDataFeature

Metadata for a **multidimensional column processed via size reduction** in a `DataTreatment`.

Unlike [`AggregateFeat`](@ref), which flattens multidimensional elements into
scalar tabular columns, `ReduceFeat` preserves the original dimensionality of
each element and simply reduces its size to make it manageable for machine learning
pipelines (e.g., downsampling a 10 000-point time series to 256 points, or resizing
a 1024×128 spectrogram to 64×16).

This struct stores the metadata needed to reconstruct and validate that process.

# Type Parameter
- `T`: the element type of the inner arrays (e.g., `Float64`).

# Fields
- `id::Vector`: column index identifier within the original dataset.
- `vname::String`: column name.
- `reducefunc::Base.Callable`: the function used to reduce the element size
  (e.g., a resampling or interpolation callable).
- `valididxs::Vector{Int}`: row indices with valid (non-missing, non-NaN) elements.
- `missingidxs::Vector{Int}`: row indices where the element is `missing`.
- `nanidxs::Vector{Int}`: row indices where the element is `NaN`.
- `hasmissing::Vector{Int}`: row indices where the element is a valid array but
  contains `missing` values internally.
- `hasnans::Vector{Int}`: row indices where the element is a valid array but
  contains `NaN` values internally.

# Example
```julia
ReduceFeat{Float64}([4], "spectrogram", my_downsample, [1,2,3], Int[], Int[], Int[], [3])
```

See also: [`AggregateFeat`](@ref), [`reducesize`](@ref)
"""
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