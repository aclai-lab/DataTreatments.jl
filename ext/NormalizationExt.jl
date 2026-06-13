module NormalizationExt

using Normalization
using DataTreatments

import Normalization: __mapdims!, fit!, fit, normalize
import Normalization: dimparams, negdims, estimators, dims!, params!

# ---------------------------------------------------------------------------- #
#                 extend fit & normalize to multidim elements                  #
# ---------------------------------------------------------------------------- #
# Wrap an estimator function so it skips missing elements
"""
    _missingsafe(f) -> Function

Wraps an estimator function `f` so that it silently skips `missing`
elements before computing. Used internally to make Normalization.jl
estimators robust to arrays containing `missing` values.

# Arguments
- `f`: Any estimator function accepting an iterable (e.g. `mean`,
  `std`, `mad`).

# Returns
- A new function with the same signature as `f` that filters out
  `missing` values before calling `f`.
"""
function _missingsafe(f)
    function (x; kwargs...)
        # collect into a flat Vector, skipping missing outer elements
        # and flattening inner arrays, so norm/std/mad always get a
        # concrete array with known length
        filtered = collect(
            Iterators.flatten(
                v for v in x if !ismissing(v)
            )
        )
        f(filtered; kwargs...)
    end
end

"""
    __mapdims!(z, f, x::AbstractArray{<:Union{Missing,AbstractArray}}, y)

Extends `Normalization.__mapdims!` to handle arrays whose elements
are themselves arrays (i.e. multidimensional time series columns).

Applies the fitted normalization parameters `f(map(only, y)...)` to
each non-missing element of `x`, writing results into `z` in-place.
Missing elements are left unchanged.

# Arguments
- `z`: Output array (same shape as `x`).
- `f`: Fitted normalization parameter constructor.
- `x::AbstractArray{<:Union{Missing,AbstractArray}}`: Input array of
  array-valued elements, possibly containing `missing`.
- `y`: Parameter slices used to build the normalization map.
"""
function __mapdims!(z, f, x::AbstractArray{<:Union{Missing,AbstractArray}}, y)
    params = f(map(only, y)...)
    for i in eachindex(z, x)
        if !ismissing(x[i])
            @inbounds map!(params, z[i], x[i])
        end
    end
end

"""
    fit!(T::AbstractNormalization,
         X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
         dims=Normalization.dims(T)) where {A}

Extends `Normalization.fit!` to support arrays whose elements are
themselves arrays (e.g. time series stored in matrix columns), with
`missing` values safely skipped during parameter estimation.

Computes normalization parameters from `X` and stores them in `T`
in-place.

!!! note
    The element type of `T` must match the element type `A` of the
    inner arrays. A `TypeError` is thrown otherwise.

# Arguments
- `T::AbstractNormalization`: Normalization object to fit in-place.
- `X::AbstractArray{<:Union{Missing,AbstractArray{A}}}`: Input array
  of array-valued elements, possibly containing `missing`.

# Keyword Arguments
- `dims`: Dimensions along which to compute parameters. Defaults to
  the value stored in `T`.

# Throws
- `TypeError`: if `eltype(T) != A`.
"""
function fit!(
    T::AbstractNormalization,
    X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
    dims=Normalization.dims(T)
) where {A}
    eltype(T) == A || throw(TypeError(:fit!, "Normalization", eltype(T), X))

    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)

    ps = map(estimators(T)) do f
        sf = _missingsafe(f)
        reshape(
            isnothing(dims) ?
                sf(Iterators.flatten(Xs)) :
                [sf(slice) for slice in Xs]
            , nps...
        )
    end

    dims!(T, dims)
    params!(T, ps)
    nothing
end

"""
    fit(::Type{𝒯},
        X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
        dims=nothing) where {A,T,𝒯<:AbstractNormalization{T}}

Extends `Normalization.fit` to support arrays whose elements are
themselves arrays (e.g. time series stored in matrix columns), with
`missing` values safely skipped during parameter estimation.

Returns a new fitted normalization object of type `𝒯`.

# Arguments
- `𝒯<:AbstractNormalization{T}`: Normalization type to instantiate.
- `X::AbstractArray{<:Union{Missing,AbstractArray{A}}}`: Input array
  of array-valued elements, possibly containing `missing`.

# Keyword Arguments
- `dims`: Dimensions along which to compute parameters. Defaults to
  `nothing` (global estimation).

# Returns
- A fitted instance of `𝒯`.
"""
function fit(
    ::Type{𝒯},
    X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
    dims=nothing
) where {A,T,𝒯<:AbstractNormalization{T}}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)

    ps = map(estimators(𝒯)) do f
        sf = _missingsafe(f)
        reshape(
            isnothing(dims) ?
                sf(Iterators.flatten(Iterators.flatten(Xs))) :
                [sf(Iterators.flatten(skipmissing(slice))) for slice in Xs]
            , nps...
        )
    end
    
    𝒯(dims, ps)
end

"""
    fit(::Type{𝒯},
        X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
        kwargs...) where {A,𝒯<:AbstractNormalization}

Convenience overload of [`fit`](@ref) for unparameterised
normalization types. Automatically infers the inner element type `A`
and delegates to `fit(𝒯{A}, X; kwargs...)`.

# Arguments
- `𝒯<:AbstractNormalization`: Unparameterised normalization type.
- `X::AbstractArray{<:Union{Missing,AbstractArray{A}}}`: Input array
  of array-valued elements, possibly containing `missing`.
"""
function fit(
    ::Type{𝒯},
    X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
    kwargs...
) where {A,𝒯<:AbstractNormalization}
    fit(𝒯{A}, X; kwargs...)
end

"""
    normalize(X::AbstractArray{<:Union{Missing,AbstractArray}},
              T::AbstractNormalization; kwargs...)

Extends `Normalization.normalize` to support arrays whose elements
are themselves arrays, possibly containing `missing` values.

Creates a deep copy of `X` (copying inner arrays, preserving
`missing` entries), applies the fitted normalization `T` in-place,
and returns the result. The original array `X` is not modified.

# Arguments
- `X::AbstractArray{<:Union{Missing,AbstractArray}}`: Input array of
  array-valued elements, possibly containing `missing`.
- `T::AbstractNormalization`: A fitted normalization object.

# Returns
- A normalised copy of `X` with the same structure and `missing`
  entries preserved.
"""
function normalize(
    X::AbstractArray{<:Union{Missing,AbstractArray}},
    T::AbstractNormalization;
    kwargs...
)
    Y = map(x -> ismissing(x) ? missing : copy(x), X)
    normalize!(Y, T; kwargs...)
    return Y
end

end