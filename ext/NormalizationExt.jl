module NormalizationExt

using Normalization
using DataTreatments

import Normalization: __mapdims!, fit!, fit, normalize
import Normalization: dimparams, negdims, estimators, dims!, params!

# ---------------------------------------------------------------------------- #
#                 extend fit & normalize to multidim elements                  #
# ---------------------------------------------------------------------------- #

# Wrap an estimator function so it skips missing elements
function _missingsafe(f)
    function (x; kwargs...)
        filtered = Iterators.filter(!ismissing, x)
        f(filtered; kwargs...)
    end
end

function __mapdims!(z, f, x::AbstractArray{<:Union{Missing,AbstractArray}}, y)
    params = f(map(only, y)...)
    for i in eachindex(z, x)
        if !ismissing(x[i])
            @inbounds map!(params, z[i], x[i])
        end
    end
end

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
                sf(Iterators.flatten(Xs...)) :
                sf.(Iterators.flatten.(Xs))
            , nps...
        )
    end
    
    dims!(T, dims)
    params!(T, ps)
    nothing
end

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

function fit(
    ::Type{𝒯},
    X::AbstractArray{<:Union{Missing,AbstractArray{A}}};
    kwargs...
) where {A,𝒯<:AbstractNormalization}
    fit(𝒯{A}, X; kwargs...)
end

function normalize(X::AbstractArray{<:Union{Missing,AbstractArray}}, T::AbstractNormalization; kwargs...)
    Y = map(x -> ismissing(x) ? missing : copy(x), X)
    normalize!(Y, T; kwargs...)
    return Y
end

end