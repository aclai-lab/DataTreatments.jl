abstract type AbstractParamNormalization{T} <: AbstractNormalization{T} end
const ParamNormUnion = Union{<:AbstractParamNormalization, Type{<:AbstractParamNormalization}}

macro _ParamNormalization(name, 𝑝, 𝑜, 𝑓)
    quote
        mutable struct $(esc(name)){T} <: AbstractParamNormalization{T}
            dims
            p::NTuple{length($(esc(𝑝))), AbstractArray{T}}
            o::NTuple{length($(esc(𝑜))), Real}
        end
        Normalization.estimators(::Type{N}) where {N<:$(esc(name))} = $(esc(𝑝))
        DataTreatments.options(::Type{N}) where {N<:$(esc(name))} = $(esc(𝑜))
        Normalization.forward(::Type{N}) where {N<:$(esc(name))} = $(esc(𝑓))
    end
end
options(::N) where {N<:AbstractParamNormalization} = options(N)
params(N::AbstractParamNormalization) = N.p
_options(N::AbstractParamNormalization) = N.o

# ---------------------------------------------------------------------------- #
#                                @_Normalization                               #
# ---------------------------------------------------------------------------- #
@_Normalization ZScoreRobust (median, (x)->median(abs.(x .- median(x)))) Normalization.zscore

@_ParamNormalization ScaledMinMax (minimum, maximum) (:lower, :upper) scaled_minmax

@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale
@_Normalization ScaleIqr (iqr,) scale

@_Normalization CenterMedian (median,) Normalization.center

@_Normalization PNorm1 ((x)->norm(x, 1),) scale
@_Normalization PNorm2 ((x)->norm(x, 2),) scale
@_Normalization PNormInf ((x)->norm(x, Inf),) scale

# ---------------------------------------------------------------------------- #
#                                     dims                                     #
# ---------------------------------------------------------------------------- #
function __parammapdims!(z, f, x, y, o)
    @inbounds map!(f(map(only, y)..., o...), z, x)
end

function _parammapdims!(zs::Slices{<:AbstractArray}, f, xs::Slices{<:AbstractArray}, ys::NTuple{N, <:AbstractArray}, o) where {N}
    @sync Threads.@threads for i in eachindex(xs) #
        y = ntuple((j -> @inbounds ys[j][i]), Val(N)) # Extract parameters for nth slice
        __parammapdims!(zs[i], f, xs[i], y, o)
    end
end

function parammapdims!(z, f, x::AbstractArray{T, n}, y, o; dims) where {T, n}
    isnothing(dims) && (dims = 1:n)
    max(dims...) <= n || error("A chosen dimension is greater than the number of dimensions of the reference array")
    unique(dims) == [dims...] || error("Repeated dimensions")
    length(dims) == n && return __parammapdims!(z, f, x, y, o) # ? Shortcut for global normalisation
    all(all(size.(y, i) .== 1) for i ∈ dims) || error("Inconsistent dimensions; dimensions $dims must have size 1")

    negs = negdims(dims, n)
    all(all(size(x, i) .== size.(y, i)) for i ∈ negs) || error("Inconsistent dimensions; dimensions $negs must have size $(size(x)[collect(negs)])")

    xs = eachslice(x; dims=negs)
    zs = eachslice(z; dims=negs)
    ys = eachslice.(y; dims=negs)
    _parammapdims!(zs, f, xs, ys, o)
end

function optparams!(N::ScaledMinMax; lower::Real=0.0, upper::Real=1.0)
    normalization(N).o = (lower, upper)
end

function optparams!(N::PNorm; p::Real=2.0)
    normalization(N).o = (p)
end

# ---------------------------------------------------------------------------- #
#                                     fit                                      #
# ---------------------------------------------------------------------------- #
function fit!(T::AbstractParamNormalization, X::AbstractArray{A}; dims=Normalization.dims(T), kwargs...) where {A}
    eltype(T) == A || throw(TypeError(:fit!, "Normalization", eltype(T), X))

    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(T)) do f
        reshape(map(f, Xs), nps...)
    end

    dims!(T, dims)
    params!(T, ps)
    optparams!(T; kwargs...)
    nothing
end

function fit(::Type{𝒯}, X::AbstractArray{A}; dims=nothing, kwargs...) where {A,T,𝒯<:AbstractParamNormalization{T}}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(𝒯)) do f
        reshape(map(f, Xs), nps...)
    end
    𝒯(dims, ps; kwargs...)
end
function fit(::Type{𝒯}, X::AbstractArray{A}; dims=nothing, kwargs...) where {A,𝒯<:AbstractParamNormalization}
    fit(𝒯{A}, X; dims, kwargs...)
end

function fit(N::AbstractParamNormalization, X::AbstractArray{A}; dims=Normalization.dims(N), kwargs...) where {A}
    fit(typeof(N), X; dims, kwargs...)
end

# ---------------------------------------------------------------------------- #
#                                  normalize                                   #
# ---------------------------------------------------------------------------- #
function normalize!(Z::AbstractArray, X::AbstractArray, T::AbstractParamNormalization; kwargs...)
    isfit(T) || fit!(T, X; kwargs...)
    dims = Normalization.dims(T)
    parammapdims!(Z, forward(T), X, params(T), _options(T); dims)
    return nothing
end
function normalize!(Z, X, ::Type{𝒯}; kwargs...) where {𝒯 <: AbstractParamNormalization}
    normalize!(Z, X, fit(𝒯, X; kwargs...))
end
normalize!(X, T::ParamNormUnion; kwargs...) = normalize!(X, X, T; kwargs...)

function normalize(X, T::AbstractParamNormalization; kwargs...)
    Y = copy(X)
    normalize!(Y, T; kwargs...)
    return Y
end
function normalize(X, ::Type{𝒯}; kwargs...) where {𝒯 <: AbstractParamNormalization}
    normalize(X, fit(𝒯, X; kwargs...))
end

normalize(X, p::NamedTuple) = normalize(X, p[1]; Base.tail(p)...)

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
(::Type{N})(
    dims=nothing,
    p=ntuple(_->Vector{T}(), length(estimators(N)));
    lower::Real=0.0,
    upper::Real=1.0
) where {T, N<:ScaledMinMax{T}} = N(dims, p, (lower, upper));

scale(s) = Base.Fix2(/, s)

function scaled_minmax(xmin, xmax, lower, upper)
    scale = (upper - lower) / (xmax - xmin)
    (x) -> clamp(lower + (x - xmin) * scale, lower, upper)
end

# ---------------------------------------------------------------------------- #
#                                    callers                                   #
# ---------------------------------------------------------------------------- #
checkdims(dims::Union{Int64,Nothing}=nothing) =
    (isnothing(dims) || dims == 1 || dims == 2) ||
        error("dims must be nothing, 1 (column-wise), or 2 (row-wise)")

checkmethod(method::Symbol, methods::Tuple{Vararg{Symbol}}) =
    (method in methods) || error("method must be $methods")

checkrange(lower::Real, upper::Real) =
    lower < upper || error("lower must be less than upper")

checkp(p::Real) =
    (p == 1 || p == 2 || p == Inf) || error("p must be 1, 2, or Inf")

function (::Type{ZScore})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:std)
    checkdims(dims)
    checkmethod(method, (:std, :robust, :half))
    method == :robust && return (type = ZScoreRobust, dims = dims)
    method == :half && return (type = HalfZScore, dims = dims)
    return (type = ZScore, dims = dims)
end

function (::Type{MinMax})(; dims::Union{Int64,Nothing}=nothing, lower::Real=0.0, upper::Real=1.0)
    checkdims(dims)
    checkrange(lower, upper)
    (lower == 0.0 && upper == 1.0) ? (type = MinMax, dims = dims) : (type = ScaledMinMax, dims = dims, lower = lower, upper = upper)
end

function (::Type{Scale})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:std)
    checkdims(dims)
    checkmethod(method, (:std, :mad, :first, :iqr))
    method == :mad && return (type = ScaleMad, dims = dims)
    method == :first && return (type = ScaleFirst, dims = dims)
    method == :iqr && return (type = ScaleIqr, dims = dims)
    return (type = Scale, dims = dims)
end

function (::Type{Sigmoid})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = Sigmoid, dims = dims)
end

function (::Type{Center})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:mean)
    checkdims(dims)
    checkmethod(method, (:mean, :median))
    method == :median && return (type = CenterMedian, dims = dims)
    return (type = Center, dims = dims)
end

function (::Type{UnitEnergy})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = UnitEnergy, dims = dims)
end

function (::Type{UnitPower})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = UnitPower, dims = dims)
end

function (::Type{PNorm})(; dims::Union{Int64,Nothing}=nothing, p::Real=2.0)
    checkdims(dims)
    checkp(p)
    p == 1 && return (type = PNorm1, dims = dims)
    p == Inf && return (type = PNormInf, dims = dims)
    return (type = PNorm2, dims = dims)
end
