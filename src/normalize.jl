# ---------------------------------------------------------------------------- #
#                                   NormSpec                                   #
# ---------------------------------------------------------------------------- #
struct NormSpec{T<:AbstractNormalization}
    type::Type{T}
    dims::Union{Int64,Nothing}

    NormSpec(type::Type{T}, dims::Union{Int64,Nothing}) where {T<:AbstractNormalization} =
        new{T}(type, dims)
end

_nt(ns::NormSpec) = (type = ns.type, dims = ns.dims)

Base.show(io::IO, ns::NormSpec) = show(io, _nt(ns))
Base.show(io::IO, ::MIME"text/plain", ns::NormSpec) = show(io, _nt(ns))

Base.Tuple(ns::NormSpec) = (ns.type, ns.dims)

# ---------------------------------------------------------------------------- #
#                                @_Normalization                               #
# ---------------------------------------------------------------------------- #
@_Normalization ZScoreRobust (median, (x)->median(abs.(x .- median(x)))) Normalization.zscore

scale(s) = Base.Fix2(/, s)
@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale
@_Normalization ScaleIqr (iqr,) scale

@_Normalization CenterMedian (median,) Normalization.center

@_Normalization PNorm1 ((x)->norm(x, 1),) scale
@_Normalization PNorm ((x)->norm(x, 2),) scale
@_Normalization PNormInf ((x)->norm(x, Inf),) scale

# ---------------------------------------------------------------------------- #
#                                    callers                                   #
# ---------------------------------------------------------------------------- #
checkdims(dims::Union{Int64,Nothing}=nothing) =
    (isnothing(dims) || dims == 1 || dims == 2) ||
        error("dims must be nothing, 1 (column-wise), or 2 (row-wise)")

checkmethod(method::Symbol, methods::Tuple{Vararg{Symbol}}) =
    (method in methods) || error("method must be $methods")

checkp(p::Real) =
    (p == 1 || p == 2 || p == Inf) || error("p must be 1, 2, or Inf")

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:std) where {T<:ZScore}
    checkdims(dims)
    checkmethod(method, (:std, :robust, :half))
    S = method == :robust ? ZScoreRobust : method == :half ? HalfZScore : T
    return NormSpec(S, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing) where {T<:MinMax}
    checkdims(dims)
    return NormSpec(T, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:std) where {T<:Scale}
    checkdims(dims)
    checkmethod(method, (:std, :mad, :first, :iqr))
    S = method == :mad ? ScaleMad : method == :first ? ScaleFirst : method == :iqr ? ScaleIqr : T
    return NormSpec(S, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing) where {T<:Sigmoid}
    checkdims(dims)
    return NormSpec(T, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:mean) where {T<:Center}
    checkdims(dims)
    checkmethod(method, (:mean, :median))
    S = method == :median ? CenterMedian : T
    return NormSpec(S, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing) where {T<:UnitEnergy}
    checkdims(dims)
    return NormSpec(T, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing) where {T<:UnitPower}
    checkdims(dims)
    return NormSpec(T, dims)
end

function (::Type{T})(; dims::Union{Int64,Nothing}=nothing, p::Real=2.0) where {T<:PNorm}
    checkdims(dims)
    checkp(p)
    S = p == 1 ? PNorm1 : p == Inf ? PNormInf : T
    return NormSpec(S, dims)
end
