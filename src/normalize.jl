# ---------------------------------------------------------------------------- #
#                              custom functions                                #
# ---------------------------------------------------------------------------- #
rootenergy(x::AbstractArray) = sum(abs2, x) |> sqrt
rootpower(x) = sqrt(mean(abs2, x))
halfstd(x) = std(x) ./ convert(eltype(x), sqrt(1 - (2 / π)))

# ---------------------------------------------------------------------------- #
#                               core functions                                 #
# ---------------------------------------------------------------------------- #
_zscore(y, o)  = (x) -> (x - y) / o # * But this needs to be mapped over SCALAR y
_sigmoid(y, o) = (x) -> inv(1 + exp(-(x - y) / o))
_rescale(l, u)   = (x) -> (x - l) / (u - l)
_center(y)     = (x) -> x - y
_unitenergy(e) = Base.Fix2(/, e) # For unitful consistency, the sorted parameter is the root energy
_unitpower(p)  = Base.Fix2(/, p)
_outliersuppress(y, o; thr=5.0) = (x) -> abs(o) > thr * o ? y + sign(x - y) * thr * o : x
function _minmaxclip(l,u)
    (x) -> begin
        l == u && (return x == u ? 0.5 : (x > u) * one(u)) # Return 1.0 if x > u, else 0.0
        return clamp((x - l) / (u - l), 0.0, 1.0)
    end
end

# ---------------------------------------------------------------------------- #
#                              caller functions                                #
# ---------------------------------------------------------------------------- #
zscore()::Function          = x -> _zscore(Statistics.mean(x), Statistics.std(x))
sigmoid()::Function         = x -> _sigmoid(Statistics.mean(x), Statistics.std(x))
rescale()::Function         = x -> _rescale(minimum(x), maximum(x))
center()::Function          = x -> _center(Statistics.mean(x))
unitenergy()::Function      = x -> _unitenergy(rootenergy(x))
unitpower()::Function       = x -> _unitpower(rootpower(x))
halfzscore()::Function      = x -> _zscore(minimum(x), halfstd(x))
outliersuppress()::Function = x -> _outliersuppress(Statistics.mean(x), Statistics.std(x))
minmaxclip()::Function      = x -> _minmaxclip(minimum(x), maximum(x))

# ---------------------------------------------------------------------------- #
#                              normalize functions                             #
# ---------------------------------------------------------------------------- #
function element_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    _X = Iterators.flatten(X)
    nfunc = n(collect(_X))
    [nfunc(X[idx]) for idx in CartesianIndices(X)]
end

function tabular_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = @inbounds [n(collect(cols[i])) for i in eachindex(cols)]
    [nfuncs[idx[2]](X[idx]) for idx in CartesianIndices(X)]
end

@inline function _ds_norm!(Xn::AbstractArray, X::AbstractArray, nfunc)
    @inbounds @simd for i in eachindex(X, Xn)
        Xn[i] = nfunc(X[i])
    end
    return Xn
end

function ds_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    # compute normalization functions for each column
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = Vector{Function}(undef, length(cols))
    Threads.@threads for i in axes(X, 2)
        nfuncs[i] = n(collect(cols[i]))
    end
    
    # apply normalization
    Xn = similar(X)
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            Xn[i, j] = similar(X[i, j])
            _ds_norm!(Xn[i, j], X[i, j], nfuncs[j])
        end
    end
    
    return Xn
end
