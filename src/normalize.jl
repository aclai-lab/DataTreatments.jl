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
function _outliersuppress(y, o, thr=5.0)
    (x) -> begin
        o = x - y
        if abs(o) > thr * o
            return y + sign(o) * thr * o
        else
            return x
        end
    end
end

function _minmaxclip(l,u)
    (x) -> begin
        if l == u
            if x == u
                return 0.5
            else
                return (x > u) * one(u) # Return 1.0 if x > u, else 0.0
            end
        else
            return clamp((x - l) / (u - l), 0.0, 1.0)
        end
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

_ds_norm(X::AbstractArray, nfunc::Base.Callable) = nfunc.(X)

function ds_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = [n(collect(cols[i])) for i in eachindex(cols)]
    [_ds_norm(X[idx], nfuncs[idx[2]]) for idx in CartesianIndices(X)]
end
