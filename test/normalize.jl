# https://www.sciencedirect.com/science/article/pii/S2214579623000400?via%3Dihub

# this code is for development use only

using DataTreatments
# using Normalization
using Statistics

X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]



@btime begin
    Xn = similar(X)
    i = 1

    for col in eachcol(X)
        x = reduce(vcat, col)
        n = fit(MinMax, reduce(vcat, col); dims=nothing)
        for j in eachindex(col)
            Xn[j,i] = normalize(X[j,i], n)
        end
        i += 1
    end
end
# 1.386 s (164304 allocations: 4.48 GiB)

@btime begin
    Xn = [similar(mat) for mat in X]
    buffer = Matrix{Float64}(undef, 200, 100*100)  # Pre-allocate once
    
    for (i, col) in enumerate(eachcol(X))
        # Copy into buffer without allocating
        for (k, mat) in enumerate(col)
            buffer[:, (k-1)*100+1:k*100] .= mat
        end
        
        n = fit(MinMax, buffer; dims=nothing)
        
        for j in eachindex(col)
            Xn[j,i] .= normalize(X[j,i], n)
        end
    end
end

@btime begin
    i = 1

    Threads.@threads for col in eachcol(X)
        n = fit(MinMax, reduce(vcat, col); dims=nothing)
        for j in eachindex(col)
            normalize!(X[j,i], n)
        end
        i += 1
    end
end


Normalization.fit(MinMax, X[:,1]; dims=[1,2])

n = [fit(MinMax, reduce(vcat, col); dims=nothing) for col in eachcol(X)]

@btime [fit(MinMax, reduce(vcat, col); dims=nothing) for col in eachcol(X)]
# 396.777 ms (3704 allocations: 1.49 GiB)

@btime map(col -> fit(MinMax, reduce(vcat, col); dims=nothing), eachcol(X))
# 416.954 ms (3706 allocations: 1.49 GiB)

@btime begin
    n = Vector{MinMax}(undef, size(X, 2))
    Threads.@threads for i in axes(X, 2)
        min = minimum(minimum.(@views X[:,i]))
        max = maximum(maximum.(@views X[:,i]))
        n[i] = fit(MinMax, [min, max]; dims=nothing)
    end

    Threads.@threads for idx in CartesianIndices(X)
        normalize(X[idx], n[idx[2]])
    end
end
# 73.938 ms (68391 allocations: 1.31 MiB)


using ThreadsX
@btime ThreadsX.map(col -> fit(MinMax, reduce(vcat, col); dims=nothing), eachcol(X))
# 160.248 ms (4166 allocations: 1.49 GiB)

X = [rand(200) .* 1000 for _ in 1:100, _ in 1:100]

@btime begin
    n = Vector{Tuple{Float64,Float64}}(undef, size(X, 2))
    Threads.@threads for i in axes(X, 2)
        min = minimum(minimum.(X[:,i]))
        max = maximum(maximum.(X[:,i]))
        n[i] = (min, max)
    end

    # Threads.@threads for idx in CartesianIndices(X)
    #     normalize(X[idx], n[idx[2]])
    # end
end
# 71.510 ms (61791 allocations: 1.29 MiB)

@btime begin
    colnorm = Vector{Tuple{Float64,Float64}}(undef, size(X, 2))
    Threads.@threads for i in axes(X, 2)
        col = @view X[:,i]
        min_val = minimum(minimum, col)
        max_val = maximum(maximum, col)
        colnorm[i] = (min_val, max_val)
    end
end
# 71.772 ms (2791 allocations: 80.17 KiB)

@btime begin
    itcol = Iterators.flatten.(eachcol(X))
    min_val = minimum.(itcol)
    max_val = maximum.(itcol)
end
# 209.033 ms (11 allocations: 5.91 KiB)

@btime begin
    itcol = Iterators.flatten.(eachcol(X))
    colnorm = Vector{Tuple{Float64,Float64}}(undef, length(itcol))

    Threads.@threads for i in eachindex(itcol)
        colnorm[i] = extrema(itcol[i])
    end
end
# 80.826 ms (96 allocations: 10.99 KiB)

@btime begin
    itcol = Iterators.flatten.(eachcol(X))
    colnorm = collect(extrema(itcol[i]) for i in eachindex(itcol))
end;
# 544.307 ms (10 allocations: 5.76 KiB)

using ThreadsX
@btime begin
    itcol = Iterators.flatten.(eachcol(X))
    colnorm = ThreadsX.map(extrema, itcol)
end
# 76.412 ms (323 allocations: 32.10 KiB)

@btime begin
    itcol = Iterators.flatten.(eachcol(X))
    @inbounds colnorm = map(extrema, itcol)
end
# 557.813 ms (8 allocations: 5.71 KiB)

function _minmax(
    X :: AbstractArray;
    lower :: Float64,
    upper :: Float64
)
    itcol = Iterators.flatten.(eachcol(X))
    colnorm = Vector{Tuple{Float64,Float64}}(undef, length(itcol))

    Threads.@threads for i in eachindex(itcol)
        colnorm[i] = extrema(itcol[i])
    end
end

lower = 10.0
upper = 25.0

itcol = Iterators.flatten.(eachcol(X))
colnorm = Vector{Tuple{Float64,Float64}}(undef, length(itcol))

Threads.@threads for i in eachindex(itcol)
    colnorm[i] = extrema(itcol[i])

    for j in axes(X, 1)
        @show j,i
    end
end

X = [rand(4, 3) .* 10 for _ in 1:3, _ in 1:2]

n = fit(ZScore, X[1,1])
n = fit(Sigmoid, X[1,1])
n = fit(MinMax, X[1,1])
n = fit(Center, X[1,1])
n = fit(UnitEnergy, X[1,1])
n = fit(UnitPower, X[1,1])
n = fit(HalfZScore, X[1,1])
n = fit(OutlierSuppress, X[1,1])
n = fit(MinMaxClip, X[1,1])
normalize(X[1,1], n)

a = eachslice(X, dims=2, drop=false)
b = eachslice(X, dims=2, drop=true)
c = Iterators.flatten.(eachcol(X))

@btime collect(a[1])
# 38.310 ns (3 allocations: 128 bytes)
@btime collect(b[1])
# 36.311 ns (3 allocations: 128 bytes)
@btime collect(c[1])
# 142.053 ns (6 allocations: 1.66 KiB)

@evalnorm X[1,1] zscore()

rootenergy(x::AbstractArray) = sum(abs2, x) |> sqrt
rootpower(x) = sqrt(mean(abs2, x))
unitpower(x) = Base.Fix2(/, x)
halfstd(x) = std(x) ./ convert(eltype(x), sqrt(1 - (2 / π)))

zscore()::Function          = x -> _zscore(Statistics.mean(x), Statistics.std(x))
sigmoid()::Function         = x -> _sigmoid(Statistics.mean(x), Statistics.std(x))
minmax()::Function          = x -> _minmax(minimum(x), maximum(x))
center()::Function          = x -> _center(Statistics.mean(x))
unitenergy()::Function      = x -> _unitenergy(rootenergy(x))
unitpower()::Function       = x -> _unitpoewr(rootpower(x))
halfzscore()::Function      = x -> _zscore(minimum(x), halfstd(x))
outliersuppress()::Function = x -> _outliersuppress(Statistics.mean(x), Statistics.std(x))
minmaxclip()::Function      = x -> _minmaxclip(minimum(x), maximum(x))

macro evalnorm(x, n)
    quote
        _x = $(esc(x))
        _n = $(esc(n))()
        cols = Iterators.flatten.(eachcol(_x))

        # compute normalization function for each column
        nfuncs = @inbounds [_n(collect(cols[i])) for i in eachindex(cols)]
        
        # apply to each element using CartesianIndices
        @inbounds [nfuncs[idx[2]](_x[idx]) for idx in CartesianIndices(_x)]
    end
end
# 2.257 μs (80 allocations: 2.59 KiB)

X = rand(1000,750)

function element_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    _X = Iterators.flatten(X)

    # compute normalization function for each column
    nfunc = n(collect(_X))

    # apply to each element using CartesianIndices
    [nfunc(X[idx]) for idx in CartesianIndices(X)]
end

function tabular_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))

    # compute normalization function for each column
    nfuncs = @inbounds [n(collect(cols[i])) for i in eachindex(cols)]
    
    # apply to each element using CartesianIndices
    [nfuncs[idx[2]](X[idx]) for idx in CartesianIndices(X)]
end
# 1.556 ms (2259 allocations: 11.54 MiB)

using Accessors
_minmax(l, u) = @o (_ - l) / (u - l)

_mm(l, u) = (x) -> (x - l) / (u - l)

@btime _minmax(0,1.5)(20)
# 1.608 ns (0 allocations: 0 bytes)
@btime _mm(0,1.5)(20)
# 1.608 ns (0 allocations: 0 bytes)

function normalize!(Z::AbstractArray, X::AbstractArray, T::AbstractNormalization)
    @show "PASO"
    dims = Normalization.dims(T)
    isfit(T) || fit!(T, X; dims)
    mapdims!(Z, forward(T), X, params(T); dims)
    return nothing
end

fit(UnitEnergy, X[1,1])
@evalnorm X[1,1] unitenergy

fit(UnitPower, X[1,1])
@evalnorm X[1,1] unitpower

fit(HalfZScore, X[1,1])
@evalnorm X[1,1] halfzscore

struct Gino
    a::Int
end

function q(::Type{Gino}, a::Int64)
    @show a
end

_zscore(y, o)  = (x) -> (x - y) / o # * But this needs to be mapped over SCALAR y
_sigmoid(y, o) = (x) -> inv(1 + exp(-(x - y) / o))
_minmax(l, u)  = (x) -> (x - l) / (u - l)
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

dims_val = Normalization.dims(T)
ps = params(T)
f = forward(T)
n = ndims(X)
negs = isnothing(dims_val) ? (1:n) : negdims(dims_val, n)

xs = eachslice(X[1,1]; dims=[1,2])
zs = eachslice(Z; dims=negs)
ys = eachslice.(ps; dims=negs)

Threads.@threads for i in eachindex(xs)
    y = ntuple(j -> @inbounds ys[j][i], Val(length(ps)))
    @inbounds map!(f(map(only, y)...), zs[i], xs[i])
end

# space-separated columns
x = Float64[1 2 3; 4 5 6; 7 8 9]
@btime begin
    n=fit(MinMax, x; dims=1)
    q=normalize(x,n)
end

@btime @evalnorm x minmax
# 2.301 μs (80 allocations: 2.59 KiB)

@evalnorm X minmax


### test
X = rand(1000,750)

@test_nowarn element_norm(X, zscore())
@test_nowarn element_norm(X, sigmoid())
@test_nowarn element_norm(X, rescale())
@test_nowarn element_norm(X, center())
@test_nowarn element_norm(X, unitenergy())
@test_nowarn element_norm(X, unitpower())
@test_nowarn element_norm(X, halfzscore())
@test_nowarn element_norm(X, outliersuppress())
@test_nowarn element_norm(X, minmaxclip())

X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

@test_nowarn ds_norm(X, zscore())
@test_nowarn ds_norm(X, sigmoid())
@test_nowarn ds_norm(X, rescale())
@test_nowarn ds_norm(X, center())
@test_nowarn ds_norm(X, unitenergy())
@test_nowarn ds_norm(X, unitpower())
@test_nowarn ds_norm(X, halfzscore())
@test_nowarn ds_norm(X, outliersuppress())
@test_nowarn ds_norm(X, minmaxclip())


function element_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    _X = Iterators.flatten(X)

    # compute normalization function for each column
    nfunc = n(collect(_X))

    # apply to each element using CartesianIndices
    [nfunc(X[idx]) for idx in CartesianIndices(X)]
end

function ds_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))

    # compute normalization function for each column
    nfuncs = @inbounds [n(collect(cols[i])) for i in eachindex(cols)]
    
    # apply to each element using CartesianIndices
    [element_norm(X[idx], nfuncs[idx[2]]) for idx in CartesianIndices(X)]
end
# 1.556 ms (2259 allocations: 11.54 MiB)

X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

function alement_norm(X::AbstractArray, nfunc::Base.Callable)::AbstractArray
    # Broadcast the normalization function over all elements
    nfunc.(X)
end

function ds_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))

    # Compute normalization function for each column
    nfuncs = collect(n(collect(cols[i])) for i in eachindex(cols))

    # Apply to each element using CartesianIndices
    Xn = similar(X)
    for idx in CartesianIndices(X)
        col_idx = idx[2]
        Xn[idx] = element_norm(X[idx], nfuncs[col_idx])
    end
    Xn
end



element_norm(X, zscore())
element_norm(X, sigmoid())
element_norm(X, rescale())
element_norm(X, center())
element_norm(X, unitenergy())
element_norm(X, unitpower())
element_norm(X, halfzscore())
element_norm(X, outliersuppress())
element_norm(X, minmaxclip())

alement_norm(X, zscore())
alement_norm(X, sigmoid())
alement_norm(X, rescale())
alement_norm(X, center())
alement_norm(X, unitenergy())
alement_norm(X, unitpower())
alement_norm(X, halfzscore())
alement_norm(X, outliersuppress())
alement_norm(X, minmaxclip())

@inline _ds_norm(X::AbstractArray, nfunc::Base.Callable) = nfunc.(X)

function ds_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = [n(collect(cols[i])) for i in eachindex(cols)]
    [_ds_norm(X[idx], nfuncs[idx[2]]) for idx in CartesianIndices(X)]
end
# 840.775 ms (153308 allocations: 6.11 GiB)
# 840.981 ms (33308 allocations: 6.10 GiB)

function ds_norm(X::AbstractArray, n::Base.Callable)::AbstractArray
    cols = Iterators.flatten.(eachcol(X))

    # nfuncs = [n(collect(cols[i])) for i in eachindex(cols)]

    nfuncs = Vector{Base.Callable}(undef, length(cols))
    Threads.@threads for i in 1:size(X, 2)
        nfuncs[i] = n(collect(cols[i]))
    end

    Xn = similar(X)

    # for idx in CartesianIndices(X)
    #     col_idx = idx[2]
    #     Xn[idx] = _ds_norm(X[idx], nfuncs[col_idx])
    # end
    # Xn

    nrows = size(X, 1)
    Threads.@threads for j in 1:ncols
        @inbounds nfunc_j = nfuncs[j]
        @inbounds for i in 1:nrows
            Xn[i, j] = _ds_norm(X[i, j], nfunc_j)
        end
    end
    
    Xn
end
# 862.071 ms (153308 allocations: 6.11 GiB)
# 857.530 ms (33308 allocations: 6.10 GiB)

X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

X = [rand(20, 10, 30) .* 1000 for _ in 1:10, _ in 1:100]

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

ds_norm(X, zscore())
ds_norm(X, sigmoid())
ds_norm(X, rescale())
ds_norm(X, center())
ds_norm(X, unitenergy())
ds_norm(X, unitpower())
ds_norm(X, halfzscore())
ds_norm(X, outliersuppress())
ds_norm(X, minmaxclip())

