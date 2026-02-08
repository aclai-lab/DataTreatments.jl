# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const NormType = Union{AbstractArray, Base.Iterators.Flatten}

# ---------------------------------------------------------------------------- #
#                               core functions                                 #
# ---------------------------------------------------------------------------- #
function _zscore(x::NormType; method::Symbol=:std)
    (y,o) = if method == :std
        (Statistics.mean(x), Statistics.std(x))
    elseif method == :robust
        _y = Statistics.median(x)
        (_y, Statistics.median(abs.(x .- _y)))
    elseif method == :half
        _h = std(x) ./ sqrt(1 - (2 / π))
        (minimum(x), _h)
    else
        throw(ArgumentError("method must be :std, :robust or :half, got :$method"))
    end
    (x) -> (x - y) / o
end

function _sigmoid(x::NormType)
    y, o = Statistics.mean(x), Statistics.std(x)
    (x) -> inv(1 + exp(-(x - y) / o))
end

function _pnorm(x::NormType; p::Real=2)
    x_filtered = filter(!isnan, collect(x))
    s = isempty(x_filtered) ? one(eltype(x)) : LinearAlgebra.norm(x_filtered, p)
    Base.Fix2(/, s)
end

function _scale(x::NormType; factor::Symbol=:std)
    s = if factor == :std
        Statistics.std(x)
    elseif factor == :mad
        StatsBase.mad(x; normalize=false)
    elseif factor == :first
        first(x)
    elseif factor == :iqr
        StatsBase.iqr(x)
    else
        throw(ArgumentError("factor must be :std, :mad, :first, or :iqr, got :$factor"))
    end
    
    Base.Fix2(/, s)
end

function _minmax(x::NormType; lower::Real=0.0, upper::Real=1.0)
    xmin, xmax = extrema(x)    
    scale = (upper - lower) / (xmax - xmin)
    (x) -> clamp(lower + (x - xmin) * scale, lower, upper)
end

function _center(x::NormType; method::Symbol=:mean)
    method in (:mean, :median) || throw(ArgumentError("method must be :mean or :median, got :$method"))
    y = getproperty(Statistics, method)(x)
    (x) -> x - y
end

function _unitpower(x::NormType)
    p = mean(abs2, x) |> sqrt
    Base.Fix2(/, p)
end

function _outliersuppress(x::NormType; thr::Real=0.5)
    y, o = Statistics.mean(x), Statistics.std(x)
    (x) -> abs(o) > thr * o ? y + sign(x - y) * thr * o : x
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
"""
    zscore(; [method::Symbol]) -> Function
    zscore(x::NormType; kwargs...) 

Create a z-score normalization function that standardizes data by centering and scaling.

# Arguments
- `method::Symbol`: Method for computing the z-score

# Methods

## Standard Z-Score (`:std`, default)
Centers data to mean 0 and scales to standard deviation 1.
```math
z = \\frac{x - \\mu}{\\sigma}
```
where μ is the mean and σ is the standard deviation.

## Robust Z-Score (:robust)
Centers data to median 0 and scales to median absolute deviation 1.
More resistant to outliers than standard z-score.
```math
z = \\frac{x - \\text{median}(x)}{\\text{MAD}(x)}
```
where MAD is the median absolute deviation.

## Half-Normal Z-Score (:half)
Normalizes to the standard half-normal distribution using minimum and half-standard deviation.
```math
z = \\frac{x - \\min(x)}{\\sigma_{\\text{half}}}
```
where σ_half = σ / √(1 - 2/π).

## Examples
```julia
# Standard z-score normalization
X = rand(100, 50)
X_norm = element_norm(X, zscore())
# Result: mean ≈ 0, std ≈ 1

# Robust z-score (resistant to outliers)
X_robust = element_norm(X, zscore(method=:robust))
# Result: median ≈ 0, MAD ≈ 1

# Half-normal z-score
X_half = element_norm(X, zscore(method=:half))
# Result: minimum ≈ 0, scaled by half-standard deviation
```
"""
zscore(; kwargs...) = x -> _zscore(x; kwargs...)
zscore(x::NormType; kwargs...) = _zscore(x; kwargs...)

"""
    sigmoid() -> Function
    sigmoid(x::NormType) 

Create a sigmoid normalization function that maps data to the interval (0, 1).

The sigmoid (or logistic) function provides a smooth, S-shaped transformation that 
maps the entire real line to the bounded interval (0, 1), with the steepest slope 
at the mean of the data.

# Formula
```math
\\sigma(x) = \\frac{1}{1 + e^{-\\frac{x - \\mu}{\\sigma}}}
```
where:
- μ (mu) is the mean of the input data
- σ (sigma) is the standard deviation of the input data
- The output is bounded: 0 < σ(x) < 1

## Examples
```julia
# Sigmoid normalization
X = rand(100, 50)
X_sigmoid = element_norm(X, sigmoid())
# Result: all values in (0, 1), mean(X) → 0.5
```
"""
sigmoid() = x -> _sigmoid(x)
sigmoid(x::NormType) = _sigmoid(x)

"""
    pnorm(; [p::Real]) -> Function
    pnorm(x::NormType; kwargs...) 

Create a normalization function that scales data by the p-norm.

The p-norm normalization divides each element by the p-norm of the entire dataset,
ensuring that the normalized data has unit p-norm. This is particularly useful for
standardizing data magnitudes across different scales.

# Arguments
- `p::Real`: The norm order (default: 2)
  - `p = 1`: Manhattan norm (sum of absolute values)
  - `p = 2`(default): Euclidean norm (default, root sum of squares)
  - `p = Inf`: Infinity norm (maximum absolute value)
  - `p > 0`: General p-norm

# Formula

## General p-norm (p ≥ 1):
```math
\\|x\\|_p = \\left(\\sum_{i=1}^{n} |x_i|^p\\right)^{1/p}
```

## Examples
```julia
# L2 norm (Euclidean, default)
X = rand(100, 50)
X_norm = element_norm(X, norm())
# Result: ‖X‖₂ = 1

# L1 norm (Manhattan)
X_L1 = element_norm(X, norm(p=1))
# Result: sum(abs, X) = 1

# L∞ norm (Maximum)
X_Linf = element_norm(X, norm(p=Inf))
# Result: maximum(abs, X) = 1

# Custom p-norm
X_L4 = element_norm(X, norm(p=4))
# Result: (sum(X.^4))^(1/4) = 1
```
"""
pnorm(; kwargs...) = x -> _pnorm(x; kwargs...)
pnorm(x::NormType; kwargs...) = _pnorm(x; kwargs...)


"""
    scale(; [factor::Symbol]) -> Function
    scale(x::NormType; kwargs...) 

Create a normalization function that scales data by a specified scale factor.

Scale normalization divides data by a characteristic scale measure, standardizing
the spread or magnitude without necessarily centering the data. This is useful when
you want to normalize variability but preserve the mean or baseline.

# Arguments
- `factor::Symbol`: Scale factor to use

# Scale Factor Options

## Standard Deviation (`:std`, default)
Scale data to have standard deviation of 1.
```math
x_{\\text{scaled}} = \\frac{x}{\\sigma}
```
where σ is the standard deviation.

## Median Absolute Deviation (:mad)
Scale data to have median absolute deviation of 1.
```math
x_{\\text{scaled}} = \\frac{x}{\\text{MAD}(x)}
```
where MAD = median(|x - median(x)|).

## First Element (:first)
Scale data by the value of the first element.
```math
x_{\\text{scaled}} = \\frac{x}{x_1}
```

## Interquartile Range (:iqr)
Scale data to have interquartile range of 1.
```math
x_{\\text{scaled}} = \\frac{x}{\\text{IQR}(x)}
```
where IQR = Q₃ - Q₁ (75th percentile - 25th percentile).

## Examples
```julia
# Standard deviation scaling (default)
X = rand(100, 50)
X_scaled = element_norm(X, scale())
# Result: std(X_scaled) ≈ 1, mean unchanged

# Robust scaling with MAD
X_outliers = [1, 2, 3, 4, 100]  # Has outlier
X_mad = element_norm(X_outliers, scale(factor=:mad))
# More robust than std scaling

# IQR scaling
X_iqr = element_norm(X, scale(factor=:iqr))
# Result: IQR(X_iqr) ≈ 1

# Baseline normalization (first element)
prices = [100.0, 105.0, 98.0, 110.0]
prices_norm = element_norm(prices, scale(factor=:first))
# Result: [1.0, 1.05, 0.98, 1.10] - percentage of initial price
```
"""
scale(; kwargs...) = x -> _scale(x; kwargs...)
scale(x::NormType; kwargs...) = _scale(x; kwargs...)

"""
    minmax(; [lower::Real, upper::Real]) -> Function
    minmax(x::NormType; kwargs...) 

Create a min-max normalization function that rescales data to a specified range.

# Arguments
- `lower::Real`: Lower bound of the output range (default: 0.0)
- `upper::Real`: Upper bound of the output range (default: 1.0)

# Formula
```math
x_{\\text{scaled}} = \\text{lower} + \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} \\cdot (\\text{upper} - \\text{lower})
```
This maps the original range [x_min, x_max] to [lower, upper] via affine transformation.

## Examples
```julia
# Standard min-max scaling to [0, 1]
X = rand(100, 50)
X_norm = element_norm(X, minmax())
# Result: min ≈ 0, max ≈ 1

# Custom range scaling to [-1, 1]
X_scaled = element_norm(X, minmax(lower=-1.0, upper=1.0))
# Result: min ≈ -1, max ≈ 1
```
"""
minmax(; kwargs...) = x -> _minmax(x; kwargs...)
minmax(x::NormType; kwargs...) = _minmax(x; kwargs...)

"""
    center(; [method::Symbol]) -> Function
    center(x::NormType; kwargs...) 

Create a centering normalization function that shifts data to have zero central tendency.

Centering (also known as mean/median centering or demeaning) translates data by 
subtracting a measure of central tendency, shifting the distribution without changing 
its spread or shape. This is useful for removing baseline offsets and focusing on 
relative deviations.

# Arguments
- `method::Symbol`: Centering method (default: `:mean`)
  - `:mean`(default): Center around arithmetic mean (subtracts mean)
  - `:median`: Center around median (subtracts median, more robust to outliers)

# Formula

## Mean Centering (`:mean`, default)
```math
x_{\\text{centered}} = x - \\bar{x}
```

## Median Centering (:median)
```math
x_{\\text{centered}} = x - \\text{median}(x)
```

## Examples
```julia# Mean centering (default)
X = rand(100, 50)
X_centered = element_norm(X, center())
# Result: mean(X_centered) ≈ 0, std unchanged

# Median centering
X_outliers = [1, 2, 3, 4, 100]  # Has outlier
X_med = element_norm(X_outliers, center(method=:median))
# Result: median(X_med) = 0, outlier less influential
```
"""
center(; kwargs...) = x -> _center(x; kwargs...)
center(x::NormType; kwargs...) = _center(x; kwargs...)

"""
    unitpower() -> Function
    unitpower(x::NormType) 

Create a normalization function that scales data to have unit root mean square (RMS) power.

Unit power normalization divides each element by the root mean square (RMS) of the 
entire dataset, ensuring that the normalized data has RMS = 1. This is commonly used 
in signal processing to normalize signal power.

# Formula
```math
x_{\\text{normalized}} = \\frac{x}{\\text{RMS}(x)}
```
where the Root Mean Square (RMS) is:
```math
\\text{RMS}(x) = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n} x_i^2} = \\sqrt{\\text{mean}(x^2)}
```

## Examples
```julia
# Unit power normalization
X = rand(100, 50)
X_norm = element_norm(X, unitpower())
# Result: RMS(X_norm) = 1
```
"""
unitpower() = x -> _unitpower(x)
unitpower(x::NormType) = _unitpower(x)

"""
    outliersuppress(; [thr::Real]) -> Function
    outliersuppress(x::NormType; kwargs...) 

Create a normalization function that suppresses outliers by capping values beyond a threshold.

Outlier suppression identifies values that deviate more than a specified number of 
standard deviations from the mean and replaces them with the threshold boundary value.
This technique reduces the influence of extreme values while preserving the sign and 
general structure of the data.

# Arguments
- `thr::Real=5.0`: Threshold in standard deviations (default: 5.0)

# Threshold choice
Lower thresholds more aggressively modify data
- Use thr=0.3 for typical outlier removal (3-sigma rule)
- Use thr=0.5 (default) for conservative outlier handling

# Formula
```math
x_{\\text{suppressed}} = \\begin{cases}
\\mu + \\text{thr} \\cdot \\sigma & \\text{if } x > \\mu + \\text{thr} \\cdot \\sigma \\\\
\\mu - \\text{thr} \\cdot \\sigma & \\text{if } x < \\mu - \\text{thr} \\cdot \\sigma \\\\
x & \\text{otherwise}
\\end{cases}
```
where:
- μ is the mean of the data
- σ is the standard deviation
- Values within [μ - thr·σ, μ + thr·σ] remain unchanged

## Examples
```julia
# Default threshold (0.5 standard deviations)
X = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
X_suppressed = element_norm(X, outliersuppress())
# Result: [1, 2, 3, 4, 5, ~mean+5*std] - outlier capped

# More aggressive suppression (3 std)
X_aggressive = element_norm(X, outliersuppress(thr=0.3))
# Caps values beyond mean ± 3*std (more values affected)
```
"""
outliersuppress(; kwargs...) = x -> _outliersuppress(x; kwargs...)
outliersuppress(x::NormType; kwargs...) = _outliersuppress(x; kwargs...)

# ---------------------------------------------------------------------------- #
#                                  normalize                                   #
# ---------------------------------------------------------------------------- #
@inline normalize(
    X::Union{AbstractArray{T}, AbstractArray{<:AbstractArray{T}}},
    args...; kwargs...
) where {T<:Float64} =
    normalize!(deepcopy(X), args...; kwargs...)

@inline normalize(X::AbstractArray{T}, args...; kwargs...) where {T<:Real} =
    normalize!(Float64.(X), args...; kwargs...)

@inline normalize(
    X::AbstractArray{<:AbstractArray{T}},
    args...;
    kwargs...
) where {T<:Real} =
    normalize!(convert(X; type=Float64), args...; kwargs...)

function normalize!(
    X::Union{AbstractArray{T}, AbstractArray{<:AbstractArray{T}}},
    nfunc::Base.Callable;
    tabular::Bool=false,
    dim :: Symbol=:col
) where {T<:Float64}
    return if tabular
        if dim == :col
            for i in axes(X, 2)
                X[:,i] = _normalize!(X[:,i], nfunc(Iterators.flatten(X[:,i])))
            end
        elseif dim == :row
            for i in axes(X, 1)
                X[i,:] = _normalize!(X[i,:], nfunc(Iterators.flatten(X[i,:])))
            end
        else
            throw(ArgumentError("dim must be :col or :row, got :$dim"))
        end

        X
    else
        _normalize!(X, nfunc(Iterators.flatten(X)))
    end
end

# ---------------------------------------------------------------------------- #
#                         internal normalize functions                         #
# ---------------------------------------------------------------------------- #
function _normalize!(
    X::AbstractArray{<:AbstractArray{T}},
    nfunc::Base.Callable
) where {T<:Float64}
    Threads.@threads for idx in CartesianIndices(X)
        X[idx] = nfunc.(X[idx])
    end

    return X
end

function _normalize!(
    X::AbstractArray{T},
    nfunc::Base.Callable
) where {T<:Float64}
    for idx in CartesianIndices(X)
        X[idx] = nfunc(X[idx])
    end

    return X
end

# ---------------------------------------------------------------------------- #
#                              normalize functions                             #
# ---------------------------------------------------------------------------- #
"""
    element_norm(X::AbstractArray, n::Base.Callable) -> AbstractArray

Normalize a single array element using global statistics computed across all elements.

# Arguments
- `X::AbstractArray`: Input array of any dimension (vector, matrix, tensor, etc.)
- `n::Base.Callable`: Normalization function constructor that computes parameters from data

# Examples
```julia
X = rand(100, 50)
X_norm = element_norm(X, zscore())      # Z-score normalization
X_norm = element_norm(X, minmax())     # Min-max scaling
X_norm = element_norm(X, center())      # Mean centering
```
"""
function element_norm(X::AbstractArray{T}, n::Base.Callable)::AbstractArray where {T<:AbstractFloat}
    _X = Iterators.flatten(X)
    nfunc = n(collect(_X))
    [nfunc(X[idx]) for idx in CartesianIndices(X)]
end
element_norm(X::AbstractArray{T}, args...) where {T<:Real} = element_norm(Float64.(X), args...)

"""
    tabular_norm(X::AbstractArray, n::Base.Callable; [dim::Symbol=:col]) -> AbstractArray

Normalize a tabular array by computing separate normalization parameters for each column or row.

# Arguments
- `X::AbstractArray`: Input array (typically a matrix where columns represent features)
- `n::Base.Callable`: Normalization function constructor (e.g., `zscore()`, `minmax()`)
- `dim::Symbol=:col`: Dimension along which to normalize
  - `:col` (default): Normalize each column independently (column-wise)
  - `:row`: Normalize each row independently (row-wise)

# Examples
```julia
# Column-wise normalization (default) - each feature normalized independently
X = rand(100, 50)  # 100 samples, 50 features
X_norm = tabular_norm(X, zscore())
# Each column: mean ≈ 0, std ≈ 1

# Row-wise normalization - each sample normalized independently
X_scaled = tabular_norm(X, minmax(); dim=:row)
# Each row scaled to [0, 1] independently
```

# Notes
- Each column/row uses only its own statistics, not global statistics
- Automatically converts Real arrays to Float64
"""
function tabular_norm(
    X   :: AbstractArray{T},
    n   :: Base.Callable;
    dim :: Symbol=:col
)::AbstractArray where {T<:AbstractFloat}
    dim in (:col, :row) || throw(ArgumentError("dim must be :col or :row, got :$dim"))

    dim == :row && (X = X')
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = @inbounds [n(collect(cols[i])) for i in eachindex(cols)]
    dim == :row ? [nfuncs[idx[2]](X[idx]) for idx in CartesianIndices(X)]' :
                  [nfuncs[idx[2]](X[idx]) for idx in CartesianIndices(X)]
end
tabular_norm(X::AbstractArray{T}, args...; kwargs...) where {T<:Real} = 
    tabular_norm(Float64.(X), args...;kwargs...)

""" 
    grouped_norm(X::AbstractArray, n::Base.Callable; featvec::Vector) -> AbstractArray

Normalize grouped columns of a dataset by applying the same normalization
coefficient to all columns generated by the same feature/transform.

# Arguments
- `X::AbstractArray{<:AbstractFloat}`: Tabular dataset (observations × features).
- `n::Base.Callable`: Normalization constructor (e.g. `zscore()`, `minmax()`).
- `featvec::Vector{<:Base.Callable}`: Feature functions associated with each
  column in `X`. Columns sharing the same callable form a group and reuse the
  same normalization statistics.

# Examples
```julia
# Dataset with features from 2 variables and 2 transforms
X = rand(100, 4)  # 100 samples, 4 features
featvec = [mean, mean, std, std]  # First 2 cols are means, last 2 are stds

# Normalize: all mean-based columns share normalization, std-based share another
X_norm = grouped_norm(X, zscore(); featvec)
# Columns 1-2: normalized together (mean ≈ 0, std ≈ 1 across both)
# Columns 3-4: normalized together (mean ≈ 0, std ≈ 1 across both)

# With windowed features
win = splitwindow(nwindows=3)
features = (mean, maximum)
dt = DataTreatment(X_nested, :aggregate; win=(win,), features)
# Each variable produces 3×2=6 features: mean_w1, mean_w2, mean_w3, max_w1, max_w2, max_w3

X_grouped = grouped_norm(dt.dataset, minmax(); featvec=get_vecfeatures(dt.featureid))
# All mean features normalized together, all max features normalized together
```

# Notes
- Returns a new array; use `grouped_norm!` for in-place normalization
- Columns are grouped by feature function identity
- Each group computes normalization statistics from all values across its columns
"""
function grouped_norm(
    X       :: AbstractArray{T},
    n       :: Base.Callable;
    featvec :: Vector{<:Base.Callable}
)::AbstractArray where {T<:AbstractFloat}
    Xn = copy(X)
    grouped_norm!(Xn, n; featvec)
    return Xn
end
grouped_norm(X::AbstractArray{T}, args...; kwargs...) where {T<:Real} =
    grouped_norm(Float64.(X), args...; kwargs...)

@inline function _ds_norm!(Xn::AbstractArray, X::AbstractArray, nfunc)
    @inbounds @simd for i in eachindex(X, Xn)
        Xn[i] = nfunc(X[i])
    end
    return Xn
end

"""
    grouped_norm!(X::AbstractArray, n::Base.Callable; featvec::Vector) -> Nothing

In-place version of [`grouped_norm`](@ref). Modifies `X` directly by normalizing grouped 
columns using the same normalization function for columns sharing the same feature type.
"""
function grouped_norm!(
    X       :: AbstractArray{T},
    n       :: Base.Callable;
    featvec :: Vector{<:Base.Callable}
)::Nothing where {T<:AbstractFloat}
    groups = Dict{Base.Callable, Vector{Int}}()
    for (col_idx, feat) in enumerate(featvec)
        push!(get!(groups, feat, Int[]), col_idx)
    end
    group_pairs = collect(pairs(groups))
    nrows = size(X, 1)

    Threads.@threads for gidx in eachindex(group_pairs)
        _, cols = group_pairs[gidx]
        nfunc = n(@views [X[row, col] for col in cols, row in 1:nrows])

        @inbounds for col in cols, row in 1:nrows
            X[row, col] = nfunc(X[row, col])
        end
    end

    return nothing
end

"""
    ds_norm(X::AbstractArray{<:AbstractArray}, n::Base.Callable) -> AbstractArray

Normalize a dataset composed of n-dimensional elements (e.g., sequences, time series,
or images) by computing normalization parameters for each column of the outer array.

# Arguments
- `X::AbstractArray{<:AbstractArray}`: Nested array structure where each element is an array
  - Outer array typically represents (samples × features/channels)
  - Inner arrays can be vectors, matrices, or tensors
- `n::Base.Callable`: Normalization function constructor (e.g., `zscore()`, `minmax()`)

# Examples
```julia
# Time series dataset: each element is a time series
X = [rand(100) for _ in 1:50, _ in 1:3]  # 50 samples × 3 channels
X_norm = ds_norm(X, zscore())
# Each channel normalized across all 50 time series

# Image dataset: each element is an image (matrix)
images = [rand(28, 28) for _ in 1:100, _ in 1:1]  # 100 grayscale images
images_norm = ds_norm(images, minmax())
# All images normalized using global min/max
```
"""
function ds_norm(
    X :: AbstractArray{T},
    n :: Base.Callable
)::AbstractArray where {T<:AbstractArray{<:AbstractFloat}}
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
ds_norm(X::AbstractArray{T}, args...) where {T<:AbstractArray{<:Real}} = 
    ds_norm([Float64.(x) for x in X], args...)