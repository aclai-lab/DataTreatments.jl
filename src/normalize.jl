# ---------------------------------------------------------------------------- #
#                                extend methods                                #
# ---------------------------------------------------------------------------- #
@_Normalization ZScoreRobust (median, (x)->median(abs.(x .- median(x)))) zscore

function(::Type{ZScore})(method::Symbol)
    return if method == :robust
        ZScoreRobust
    elseif method == :half
        HalfZScore
    else
        ZScore
    end
end

# function _pnorm(x::NormType; p::Real=2)
#     x_filtered = filter(!isnan, collect(x))
#     s = isempty(x_filtered) ? one(eltype(x)) : LinearAlgebra.norm(x_filtered, p)
#     Base.Fix2(/, s)
# end

@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale
@_Normalization ScaleIqr (iqr,) scale

scale(s) = Base.Fix2(/, s)

function(::Type{Scale})(factor::Symbol=:std)
    return if factor == :mad
        ScaleMad
    elseif factor == :first
        ScaleFirst
    elseif factor == :iqr
        ScaleIqr
    else
        Scale
    end
end


# # ---------------------------------------------------------------------------- #
# #                                    types                                     #
# # ---------------------------------------------------------------------------- #
# const NormType = Union{AbstractArray, Base.Iterators.Flatten}
# const FloatType = Union{Float32, Float64}

# # ---------------------------------------------------------------------------- #
# #                               core functions                                 #
# # ---------------------------------------------------------------------------- #
# function _zscore(x::NormType; method::Symbol=:std)
#     (y,o) = if method == :std
#         (Statistics.mean(x), Statistics.std(x))
#     elseif method == :robust
#         _y = Statistics.median(x)
#         (_y, Statistics.median(abs.(x .- _y)))
#     elseif method == :half
#         _h = std(x) ./ sqrt(1 - (2 / π))
#         (minimum(x), _h)
#     else
#         throw(ArgumentError("method must be :std, :robust or :half, got :$method"))
#     end
#     (x) -> (x - y) / o
# end

# function _sigmoid(x::NormType)
#     y, o = Statistics.mean(x), Statistics.std(x)
#     (x) -> inv(1 + exp(-(x - y) / o))
# end

# function _pnorm(x::NormType; p::Real=2)
#     x_filtered = filter(!isnan, collect(x))
#     s = isempty(x_filtered) ? one(eltype(x)) : LinearAlgebra.norm(x_filtered, p)
#     Base.Fix2(/, s)
# end

# function _scale(x::NormType; factor::Symbol=:std)
#     s = if factor == :std
#         Statistics.std(x)
#     elseif factor == :mad
#         StatsBase.mad(x; normalize=false)
#     elseif factor == :first
#         first(x)
#     elseif factor == :iqr
#         StatsBase.iqr(x)
#     else
#         throw(ArgumentError("factor must be :std, :mad, :first, or :iqr, got :$factor"))
#     end
    
#     Base.Fix2(/, s)
# end

# function _minmax(x::NormType; lower::Real=0.0, upper::Real=1.0)
#     xmin, xmax = extrema(x)    
#     scale = (upper - lower) / (xmax - xmin)
#     (x) -> clamp(lower + (x - xmin) * scale, lower, upper)
# end

# function _center(x::NormType; method::Symbol=:mean)
#     method in (:mean, :median) || throw(ArgumentError("method must be :mean or :median, got :$method"))
#     y = getproperty(Statistics, method)(x)
#     (x) -> x - y
# end

# function _unitpower(x::NormType)
#     p = mean(abs2, x) |> sqrt
#     Base.Fix2(/, p)
# end

# function _outliersuppress(x::NormType; thr::Real=0.5)
#     y, o = Statistics.mean(x), Statistics.std(x)
#     (x) -> abs(o) > thr * o ? y + sign(x - y) * thr * o : x
# end

# # ---------------------------------------------------------------------------- #
# #                                   methods                                    #
# # ---------------------------------------------------------------------------- #
# """
#     zscore(; [method::Symbol]) -> Function
#     zscore(x::NormType; kwargs...) 

# Create a z-score normalization function that standardizes data by centering and scaling.

# # Arguments
# - `method::Symbol`: Method for computing the z-score

# # Methods

# ## Standard Z-Score (`:std`, default)
# Centers data to mean 0 and scales to standard deviation 1.
# ```math
# z = \\frac{x - \\mu}{\\sigma}
# ```
# where μ is the mean and σ is the standard deviation.

# ## Robust Z-Score (:robust)
# Centers data to median 0 and scales to median absolute deviation 1.
# More resistant to outliers than standard z-score.
# ```math
# z = \\frac{x - \\text{median}(x)}{\\text{MAD}(x)}
# ```
# where MAD is the median absolute deviation.

# ## Half-Normal Z-Score (:half)
# Normalizes to the standard half-normal distribution using minimum and half-standard deviation.
# ```math
# z = \\frac{x - \\min(x)}{\\sigma_{\\text{half}}}
# ```
# where σ_half = σ / √(1 - 2/π).

# ## Examples
# ```julia
# X = rand(100, 50)

# # Standard z-score normalization
# X_norm = DataTreatments.normalize(X, zscore())
# # Result: mean ≈ 0, std ≈ 1

# # Robust z-score (resistant to outliers)
# X_robust = DataTreatments.normalize(X, zscore(method=:robust))
# # Result: median ≈ 0, MAD ≈ 1

# # Half-normal z-score
# X_half = DataTreatments.normalize(X, zscore(method=:half))
# # Result: minimum ≈ 0, scaled by half-standard deviation
# ```
# """
# zscore(; kwargs...) = x -> _zscore(x; kwargs...)
# zscore(x::NormType; kwargs...) = _zscore(x; kwargs...)

# """
#     sigmoid() -> Function
#     sigmoid(x::NormType) 

# Create a sigmoid normalization function that maps data to the interval (0, 1).

# The sigmoid (or logistic) function provides a smooth, S-shaped transformation that 
# maps the entire real line to the bounded interval (0, 1), with the steepest slope 
# at the mean of the data.

# # Formula
# ```math
# \\sigma(x) = \\frac{1}{1 + e^{-\\frac{x - \\mu}{\\sigma}}}
# ```
# where:
# - μ (mu) is the mean of the input data
# - σ (sigma) is the standard deviation of the input data
# - The output is bounded: 0 < σ(x) < 1

# ## Example
# ```julia
# X = rand(100, 50)

# # Sigmoid normalization
# X_sigmoid = DataTreatments.normalize(X, sigmoid())
# # Result: all values in (0, 1), mean(X) → 0.5
# ```
# """
# sigmoid() = x -> _sigmoid(x)
# sigmoid(x::NormType) = _sigmoid(x)

# """
#     pnorm(; [p::Real]) -> Function
#     pnorm(x::NormType; kwargs...) 

# Create a normalization function that scales data by the p-norm.

# The p-norm normalization divides each element by the p-norm of the entire dataset,
# ensuring that the normalized data has unit p-norm. This is particularly useful for
# standardizing data magnitudes across different scales.

# # Arguments
# - `p::Real`: The norm order (default: 2)
#   - `p = 1`: Manhattan norm (sum of absolute values)
#   - `p = 2`(default): Euclidean norm (default, root sum of squares)
#   - `p = Inf`: Infinity norm (maximum absolute value)
#   - `p > 0`: General p-norm

# # Formula

# ## General p-norm (p ≥ 1):
# ```math
# \\|x\\|_p = \\left(\\sum_{i=1}^{n} |x_i|^p\\right)^{1/p}
# ```

# ## Examples
# ```julia
# X = rand(100, 50)

# # L2 norm (Euclidean, default)
# X_norm = DataTreatments.normalize(X, pnorm())
# # Result: ‖X‖₂ = 1

# # L1 norm (Manhattan)
# X_L1 = DataTreatments.normalize(X, pnorm(p=1))
# # Result: sum(abs, X) = 1

# # L∞ norm (Maximum)
# X_Linf = DataTreatments.normalize(X, pnorm(p=Inf))
# # Result: maximum(abs, X) = 1

# # Custom p-norm
# X_L4 = DataTreatments.normalize(X, pnorm(p=4))
# # Result: (sum(X.^4))^(1/4) = 1
# ```
# """
# pnorm(; kwargs...) = x -> _pnorm(x; kwargs...)
# pnorm(x::NormType; kwargs...) = _pnorm(x; kwargs...)


# """
#     scale(; [factor::Symbol]) -> Function
#     scale(x::NormType; kwargs...) 

# Create a normalization function that scales data by a specified scale factor.

# Scale normalization divides data by a characteristic scale measure, standardizing
# the spread or magnitude without necessarily centering the data. This is useful when
# you want to normalize variability but preserve the mean or baseline.

# # Arguments
# - `factor::Symbol`: Scale factor to use

# # Scale Factor Options

# ## Standard Deviation (`:std`, default)
# Scale data to have standard deviation of 1.
# ```math
# x_{\\text{scaled}} = \\frac{x}{\\sigma}
# ```
# where σ is the standard deviation.

# ## Median Absolute Deviation (:mad)
# Scale data to have median absolute deviation of 1.
# ```math
# x_{\\text{scaled}} = \\frac{x}{\\text{MAD}(x)}
# ```
# where MAD = median(|x - median(x)|).

# ## First Element (:first)
# Scale data by the value of the first element.
# ```math
# x_{\\text{scaled}} = \\frac{x}{x_1}
# ```

# ## Interquartile Range (:iqr)
# Scale data to have interquartile range of 1.
# ```math
# x_{\\text{scaled}} = \\frac{x}{\\text{IQR}(x)}
# ```
# where IQR = Q₃ - Q₁ (75th percentile - 25th percentile).

# ## Examples
# ```julia
# X = rand(100, 50)

# # Standard deviation scaling (default)
# X_scaled = DataTreatments.normalize(X, scale())
# # Result: std(X_scaled) ≈ 1, mean unchanged

# X_outliers = [1, 2, 3, 4, 100]  # Has outlier

# # Robust scaling with MAD
# X_mad = DataTreatments.normalize(X_outliers, scale(factor=:mad))
# # More robust than std scaling

# # IQR scaling
# X_iqr = DataTreatments.normalize(X, scale(factor=:iqr))
# # Result: IQR(X_iqr) ≈ 1

# prices = [100.0, 105.0, 98.0, 110.0]

# # Baseline normalization (first element)
# prices_norm = DataTreatments.normalize(prices, scale(factor=:first))
# # Result: [1.0, 1.05, 0.98, 1.10] - percentage of initial price
# ```
# """
# scale(; kwargs...) = x -> _scale(x; kwargs...)
# scale(x::NormType; kwargs...) = _scale(x; kwargs...)

# """
#     minmax(; [lower::Real, upper::Real]) -> Function
#     minmax(x::NormType; kwargs...) 

# Create a min-max normalization function that rescales data to a specified range.

# # Arguments
# - `lower::Real`: Lower bound of the output range (default: 0.0)
# - `upper::Real`: Upper bound of the output range (default: 1.0)

# # Formula
# ```math
# x_{\\text{scaled}} = \\text{lower} + \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} \\cdot (\\text{upper} - \\text{lower})
# ```
# This maps the original range [x_min, x_max] to [lower, upper] via affine transformation.

# ## Examples
# ```julia
# X = rand(100, 50)

# # Standard min-max scaling to [0, 1]
# X_norm = DataTreatments.normalize(X, DataTreatments.minmax())
# # Result: min ≈ 0, max ≈ 1

# # Custom range scaling to [-1, 1]
# X_scaled = DataTreatments.normalize(X, DataTreatments.minmax(lower=-1.0, upper=1.0))
# # Result: min ≈ -1, max ≈ 1
# ```
# """
# minmax(; kwargs...) = x -> _minmax(x; kwargs...)
# minmax(x::NormType; kwargs...) = _minmax(x; kwargs...)

# """
#     center(; [method::Symbol]) -> Function
#     center(x::NormType; kwargs...) 

# Create a centering normalization function that shifts data to have zero central tendency.

# Centering (also known as mean/median centering or demeaning) translates data by 
# subtracting a measure of central tendency, shifting the distribution without changing 
# its spread or shape. This is useful for removing baseline offsets and focusing on 
# relative deviations.

# # Arguments
# - `method::Symbol`: Centering method (default: `:mean`)
#   - `:mean`(default): Center around arithmetic mean (subtracts mean)
#   - `:median`: Center around median (subtracts median, more robust to outliers)

# # Formula

# ## Mean Centering (`:mean`, default)
# ```math
# x_{\\text{centered}} = x - \\bar{x}
# ```

# ## Median Centering (:median)
# ```math
# x_{\\text{centered}} = x - \\text{median}(x)
# ```

# ## Examples
# ```julia
# X = rand(100, 50)

# # Mean centering (default)
# X_centered = DataTreatments.normalize(X, center())
# # Result: mean(X_centered) ≈ 0, std unchanged

# X_outliers = [1, 2, 3, 4, 100]  # Has outlier

# # Median centering
# X_med = DataTreatments.normalize(X_outliers, center(method=:median))
# # Result: median(X_med) = 0, outlier less influential
# ```
# """
# center(; kwargs...) = x -> _center(x; kwargs...)
# center(x::NormType; kwargs...) = _center(x; kwargs...)

# """
#     unitpower() -> Function
#     unitpower(x::NormType) 

# Create a normalization function that scales data to have unit root mean square (RMS) power.

# Unit power normalization divides each element by the root mean square (RMS) of the 
# entire dataset, ensuring that the normalized data has RMS = 1. This is commonly used 
# in signal processing to normalize signal power.

# # Formula
# ```math
# x_{\\text{normalized}} = \\frac{x}{\\text{RMS}(x)}
# ```
# where the Root Mean Square (RMS) is:
# ```math
# \\text{RMS}(x) = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n} x_i^2} = \\sqrt{\\text{mean}(x^2)}
# ```

# ## Example
# ```julia
# X = rand(100, 50)

# # Unit power normalization
# X_norm = DataTreatments.normalize(X, unitpower())
# # Result: RMS(X_norm) = 1
# ```
# """
# unitpower() = x -> _unitpower(x)
# unitpower(x::NormType) = _unitpower(x)

# """
#     outliersuppress(; [thr::Real]) -> Function
#     outliersuppress(x::NormType; kwargs...) 

# Create a normalization function that suppresses outliers by capping values beyond a threshold.

# Outlier suppression identifies values that deviate more than a specified number of 
# standard deviations from the mean and replaces them with the threshold boundary value.
# This technique reduces the influence of extreme values while preserving the sign and 
# general structure of the data.

# # Arguments
# - `thr::Real=5.0`: Threshold in standard deviations (default: 5.0)

# # Threshold choice
# Lower thresholds more aggressively modify data
# - Use thr=0.3 for typical outlier removal (3-sigma rule)
# - Use thr=0.5 (default) for conservative outlier handling

# # Formula
# ```math
# x_{\\text{suppressed}} = \\begin{cases}
# \\mu + \\text{thr} \\cdot \\sigma & \\text{if } x > \\mu + \\text{thr} \\cdot \\sigma \\\\
# \\mu - \\text{thr} \\cdot \\sigma & \\text{if } x < \\mu - \\text{thr} \\cdot \\sigma \\\\
# x & \\text{otherwise}
# \\end{cases}
# ```
# where:
# - μ is the mean of the data
# - σ is the standard deviation
# - Values within [μ - thr·σ, μ + thr·σ] remain unchanged

# ## Examples
# ```julia
# X = [1, 2, 3, 4, 5, 100]  # 100 is an outlier

# # Default threshold (0.5 standard deviations)
# X_suppressed = DataTreatments.normalize(X, outliersuppress())
# # Result: [1, 2, 3, 4, 5, ~mean+5*std] - outlier capped

# # More aggressive suppression (3 std)
# X_aggressive = DataTreatments.normalize(X, outliersuppress(thr=0.3))
# # Caps values beyond mean ± 3*std (more values affected)
# ```
# """
# outliersuppress(; kwargs...) = x -> _outliersuppress(x; kwargs...)
# outliersuppress(x::NormType; kwargs...) = _outliersuppress(x; kwargs...)

# # ---------------------------------------------------------------------------- #
# #                                  normalize                                   #
# # ---------------------------------------------------------------------------- #
# """
#     normalize(X, nfunc; tabular=false, dims=0)
#     normalize(df::DataFrame, nfunc; tabular=false, dims=0)
#     normalize(df::Vector{DataFrame}, nfunc; tabular=false, dims=0)
#     normalize!(X, nfunc; tabular=false, dims=0)

# Apply a normalization function to data.

# ## Supported inputs
# - `AbstractArray{<:Real}`: any numeric array (vector, matrix, or N‑D array)
# - `AbstractArray{<:AbstractArray{<:Real}}`: nested arrays (e.g., dataset of vectors or matrices)
# - `DataFrame`: tabular data with numeric columns (normalized column‑wise by default, `dims=2`)
# - `Vector{DataFrame}`: multiple dataframes to normalize together

# The `normalize` variants allocate a new array; `normalize!` mutates in place.

# ## Available normalization methods
# - [`zscore`](@ref): Standardize by centering and scaling (mean=0, std=1)
# - [`sigmoid`](@ref): Map data to (0, 1) using logistic function
# - [`pnorm`](@ref): Scale by p-norm (L1, L2, L∞, etc.)
# - [`scale`](@ref): Scale by characteristic measure (std, MAD, IQR, first element)
# - [`minmax`](@ref): Rescale to specified range [lower, upper]
# - [`center`](@ref): Shift to zero mean or median
# - [`unitpower`](@ref): Scale to unit RMS power
# - [`outliersuppress`](@ref): Cap outliers beyond threshold

# ## Whole‑dataset normalization (default)
# By default, `dims=0`, the normalization function is computed on the *entire*
# dataset (flattened) and then applied element‑wise. This works for:
# - N‑D arrays (e.g., tensors)
# - arrays of vectors (audio segments)
# - arrays of matrices (images)

# ## Tabular normalization (row/column‑wise: `dims=1` or `dims=2`)
# For tabular data (rows/columns as samples/features), set `dims` to the dimension:
# - `dims=1`: normalize each row independently
# - `dims=2`: normalize each column independently (typical: each column is a feature)

# ## Examples
# ```julia
# X = rand(100, 50)

# # Whole‑dataset zscore
# X_all = DataTreatments.normalize(X, zscore())

# # Column‑wise zscore (each feature independently)
# X_col = DataTreatments.normalize(X, zscore(); dims=2)

# # Row‑wise minmax
# X_row = DataTreatments.normalize(X, minmax(); dims=1)

# # Nested dataset (e.g., images)
# imgs = [rand(28,28) for _ in 1:100]
# imgs_norm = DataTreatments.normalize(imgs, unitpower())
# ```

# ```julia
# using DataFrames

# df = DataFrame(
#     age = [25, 30, 35, 40, 45],
#     salary = [50000, 60000, 75000, 80000, 90000],
#     score = [85, 90, 88, 92, 87]
# )

# # Normalize all numeric columns independently (column-wise)
# df_norm = DataTreatments.normalize(df, zscore())

# # Aggregate specific columns
# fileds = [:salary, :score]
# groups = DataTreatments.groupby(df, fileds)

# df_norm = DataTreatments.normalize(groups, zscore(); dims=0)
# ```
# """
# @inline normalize(
#     X::Union{AbstractArray{T}, AbstractArray{<:AbstractArray{T}}},
#     args...; kwargs...
# ) where {T<:FloatType} =
#     normalize!(deepcopy(X), args...; kwargs...)

# @inline normalize(X::AbstractArray{T}, args...; kwargs...) where {T<:Real} =
#     normalize!(Float64.(X), args...; kwargs...)

# @inline normalize(
#     X::AbstractArray{<:AbstractArray{T}},
#     args...;
#     kwargs...
# ) where {T<:Real} =
#     normalize!(convert(X; type=Float64), args...; kwargs...)

# normalize(dfs::Vector{DataFrame}, args...; kwargs...) =
#     [normalize(df, args...; kwargs...) for df in dfs]

# # dataframes are normalized column‑wise by default
# function normalize(df::DataFrame, args...; dims::Int64=2, kwargs...)
#     colnames = propertynames(df)
#     X = Matrix{Float64}(df)
#     Xresult = normalize!(X, args...; dims, kwargs...)
    
#     return DataFrame(Xresult, colnames)
# end

# function normalize!(
#     X::Union{AbstractArray{T}, AbstractArray{<:AbstractArray{T}}},
#     nfunc::Base.Callable;
#     dims::Int64=0
# ) where {T<:FloatType}
#     if dims == 0
#         # whole-dataset normalization
#         # nfunc coefficient is obtained spanning the whole dataset
#         return _normalize!(X, nfunc(Iterators.flatten(X)))
#     elseif dims == 1
#         # row-wise normalization for tabular data
#         for i in axes(X, 1)
#             # build normalization function from each row's stats
#             # nfunc coeff is obtained flattering the whole row
#             X[i,:] = _normalize!(X[i,:], nfunc(Iterators.flatten(X[i,:])))
#         end
#         return X
#     elseif dims == 2
#         # column-wise normalization for tabular data
#         for i in axes(X, 2)
#             # build normalization function from each column's stats
#             # nfunc coeff is obtained flattering the whole column
#             X[:,i] = _normalize!(X[:,i], nfunc(Iterators.flatten(X[:,i])))
#         end
#         return X
#     else
#         throw(ArgumentError("dim must be 0 no dims, 1 row-wise or 2 col-wise."))
#     end
# end

# # ---------------------------------------------------------------------------- #
# #                         internal normalize functions                         #
# # ---------------------------------------------------------------------------- #
# function _normalize!(
#     X::AbstractArray{<:AbstractArray{T}},
#     nfunc::Base.Callable
# ) where {T<:FloatType}
#     # normalize each nested element (e.g., each vector/matrix) in parallel
#     Threads.@threads for idx in CartesianIndices(X)
#         # apply normalization coefficient to every element
#         X[idx] = nfunc.(X[idx])
#     end

#     return X
# end

# function _normalize!(
#     X::AbstractArray{T},
#     nfunc::Base.Callable
# ) where {T<:FloatType}
#     # in-place element-wise normalization for numeric arrays
#     for idx in CartesianIndices(X)
#         # apply normalization coefficient to every element
#         X[idx] = nfunc(X[idx])
#     end

#     return X
# end

# # TODO pass FloatType
