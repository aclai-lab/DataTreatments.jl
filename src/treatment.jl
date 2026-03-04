# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# extract window ranges from intervals and cartesian index
@inline function get_window_ranges(intervals::Tuple, cartidx::CartesianIndex)
    ntuple(i -> intervals[i][cartidx[i]], length(intervals))
end

"""
    is_multidim_dataset(X::Union{AbstractArray, AbstractDataFrame}) -> Bool

Return true if any feature column contains array-valued elements, indicating a
multidimensional dataset that requires aggregation or size-reduction.

Arguments
- X: A matrix-like object or a DataFrame. Each column is inspected; if
  `eltype(col) <: AbstractArray` for any column, the function returns true.
"""
@inline is_multidim_dataset(X::Union{AbstractArray, AbstractDataFrame})::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

"""
    has_uniform_element_size(X::AbstractDataFrame) -> Bool
    has_uniform_element_size(X::AbstractArray) -> Bool

Return `true` if every element in the Array or DataFrame has the same size (shape), `false` otherwise.
This is useful for deciding whether window definitions can be reused across all entries.

- Empty Array or DataFrames return `true`.
- Short-circuits on the first mismatch.
- Inspects all columns and all elements.
"""
@inline function has_uniform_element_size(X::AbstractDataFrame)
    isempty(X) && return true
    refsize = nothing
    @inbounds for col in eachcol(X)
        for elem in col
            if !ismissing(elem) && !(elem isa AbstractFloat && isnan(elem))
                refsize = size(elem)
                break
            end
        end
        !isnothing(refsize) && break
    end
    isnothing(refsize) && return true  # all missing/NaN
    @inbounds for col in eachcol(X)
        for elem in col
            ismissing(elem) && continue
            elem isa AbstractFloat && isnan(elem) && continue
            size(elem) == refsize || return false
        end
    end
    return true
end

@inline function has_uniform_element_size(X::AbstractArray)
    isempty(X) && return true
    refsize = nothing
    @inbounds for x in X
        if !ismissing(x) && !(x isa AbstractFloat && isnan(x))
            refsize = size(x)
            break
        end
    end
    isnothing(refsize) && return true  # all missing/NaN
    @inbounds for x in X
        ismissing(x) && continue
        x isa AbstractFloat && isnan(x) && continue
        size(x) == refsize || return false
    end
    return true
end

"""
    safe_feat(v, f) -> Any

Apply a feature function to a vector while safely handling missing values and NaN entries.

This function filters out missing values and NaN entries before applying the feature function,
making it robust for incomplete or noisy data. It's commonly used in aggregation and size-reduction
operations on windowed data.

# Arguments
- `v::AbstractVector`: Input vector that may contain missing values or NaN entries
- `f::Callable`: Feature function to apply (e.g., `mean`, `std`, `maximum`)

# Returns
- Result of applying function `f` to the cleaned vector (typically a scalar)

# Details
- Skips all `missing` values using `skipmissing()`
- Filters out `NaN` values (for floating-point types)
- Generator expression is passed to `f` for memory efficiency
- If all values are missing/NaN, `f` receives an empty iterator
"""
@inline function safe_feat(v, f)
    f(collect(x for x in skipmissing(v) if !(x isa AbstractFloat && isnan(x))))
end

# ---------------------------------------------------------------------------- #
#                             aggregate functions                              #
# ---------------------------------------------------------------------------- #
# Flatten nested arrays by applying multiple feature functions to windowed regions.

# This function takes a matrix of arrays (e.g., `Matrix{Matrix{Float64}}`) and produces a single
# flat matrix where each row corresponds to a row in `X`, and columns contain flattened features
# computed from windowed aggregations.

# # Use Case
# This function enables the analysis of datasets containing n-dimensional elements (such as images, 
# time series, or spectrograms) with machine learning models that require tabular input. It transforms 
# complex multi-dimensional data into a flat feature matrix suitable for standard ML algorithms.

# # Arguments
# - `X::AbstractArray`: Input array where each element is an array (e.g., `Matrix{Matrix{Float64}}`)
# - `intervals::Tuple{Vararg{Vector{UnitRange{Int}}}}`: Window definitions for aggregation
# - `features::Tuple{Vararg{Base.Callable}}=(mean,)`: Tuple of functions to compute on each window

# # Returns
# - `AbstractArray`: Flattened array with dimensions `(nrows(X), ncols(X) × n_features × n_windows)`
#   where each element is of type `core_eltype(X)`

# # Details
# The output columns are organized as: for each column in `X`, all features are computed for all windows,
# with results concatenated in the order: `[col1_feat1_win1, col1_feat1_win2, col1_feat2_win1, ...]`
function aggregate(
    X::AbstractArray,
    win::Tuple{Vararg{Base.Callable}},
    features::Tuple{Vararg{Base.Callable}},
    idx::AbstractVector{Vector{Int}},
    float_type::Type
)
    colwin = [[n > length(win) ? last(win) : win[n] for n in 1:ndims(X[first(idx[i]), i])] for i in axes(X, 2)]
    nwindows = [prod(hasfield(typeof(w), :nwindows) ? w.nwindows : 1 for w in c) for c in colwin]
    nfeats = length(features)

    Xa = Matrix{Union{Missing,float_type}}(undef, size(X, 1), sum(nwindows) * nfeats)
    outtmp = 1

    @inbounds for colidx in axes(X, 2)
        outidx = outtmp

        for rowidx in axes(X, 1)
            x = X[rowidx, colidx]
            outidx = outtmp

            if rowidx in idx[colidx]
                intervals = @evalwindow X[rowidx, colidx] colwin[colidx]...
                for feat in features
                    for cartidx in CartesianIndices(length.(intervals))
                        ranges = get_window_ranges(intervals, cartidx)
                        window_view = @views x[ranges...]
                        Xa[rowidx, outidx] = safe_feat(reshape(window_view, :), feat)
                        outidx += 1
                    end
                end
            else
                intervals = nwindows[colidx] * nfeats
                Xa[rowidx, outidx:outidx+intervals-1] .= ismissing(x) ? x : float_type(x)
                outidx += intervals
            end
        end

        outtmp = outidx
    end

    return Xa, nwindows
end

# ---------------------------------------------------------------------------- #
#                             reducesize functions                              #
# ---------------------------------------------------------------------------- #
# Apply window-based size-reduction to each element of an array.

# This function is designed for arrays where each element is itself an array (e.g., `Matrix{Matrix{Float64}}`).

# # Arguments
# - `X::AbstractArray`: Input array where each element is an array to be size-reduced
# - `intervals::Tuple{Vararg{Vector{UnitRange{Int}}}}`: Window definitions for aggregation
# - `reducefunc::Base.Callable=mean`: Function to apply to each window (default: `mean`)

# # Returns
# - `AbstractArray`: Array with same outer dimensions as `X`, each element containing size-reduced results
function reducesize(
    X::AbstractArray,
    win::Tuple{Vararg{Base.Callable}},
    reducefunc::Base.Callable,
    idx::AbstractVector{Vector{Int}},
    float_type::DataType
)
    Xr = Array{Union{Missing,float_type,Array{float_type}}}(undef, size(X))

    @inbounds for colidx in axes(X, 2)
        for rowidx in axes(X, 1)
            x = X[rowidx, colidx]

            if rowidx in idx[colidx]
                intervals = @evalwindow x win...
                output_dims  = length.(intervals)
                cart_indices = CartesianIndices(output_dims)
                reduced = Array{float_type}(undef, output_dims...)

                for cartidx in cart_indices
                    ranges = get_window_ranges(intervals, cartidx)
                    reduced[cartidx] = safe_feat(reshape(@views(x[ranges...]), :), reducefunc)
                end

                Xr[rowidx, colidx] = reduced
            else
                Xr[rowidx, colidx] = ismissing(x) ? x : float_type(x)
            end
        end
    end

    return Xr
end
