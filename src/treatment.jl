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
    # warning
    # indica riga e colonna
    f(collect(x for x in skipmissing(v) if !(x isa AbstractFloat && isnan(x))))
end

# ---------------------------------------------------------------------------- #
#                             aggregate functions                              #
# ---------------------------------------------------------------------------- #
"""
    aggregate(
        X::AbstractArray,
        win::Tuple{Vararg{Base.Callable}},
        features::Tuple{Vararg{Base.Callable}},
        idx::AbstractVector{Vector{Int}},
        float_type::Type
    ) -> (Xa, nwindows)

Transform a multivariate dataset into a tabular dataset through windowing and feature aggregation.

Most machine learning algorithms work exclusively with tabular data, so transforming multivariate 
datasets is necessary. This internal function accomplishes this transformation by:

1. Separating multivariate data into windows, preserving portions of the original data rather 
   than completely flattening it (each window corresponds to a column)
2. Applying aggregation functions (e.g., mean, maximum, minimum) to the windowed data to convert 
   it into scalar values
3. Creating a tabular dataset where, for each aggregation function applied, there is a designated 
   number of window columns

# Arguments
- `X::AbstractArray`: The multivariate dataset to aggregate
- `win::Tuple{Vararg{Base.Callable}}`: Window definition functions for each dimension
- `features::Tuple{Vararg{Base.Callable}}`: Aggregation functions to apply (e.g., mean, maximum)
- `idx::AbstractVector{Vector{Int}}`: Valid indices for each column (non-missing, non-NaN elements)
- `float_type::Type`: The output floating-point type

# Returns
- `Xa::Matrix`: A tabular matrix where each column represents an aggregated window feature
- `nwindows::Vector`: Number of windows per column
"""
function aggregate(
    X::AbstractArray,
    idx::AbstractVector{Vector{Int}},
    float_type::Type;
    win::Tuple{Vararg{Base.Callable}},
    features::Tuple{Vararg{Base.Callable}},
)
    colwin = [[n > length(win) ? last(win) : win[n] for n in 1:ndims(X[first(idx[i]), i])] for i in axes(X, 2)]
    nwindows = [prod(hasfield(typeof(w), :nwindows) ? w.nwindows : 1 for w in c) for c in colwin]
    nfeats = length(features)
# vettore di indici
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

aggregate(; kwargs...) = (x, i, ft) -> aggregate(x, i, ft; kwargs...)

# ---------------------------------------------------------------------------- #
#                             reducesize functions                              #
# ---------------------------------------------------------------------------- #
"""
    reducesize(
        X::AbstractArray,
        win::Tuple{Vararg{Base.Callable}},
        reducefunc::Base.Callable,
        idx::AbstractVector{Vector{Int}},
        float_type::DataType
    ) -> Xr

Reduce the size of a multivariate dataset through windowing and dimension reduction.

This internal function is designed to reduce the size of large multivariate datasets (e.g., images, 
audio files) that may have excessive dimensionality or disproportionate resolution levels, which 
would hinder effective machine learning. The process consists of:

1. Applying windowing to subdivide the multivariate data
2. Reducing each window's dimensionality by applying a reduction function (typically mean)
3. Returning a dataset of the same type but significantly reduced in size

The resulting reduced dataset is suitable for use with modal algorithms available in the Sole 
framework of Aclai.

# Arguments
- `X::AbstractArray`: The multivariate dataset to reduce
- `win::Tuple{Vararg{Base.Callable}}`: Window definition functions for each dimension
- `reducefunc::Base.Callable`: Reduction function to apply (e.g., mean, median)
- `idx::AbstractVector{Vector{Int}}`: Valid indices for each column (non-missing, non-NaN elements)
- `float_type::DataType`: The output floating-point type

# Returns
- `Xr::Array`: A reduced-size dataset where each element contains the result of applying 
  the reduction function to its corresponding window
"""
function reducesize(
    X::AbstractArray,
    idx::AbstractVector{Vector{Int}},
    float_type::DataType;
    win::Tuple{Vararg{Base.Callable}},
    reducefunc::Base.Callable,
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

reducesize(; kwargs...) = (x, i, ft) -> reducesize(x, i, ft; kwargs...)
