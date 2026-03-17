# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# extract window ranges from intervals and cartesian index
@inline function get_window_ranges(intervals::Tuple, cartidx::CartesianIndex)
    ntuple(i -> intervals[i][cartidx[i]], length(intervals))
end

"""
    safe_feat(v, f) -> Any

Apply a feature function to a vector while safely handling missing values and NaN entries.

Filters out `missing` values and `NaN` entries before applying the feature function,
making it robust for incomplete or noisy data. Used in aggregation and size-reduction
operations on windowed data.

# Arguments
- `v::AbstractVector`: Input vector that may contain missing values or NaN entries.
- `f::Callable`: Feature function to apply (e.g., `mean`, `std`, `maximum`).

# Returns
- Result of applying `f` to the cleaned, materialized vector (typically a scalar).

# Details
- Skips all `missing` values via `skipmissing()`.
- Filters out `NaN` values (for floating-point types).
- The cleaned values are collected into a `Vector` before calling `f`, since some
  aggregation functions (e.g., `mean`, `std`) require indexable collections.
- If all values are missing/NaN, `f` receives an empty `Vector`.
"""
@inline function safe_feat(v, f)
    f(collect(x for x in skipmissing(v) if !(x isa AbstractFloat && isnan(x))))
end

# ---------------------------------------------------------------------------- #
#                             aggregate functions                              #
# ---------------------------------------------------------------------------- #
"""
    aggregate(X, idx, float_type; win, features) -> (Xa, nwindows)

Transform a multivariate dataset into a tabular dataset through windowing and
feature aggregation.

# Arguments
- `X::AbstractArray`: The multivariate dataset to aggregate.
- `idx::AbstractVector{Vector{Int}}`: Valid row indices for each column
  (non-missing, non-NaN elements).
- `float_type::Type`: The output floating-point type.

# Keyword Arguments
- `win::Tuple{Vararg{Base.Callable}}`: Window definition functions for each
  dimension. When fewer functions than dimensions are provided, the last
  function is reused for remaining dimensions.
- `features::Tuple{Vararg{Base.Callable}}`: Aggregation functions to apply
  (e.g., `mean`, `maximum`, `minimum`).

# Returns
- `Xa::Matrix{Union{Missing,float_type}}`: Tabular matrix where each column
  represents one (feature × window × original column) combination.
- `nwindows::Vector{Int}`: Number of windows generated per original column.

# Details
- For valid rows (`rowidx ∈ idx[colidx]`), windows are computed via
  [`@evalwindow`](@ref) and each `feature` is applied to each window slice
  through [`safe_feat`](@ref).
- For invalid rows (missing or NaN source elements), the original value is
  broadcast across all output slots for that column.
"""
function aggregate(
    X::AbstractArray,
    idx::AbstractVector{Vector{Int}},
    float_type::Type;
    win::Tuple{Vararg{Base.Callable}},
    features::Tuple{Vararg{Base.Callable}}
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

"""
    aggregate(; win, features) -> Function

Curried form that returns a closure `(X, idx, float_type) -> aggregate(X, idx, float_type; win, features)`.

# Keyword Arguments
- `win::Tuple{Vararg{Base.Callable}}`: Window functions. Default: `(wholewindow(),)`.
- `features::Tuple{Vararg{Base.Callable}}`: Feature functions. Default: `(maximum, minimum, mean)`.

# Returns
A `Function` suitable for use as the `aggrfunc` parameter of [`TreatmentGroup`](@ref).
"""
aggregate(;
    win::Tuple{Vararg{Base.Callable}}=(wholewindow(),),
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean)
) = (x, i, ft) -> aggregate(x, i, ft; win, features)

# ---------------------------------------------------------------------------- #
#                             reducesize functions                             #
# ---------------------------------------------------------------------------- #
"""
    reducesize(X, idx, float_type; win, reducefunc) -> (Xr, 0)

Reduce the size of a multivariate dataset through windowing and dimension reduction.

# Arguments
- `X::AbstractArray`: The multivariate dataset to reduce.
- `idx::AbstractVector{Vector{Int}}`: Valid row indices for each column
  (non-missing, non-NaN elements).
- `float_type::DataType`: The output floating-point type.

# Keyword Arguments
- `win::Tuple{Vararg{Base.Callable}}`: Window definition functions for each
  dimension.
- `reducefunc::Base.Callable`: Reduction function applied to each window
  (e.g., `mean`, `median`).

# Returns
- `Xr::Array{Union{Missing,float_type,Array{float_type}}}`: Reduced dataset
  where each valid element is an array with one value per window.
- `0::Int`: Placeholder (unused) to match the return signature of
  [`aggregate`](@ref).

# Details
- For valid rows (`rowidx ∈ idx[colidx]`), windows are computed via
  [`@evalwindow`](@ref) and `reducefunc` is applied to each window slice
  through [`safe_feat`](@ref).
- For invalid rows, the original value is preserved (missing stays missing,
  NaN is cast to `float_type`).
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

    return Xr, 0 # not used
end

"""
    reducesize(; win, reducefunc) -> Function

Curried form that returns a closure `(X, idx, float_type) -> reducesize(X, idx, float_type; win, reducefunc)`.

# Keyword Arguments
- `win::Tuple{Vararg{Base.Callable}}`: Window functions. Default: `(wholewindow(),)`.
- `reducefunc::Base.Callable`: Reduction function. Default: `mean`.

# Returns
A `Function` suitable for use as the `aggrfunc` parameter of [`TreatmentGroup`](@ref).
"""
reducesize(;
    win::Tuple{Vararg{Base.Callable}}=(wholewindow(),),
    reducefunc::Base.Callable=mean
) = (x, i, ft) -> reducesize(x, i, ft; win, reducefunc)
