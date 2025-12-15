# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# recursively extract the core element type from nested array types.
core_eltype(x) = eltype(x) <: AbstractArray ? core_eltype(eltype(x)) : eltype(x)

# extract window ranges from intervals and cartesian index
@inline function get_window_ranges(intervals::Tuple, cart_idx::CartesianIndex)
    ntuple(i -> intervals[i][cart_idx[i]], length(intervals))
end

@inline is_multidim_dataset(X::Union{AbstractArray, AbstractDataFrame})::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

# ---------------------------------------------------------------------------- #
#                             reducesize functions                              #
# ---------------------------------------------------------------------------- #
"""
    reducesize(X::AbstractArray, intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}[; reducefunc::Base.Callable=mean]) -> AbstractArray

Apply window-based size-reduction to each element of an array.

This function is designed for arrays where each element is itself an array (e.g., `Matrix{Matrix{Float64}}`).

# Arguments
- `X::AbstractArray`: Input array where each element is an array to be size-reduced
- `intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}`: Window definitions for aggregation
- `reducefunc::Base.Callable=mean`: Function to apply to each window (default: `mean`)

# Returns
- `AbstractArray`: Array with same outer dimensions as `X`, each element containing size-reduced results

# Example
```julia
X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = reducesize(Xmatrix, intervals; reducefunc=std)
```
"""
function reducesize(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)::AbstractArray
    output_dims = length.(intervals)
    Xresult = similar(X)
    cart_indices = CartesianIndices(output_dims)
    
    @inbounds Threads.@threads for i in 1:length(X)
        element = X[i]
        reduced = similar(element, output_dims...)
        
        for cart_idx in cart_indices
            ranges = get_window_ranges(intervals, cart_idx)
            reduced[cart_idx] = reducefunc(reshape(@views(element[ranges...]), :))
        end
        
        Xresult[i] = reduced
    end

    return Xresult
end

# ---------------------------------------------------------------------------- #
#                            aggregate functions                              #
# ---------------------------------------------------------------------------- #
"""
    aggregate(X::AbstractArray, intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}[; features::Tuple{Vararg{Base.Callable}}=(mean,)]) -> AbstractArray

Flatten nested arrays by applying multiple feature functions to windowed regions.

This function takes a matrix of arrays (e.g., `Matrix{Matrix{Float64}}`) and produces a single
flat matrix where each row corresponds to a row in `X`, and columns contain flattened features
computed from windowed aggregations.

# Use Case
This function enables the analysis of datasets containing n-dimensional elements (such as images, 
time series, or spectrograms) with machine learning models that require tabular input. It transforms 
complex multi-dimensional data into a flat feature matrix suitable for standard ML algorithms.

# Arguments
- `X::AbstractArray`: Input array where each element is an array (e.g., `Matrix{Matrix{Float64}}`)
- `intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}`: Window definitions for aggregation
- `features::Tuple{Vararg{Base.Callable}}=(mean,)`: Tuple of functions to compute on each window

# Returns
- `AbstractArray`: Flattened array with dimensions `(nrows(X), ncols(X) × n_features × n_windows)`
  where each element is of type `core_eltype(X)`

# Details
The output columns are organized as: for each column in `X`, all features are computed for all windows,
with results concatenated in the order: `[col1_feat1_win1, col1_feat1_win2, col1_feat2_win1, ...]`

# Example
```julia
X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
features = (mean, maximum)
result = aggregate(Xmatrix, intervals; features)
```
"""
function aggregate(
    X         :: AbstractArray,
    intervals :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    features  :: Tuple{Vararg{Base.Callable}}=(mean,)
)::AbstractArray
    nwindows = prod(length.(intervals))
    nfeats   = nwindows * length(features)
    Xresult  = Array{core_eltype(X)}(undef, size(X, 1), size(X, 2) * nfeats)
    
    @inbounds Threads.@threads for colidx in axes(X, 2)
        for rowidx in axes(X, 1)
            out_idx = (colidx - 1) * nfeats + 1
            
            for feat in features
                for cart_idx in CartesianIndices(length.(intervals))
                    ranges = get_window_ranges(intervals, cart_idx)
                    window_view = @views X[rowidx, colidx][ranges...]
                    Xresult[rowidx, out_idx] = feat(reshape(window_view, :))
                    out_idx += 1
                end
            end
        end
    end
    
    return Xresult
end

