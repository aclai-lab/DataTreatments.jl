# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# recursively extract the core element type from nested array types.
core_eltype(x) = eltype(x) <: AbstractArray ? core_eltype(eltype(x)) : eltype(x)

# extract window ranges from intervals and cartesian index
@inline function get_window_ranges(intervals::Tuple, cart_idx::CartesianIndex)
    ntuple(i -> intervals[i][cart_idx[i]], length(intervals))
end

"""
    is_multidim_dataset(X::Union{AbstractArray, AbstractDataFrame}) -> Bool

Return true if any feature column contains array-valued elements, indicating a
multidimensional dataset that requires aggregation or size-reduction.

Arguments
- X: A matrix-like object or a DataFrame. Each column is inspected; if
  `eltype(col) <: AbstractArray` for any column, the function returns true.

Examples
```julia
# DataFrame with array-valued column:
df = DataFrame(a=[rand(10) for _ in 1:3], b=rand(3))
is_multidim_dataset(df)  # true

# Plain numeric matrix:
X = rand(5, 3)
is_multidim_dataset(X)   # false

# Matrix of arrays:
X = [rand(4) for _ in 1:5, _ in 1:2]
is_multidim_dataset(X)   # true
```

Notes
- Used to select the preprocessing path (aggregate vs reducesize).
"""
@inline is_multidim_dataset(X::Union{AbstractArray, AbstractDataFrame})::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

"""
    nvals(intervals) -> Int

Return the total number of window combinations.
Examples:
- For a single vector of ranges: nvals(v) == length(v)
- For a tuple of vectors: nvals((v1, v2, ...)) == prod(length.(...))

```julia
intervals = (UnitRange{Int}[1:50, 51:100, 101:150, 151:200],
            UnitRange{Int}[1:30, 31:60, 61:90, 91:120])
nvals(intervals) # 16
```
"""
@inline nvals(intervals::Tuple{Vararg{<:AbstractVector}})::Int64 = prod(length.(intervals))
@inline nvals(intervals::AbstractVector)::Int64 = length(intervals)

"""
    convert(X::AbstractArray{<:AbstractArray{T}}; type::Type=Float64) where {T<:Real}

Convert all nested arrays in `X` to a specified numeric type in parallel.

This function takes an array of arrays (e.g., `Matrix{Matrix{Float64}}`) and converts
each inner array to the specified type using multithreaded processing for efficiency.

# Arguments
- `X::AbstractArray{<:AbstractArray{T}}`: Input array where each element is an array of real numbers
- `type::Type=Float64`: Target numeric type for conversion (e.g., `Float32`, `Float64`, `Int32`)

# Returns
- `AbstractArray{AbstractArray{type}}`: New array with the same structure as `X`, but with all
  inner arrays converted to the specified type

# Examples
```julia
# Convert matrix of Float64 matrices to Float32
X = [rand(Float64, 100, 100) for _ in 1:500, _ in 1:100]
X_f32 = convert(X; type=Float32)

# Convert to Int32
X_int = convert(X; type=Int32)

# Default conversion to Float64
X_f64 = convert(X)
```
"""
function convert(X::AbstractArray{<:AbstractArray{T}}; type::Type=Float64) where {T<:Real}
    Xconv = similar(X, AbstractArray{type})
    Threads.@threads for i in eachindex(X)
        Xconv[i] = type.(X[i])
    end
    return Xconv
end

"""
    has_uniform_element_size(X::AbstractDataFrame) -> Bool
    has_uniform_element_size(X::AbstractArray) -> Bool

Return `true` if every element in the Array or DataFrame has the same size (shape), `false` otherwise.
This is useful for deciding whether window definitions can be reused across all entries.

- Empty Array or DataFrames return `true`.
- Short-circuits on the first mismatch.
- Inspects all columns and all elements.

# Examples
```julia
df = DataFrame(a=[rand(3,4) for _ in 1:2], b=[rand(3,4) for _ in 1:2])
has_uniform_element_size(df)  # true

df = DataFrame(a=[rand(3,4), rand(2,4)], b=[rand(3,4), rand(3,4)])
has_uniform_element_size(df)  # false
```
"""
@inline function has_uniform_element_size(X::AbstractDataFrame)
    isempty(X) && return true
    refsize = size(X[1,1]) # size of first element in first column
    @inbounds for col in eachcol(X)
        for elem in col
            size(elem) == refsize || return false
        end
    end
    return true
end

@inline function has_uniform_element_size(X::AbstractArray)
    isempty(X) && return true
    refsize = size(first(X))
    @inbounds for x in X
        size(x) == refsize || return false
    end
    return true
end

# ---------------------------------------------------------------------------- #
#                             reducesize functions                              #
# ---------------------------------------------------------------------------- #
# Apply window-based size-reduction to each element of an array.

# This function is designed for arrays where each element is itself an array (e.g., `Matrix{Matrix{Float64}}`).

# # Arguments
# - `X::AbstractArray`: Input array where each element is an array to be size-reduced
# - `intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}`: Window definitions for aggregation
# - `reducefunc::Base.Callable=mean`: Function to apply to each window (default: `mean`)

# # Returns
# - `AbstractArray`: Array with same outer dimensions as `X`, each element containing size-reduced results
function reducesize(
    X          :: AbstractArray{T},
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean,
    win        :: Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
    uniform    :: Bool
) where {T<:AbstractArray{<:Real}}
    output_dims  = length.(intervals)
    Xresult      = similar(X)
    cart_indices = CartesianIndices(output_dims)
    
    @inbounds for i in 1:length(X)
        element = X[i]
        uniform || begin
            intervals = @evalwindow X[i] win...
            output_dims  = length.(intervals)
        end
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
# - `intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}`: Window definitions for aggregation
# - `features::Tuple{Vararg{Base.Callable}}=(mean,)`: Tuple of functions to compute on each window

# # Returns
# - `AbstractArray`: Flattened array with dimensions `(nrows(X), ncols(X) × n_features × n_windows)`
#   where each element is of type `core_eltype(X)`

# # Details
# The output columns are organized as: for each column in `X`, all features are computed for all windows,
# with results concatenated in the order: `[col1_feat1_win1, col1_feat1_win2, col1_feat2_win1, ...]`
function aggregate(
    X         :: AbstractArray{T},
    intervals :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    features  :: Tuple{Vararg{Base.Callable}}=(mean,),
    win       :: Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
    uniform   :: Bool
) where {T<:AbstractArray{<:Real}}
    nwindows = prod(length.(intervals))
    nfeats   = nwindows * length(features)
    Xresult  = Array{core_eltype(X)}(undef, size(X, 1), size(X, 2) * nfeats)
    
    for colidx in axes(X, 2)
        for rowidx in axes(X, 1)
            out_idx = (colidx - 1) * nfeats + 1
            uniform || (intervals = @evalwindow X[rowidx, colidx] win...)
            
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

