# this module transforms multidimensional datasets
# into formats suitable for different model algorithm families:

# 1. propositional algorithms (DecisionTree, XGBoost):
#    - applies windowing to divide time series into segments
#    - extracts scalar features (max, min, mean, etc.) from each window
#    - returns a standard tabular DataFrame

# 2. modal algorithms (ModalDecisionTree):
#    - creates windowed time series preserving temporal structure
#    - applies reduction functions to manage dimensionality

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# base type for metadata containers
abstract type AbstractDataTreatment end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const ValidVnames = Union{Symbol, String}

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# recursively extract the core element type from nested array types.
core_eltype(x) = eltype(x) <: AbstractArray ? core_eltype(eltype(x)) : eltype(x)



# ---------------------------------------------------------------------------- #
#                             applyfeat functions                              #
# ---------------------------------------------------------------------------- #
"""
    applyfeat(X::AbstractArray, intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}[; reducefunc::Base.Callable=mean]) -> AbstractArray

Apply a reduction function to windows defined by intervals over an array.

This function performs dimensionality reduction on an n-dimensional array by dividing it into 
windows and applying an aggregation function to each window.

A reduction function (also called aggregation function) is a function that takes 
multiple values and returns a single summary value, such as `mean`, `maximum`, `minimum`, 
`std`, `median`, or `mode`.

# Arguments
- `X::AbstractArray`: Input array to process
- `intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}`: Tuple of vectors, each containing ranges 
  defining windows along each dimension of `X`
- `reducefunc::Base.Callable=mean`: Function to apply to each window (default: `mean`)

# Returns
- `AbstractArray`: Reduced array with dimensions matching the number of windows in each dimension

# Example
```julia
X = rand(100, 120)
wfunc = splitwindow(nwindows=10)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals; reducefunc=maximum)
# Returns a 2×3 matrix containing maximum values for each window
```
"""
function applyfeat(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)::AbstractArray
    reduced = similar(X, length.(intervals)...)

    @inbounds map!(reduced, CartesianIndices(reduced)) do cart_idx
        ranges = ntuple(i -> intervals[i][cart_idx[i]], length(intervals))
        reducefunc(@view X[ranges...])
    end
end

# ---------------------------------------------------------------------------- #
#                             aggregate functions                              #
# ---------------------------------------------------------------------------- #
"""
    aggregate(X::AbstractArray, intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}[; reducefunc::Base.Callable=mean]) -> AbstractArray

Apply window-based aggregation to each element of an array.

This function is designed for arrays where each element is itself an array (e.g., `Matrix{Matrix{Float64}}`).
It applies `applyfeat` to each element.

# Arguments
- `X::AbstractArray`: Input array where each element is an array to be aggregated
- `intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}`: Window definitions for aggregation
- `reducefunc::Base.Callable=mean`: Function to apply to each window (default: `mean`)

# Returns
- `AbstractArray`: Array with same outer dimensions as `X`, each element containing aggregated results

# Example
```julia
X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = aggregate(Xmatrix, intervals; reducefunc=std)
```
"""
function aggregate(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)::AbstractArray
    Xresult = similar(X)
    Threads.@threads for i in eachindex(X)
        @inbounds Xresult[i] = applyfeat(X[i], intervals; reducefunc)
    end
    return Xresult
end

# ---------------------------------------------------------------------------- #
#                            reducesize functions                              #
# ---------------------------------------------------------------------------- #
"""
    reducesize(X::AbstractArray, intervals::Tuple{Vararg{Vector{UnitRange{Int64}}}}[; features::Tuple{Vararg{Base.Callable}}=(mean,)]) -> AbstractArray

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
with results concatenated in the order: `[col1_feat1_windows..., col1_feat2_windows..., col2_feat1_windows..., ...]`

# Example
```julia
X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
features = (mean, maximum)
result = reducesize(Xmatrix, intervals; features)
```
"""
function reducesize(
    X         :: AbstractArray,
    intervals :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    features  :: Tuple{Vararg{Base.Callable}}=(mean,),
)::AbstractArray
    nwindows = prod(length.(intervals))
    nfeats = nwindows * length(features)
    Xresult = Array{core_eltype(X)}(undef, size(X, 1), size(X, 2) * nfeats)
    
    @inbounds Threads.@threads for colidx in axes(X, 2)
        for rowidx in axes(X,1)
            reduced = mapreduce(vcat, features) do feat
                vec(applyfeat(X[rowidx,colidx], intervals; reducefunc=feat))
            end
            
            base_idx = (colidx - 1) * nfeats
            @inbounds copyto!(view(Xresult, rowidx, base_idx+1:base_idx+nfeats), vec(reduced))
        end
    end
    return Xresult
end

# ---------------------------------------------------------------------------- #
#                                 constructor                                  #
# ---------------------------------------------------------------------------- #
struct DataTreatment <: AbstractDataTreatment
    dataset    :: AbstractMatrix
    vnames     :: Vector{Symbol}
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}}
    features   :: Tuple{Vararg{Base.Callable}}
    reducefunc :: Base.Callable
    aggrtype   :: Symbol

    function DataTreatment(
        X           :: AbstractMatrix,
        aggrtype    :: Symbol;
        vnames      :: Vector{ValidVnames},
        win         :: Tuple{Vararg{Base.Callable}},
        features    :: Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
        modalreduce :: Base.Callable=mean
    )
        is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
            "does not contain multidimensional data."))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))

        intervals = @evalwindow X win...

        # propositional models
        isempty(features) && (treat = :none)

        colnames = if treat == :aggregate
            for f in features, v in vnames
                if length(intervals) == 1
                    # single window: apply to whole time series
                    Symbol("$(f)($(v))")
                else
                    # multiple windows: apply to each interval
                    for (i, interval) in enumerate(intervals)
                        col_name = Symbol("$(f)($(v))w$(i)")
                        apply_vectorized!(_X, X[!, v], f, col_name, interval)
                    end
                end
            end

        # modal models
        elseif treat == :reducesize
            for v in vnames
                apply_vectorized!(_X, X[!, v], modalreduce, v, intervals)
            end
            
        elseif treat == :none
            _X = X

        else
            error("Unknown treatment type: $treat")
        end

        return _X, TreatmentInfo(features, win, treat, modalreduce)
    end
end

function DataTreatment(
    X      :: AbstractDataFrame,
    args...;
    vnames :: Union{Vector{ValidVnames}, Nothing},
    kwargs...
)
    isnothing(vnames) && (vnames = propertynames(X))
    DataTreatment()
end