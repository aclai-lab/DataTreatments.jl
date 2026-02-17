module DataTreatments

using Reexport

using Statistics
using StatsBase
using LinearAlgebra
using DataFrames
using Catch22

using Normalization
@reexport using Normalization: AbstractNormalization, fit!, fit, normalize!, normalize

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractFeatureId end
abstract type AbstractDataTreatment end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const NameTypes = Union{Symbol, String}

# ---------------------------------------------------------------------------- #
#                                   files                                      #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
       trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
       forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
       stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq,
       transition_variance, periodicity, base_set, catch9, catch22_set, complete_set
include("featureset.jl")

export movingwindow, wholewindow, splitwindow, adaptivewindow
export @evalwindow
include("windowing.jl")

export is_multidim_dataset, nvals, convert
export has_uniform_element_size
include("treatment.jl")

# using Normalization: HalfZScore, MinMax, halfstd, zscore, center
using Normalization: dimparams, negdims, forward, estimators, normalization
import Normalization: @_Normalization, ZScore, Center
import Normalization: fit!, fit, normalize!, normalize, __mapdims!

using Statistics: mean, median, std
using StatsBase: mad, iqr
using LinearAlgebra: norm

export ZScore, MinMax, Scale, Center, Sigmoid, UnitEnergy, UnitPower, PNorm
include("normalize.jl")

# ---------------------------------------------------------------------------- #
#                                  FeatureId                                   #
# ---------------------------------------------------------------------------- #
"""
    FeatureId <: AbstractFeatureId

A metadata container for individual features in a processed dataset.

This struct stores information about each feature column, including the source variable name,
the transformation function applied, and the window number. It is designed for:

- **Experiment documentation**: All feature extraction parameters are preserved for reproducibility
- **Feature selection**: Metadata enables intelligent feature filtering and selection strategies
- **Traceability**: Each feature can be traced back to its source variable and transformation

# Fields
- `vname::Symbol`: Source variable name from the original dataset
- `feat::Base.Callable`: Feature extraction function (e.g., `mean`, `std`, `maximum`)
- `nwin::Int64`: Window number (1 for single window, >1 for multiple windows)

# Examples
```julia
# Single window feature
fid = FeatureId(:temperature, mean, 1)
# Displays as: mean(temperature)

# Multi-window feature
fid = FeatureId(:pressure, maximum, 3)
# Displays as: maximum(pressure)_w3

# Access metadata
get_vname(fid)    # :pressure
get_feat(fid)  # maximum
get_nwin(fid)     # 3
```

# See Also
- [`DataTreatment`](@ref): Main container using FeatureId for metadata
"""
struct FeatureId <: AbstractFeatureId
    vname :: Symbol
    feat  :: Base.Callable
    nwin  :: Int64

    function FeatureId(vname::NameTypes, feat::Base.Callable, nwin::Int64)
        new(vname, feat, nwin)
    end
end

# value access methods
Base.getproperty(f::FeatureId, s::Symbol) = getfield(f, s)
Base.propertynames(::FeatureId)           = (:vname, :feat, :nwin)

get_vname(f::FeatureId)   = f.vname
get_feat(f::FeatureId) = f.feat
get_nwin(f::FeatureId)    = f.nwin

get_vecvnames(f::Vector{FeatureId})   = [get_vname(n) for n in f]
get_vecfeatures(f::Vector{FeatureId}) = [get_feat(_f) for _f in f]
get_vecnwins(f::Vector{FeatureId})    = [get_nwin(w) for w in f]

function Base.show(io::IO, f::FeatureId)
    feat_name = nameof(f.feat)
    print(io, "$(feat_name)_$(f.vname)_w$(f.nwin)")
end

function Base.show(io::IO, ::MIME"text/plain", f::FeatureId)
    print(io, "FeatureId: ")
    show(io, f)
end

# ---------------------------------------------------------------------------- #
#                               GroupTreatment                                 #
# ---------------------------------------------------------------------------- #
struct GroupResult <: AbstractDataTreatment
    group::Vector{Int64}
    method::Vector{Symbol}

    function GroupResult(
        group::Vector{Int64},
        method::Vector{Symbol}
    )
        new(group, method)
    end
end

get_group(g::GroupResult) = g.group
get_method(g::GroupResult) = g.method

# ---------------------------------------------------------------------------- #
#                                DataTreatment                                 #
# ---------------------------------------------------------------------------- #
"""
    DataTreatment{T, S} <: AbstractDataTreatment

A container for processed multidimensional data with complete metadata for reproducibility.

This struct stores the transformed dataset along with all processing parameters, ensuring
full experiment documentation and reproducibility.

# Fields
- `dataset::AbstractMatrix{T}`: Processed flat feature matrix (samples × features)
- `featureid::Vector{FeatureId}`: Metadata for each feature column (enables feature selection)
- `reducefunc::Base.Callable`: Reduction function used (for `:reducesize` mode)
- `aggrtype::Symbol`: Processing type (`:aggregate` or `:reducesize`)

# Type Parameters
- `T`: Element type of the output dataset
- `S`: Core element type for nested structures

# Constructor
```julia
DataTreatment(
    X::Union{AbstractMatrix, AbstractDataFrame},
    aggrtype::Symbol;
    vnames::Vector{<:NameTypes},
    win::Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    reducefunc::Base.Callable=mean,
    groups::Union{Tuple{Vararg{Symbol}},Nothing}=nothing,
    norm::Union{NormSpec,Type{<:AbstractNormalization},Nothing}=nothing
)
```

# Arguments
- `X`: Input data (Matrix or DataFrame with multidimensional elements)
- `aggrtype`: Processing mode (`:aggregate` or `:reducesize`)
- `vnames`: Variable names for feature identification
- `win`: Window function(s) for data partitioning
- `features`: Tuple of statistical functions to apply (default: `(maximum, minimum, mean)`)
- `reducefunc`: Reduction function for `:reducesize` mode (default: `mean`)

# Processing Modes

## `:aggregate` Mode
Transforms the dataset from multi-dimensional to tabular format.
- The dataset is windowed to reduce its dimensionality
- Reduction functions from the `features` parameter are applied to each window (default: `mean`, `maximum`)

```julia
Xmatrix = fill(rand(200, 120), 100, 10)  # 100 samples, 10 variables
win = splitwindow(nwindows=4)
features = (mean, std, maximum)

dt = DataTreatment(Xmatrix, :aggregate;
                   win, 
                   features)
# Returns 100×(10×3×16) = 100×480 flat matrix
# 10 vars × 3 features × 16 windows (4×4 grid)
```

## `:reducesize` Mode
Reduces the dataset dimensionality by windowing.
- Once windowed, a reduction function called `reducefunc` (default: `mean`) is applied to each window
- Note: it is still possible to specify `features` as in `:aggregate`, but these will simply be saved for future use (as in modal algorithms like ModalDecisionTrees)

```julia
Xmatrix = fill(rand(200, 120), 100, 10)  # 100 samples, 10 variables
win = splitwindow(nwindows=4)
features = (mean, std, maximum)

dt = DataTreatment(Xmatrix, :reducesize;
                   win, 
                   features)
# Each 200×120 element becomes 4×4, resulting in 100×10 output
```

# Grouping

## `groups` Parameter
Optional parameter to group dataset elements before processing.
- Accepts a tuple of symbols specifying grouping columns
- Creates logical groups within the dataset for separate processing
- Common grouping strategies: `(:vname,)`, `(:vname, :feat)`, `(:vname, :timestamp)`

```julia
Xmatrix = [rand(1:100, 4, 2) for _ in 1:10, _ in 1:5]  # 10×5 dataset where each element is a 4×2 matrix
vnames = Symbol.("auto", 1:5)

dt = DataTreatment(Xmatrix, :aggregate;
                   win=splitwindow(nwindows=2),
                   features=(mean, maximum),
                   groups=(:vname, :feat))
# Processes each (vname, feat) group independently
```

# Normalization

## `norm` Parameter
Optional normalization function to apply during processing.
- Accepts a normalization function (e.g., `zscore`, `minmax`, `center`)
- Applied after windowing and feature reduction
- Can be created with keyword arguments for customization

```julia
# Min-max normalization with custom range
dt = DataTreatment(Xmatrix, :aggregate;
                   win=splitwindow(nwindows=2),
                   features=(mean, maximum),
                   norm=MinMax)

# Z-score robust normalization
dt = DataTreatment(Xmatrix, :aggregate;
                   win=splitwindow(nwindows=2),
                   features=(mean, maximum),
                   norm=ZScore(method=:robust))
```

## Grouped Normalization
When using `groups` with `norm`, **never specify `dims` parameter**.
- Grouped normalization works on all elements of each group as a whole
- The `dims` parameter should NOT be used: it operates column-wise or row-wise, which breaks group semantics
- Each group is normalized independently using all its data

```julia
# CORRECT: Grouped normalization without dims
# groups = DataTreatments.groupby(X, [[:var1, :var2]])
# normalized = DataTreatments.normalize(groups, UnitPower)
# Each group normalized using all its elements

# INCORRECT: Do not use dims with grouped normalization
# normalized = DataTreatments.normalize(groups, UnitPower(dims=2))
# This breaks the group semantics!
```

# Accessor Functions
- `get_dataset(dt)`: Extract the processed feature matrix
- `get_featureid(dt)`: Get feature metadata vector
- `get_reducefunc(dt)`: Get the reduction function used
- `get_aggrtype(dt)`: Get the processing mode
- `get_groups(dt)`: Get grouped items
- `get_norm(dt)`: Get normalization technique
- `get_vnames(dt)`: Get unique variable names
- `get_features(dt)`: Get unique feature functions
- `get_nwindows(dt)`: Get maximum window number

# See Also
- [`FeatureId`](@ref): Individual feature metadata
- [`@evalwindow`](@ref): Window evaluation macro
"""
struct DataTreatment{T,S} <: AbstractDataTreatment
    dataset::AbstractMatrix{T}
    featureid::Vector{FeatureId}
    reducefunc::Base.Callable
    aggrtype::Symbol
    groups::Union{Vector{GroupResult},Nothing}
    norm::Union{NormSpec,Nothing}

    function DataTreatment(
        X::AbstractArray{<:AbstractArray{T}},
        aggrtype::Symbol;
        vnames::Union{Vector{<:NameTypes},Nothing}=nothing,
        win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}},
        features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
        reducefunc::Base.Callable=mean,
        groups::Union{Tuple{Vararg{Symbol}},Nothing}=nothing,
        norm::Union{NormSpec,Type{<:AbstractNormalization},Nothing}=nothing
    ) where {T<:Real}
        # checks the special case of a dataset whose elements have different sizes
        # verifies that the chosen window is either adaptivewindow (recommended) or wholewindow,
        # and otherwise throws an error
        # sets the variable `uniform`, which will be passed to aggregate or reducesize
        # to indicate whether the window must be computed for each individual element
        # because if all elements share the same size,
        # computing the window each time is a significant overhead
        uniform = has_uniform_element_size(X)

        # convert to float
        T isa AbstractFloat || (X = convert(X))

        isnothing(vnames) && (vnames = [Symbol("V$i") for i in 1:size(X, 2)])
        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        win isa Base.Callable && (win = (win,))
        intervals = @evalwindow first(X) win...
        nwindows = prod(length.(intervals))

        Xresult, Xinfo = if aggrtype == :aggregate begin
            (aggregate(X, intervals; features, win, uniform),
            if nwindows == 1
                # single window: apply to whole time series
                [FeatureId(v, f, 1) for f in features, v in vnames] |> vec
            else
                # multiple windows: apply to each interval
                [FeatureId(v, f, i) for i in 1:nwindows, f in features, v in vnames] |> vec
            end
            )
        end

        elseif aggrtype == :reducesize begin
            (reducesize(X, intervals; reducefunc, win, uniform),
            [FeatureId(v, reducefunc, 1) for v in vnames]
            )
        end

        else
            error("Unknown treatment type: $treat")
        end

        grp_result = if !isnothing(groups)
            fields = collect(groups)
            groupidxs, _ = _groupby(Xresult, Xinfo, fields)
            [GroupResult(groupidx, fields) for groupidx in groupidxs]
        else
            nothing
        end

        if !isnothing(norm)
            norm isa Type{<:AbstractNormalization} && (norm = norm())
            if isnothing(grp_result)
                Xresult = normalize(Xresult, norm)
            else
                Threads.@threads for g in grp_result
                    Xresult[:, get_group(g)] =
                        normalize(Xresult[:, get_group(g)], norm)
                end
            end
        end

        new{eltype(Xresult), core_eltype(Xresult)}(
            Xresult, Xinfo, reducefunc, aggrtype, grp_result, norm
        )
    end

    function DataTreatment(
        X      :: AbstractDataFrame,
        args...;
        vnames :: Union{Vector{<:NameTypes}, Nothing}=nothing,
        kwargs...
    )
        isnothing(vnames) && (vnames = propertynames(X))
        DataTreatment(Matrix{typeof(X[1,1])}(X), args...; vnames, kwargs...)
    end
end

# value access methods
Base.getproperty(dt::DataTreatment, s::Symbol) = getfield(dt, s)
Base.propertynames(::DataTreatment) =
    (:dataset, :featureid, :reducefunc, :aggrtype, :groups, :norm)

get_dataset(dt::DataTreatment)    = dt.dataset
get_featureid(dt::DataTreatment)  = dt.featureid
get_reducefunc(dt::DataTreatment) = dt.reducefunc
get_aggrtype(dt::DataTreatment)   = dt.aggrtype
get_groups(dt::DataTreatment)       = dt.groups
get_norm(dt::DataTreatment)       = dt.norm

# Convenience methods for common operations
get_vnames(dt::DataTreatment)   = unique(get_vecvnames(dt.featureid))
get_features(dt::DataTreatment) = unique(get_vecfeatures(dt.featureid))
get_nwindows(dt::DataTreatment) = maximum(get_vecnwins(dt.featureid))

# Size and iteration methods
Base.size(dt::DataTreatment)   = size(dt.dataset)
Base.size(dt::DataTreatment, dim::Int) = size(dt.dataset, dim)
Base.length(dt::DataTreatment) = length(dt.featureid)
Base.eltype(dt::DataTreatment) = eltype(dt.dataset)

# Indexing methods
Base.getindex(dt::DataTreatment, i::Int) = dt.dataset[:, i]
Base.getindex(dt::DataTreatment, i::Int, j::Int) = dt.dataset[i, j]
Base.getindex(dt::DataTreatment, ::Colon, j::Int) = dt.dataset[:, j]
Base.getindex(dt::DataTreatment, i::Int, ::Colon) = dt.dataset[i, :]
Base.getindex(dt::DataTreatment, I...) = dt.dataset[I...]

export FeatureId, DataTreatment
export get_vname, get_feat, get_nwin
export get_vecvnames, get_vecfeatures, get_vecnwins
export get_vnames, get_features, get_nwindows
export get_dataset, get_featureid, get_reducefunc, get_aggrtype
export get_groups, get_norm, get_normdims

function Base.show(io::IO, dt::DataTreatment)
    nrows, ncols = size(dt.dataset)
    print(io, "DataTreatment($(dt.aggrtype), $(nrows)×$(ncols), $(length(dt.featureid)) features)")
end

function Base.show(io::IO, ::MIME"text/plain", dt::DataTreatment)
    nrows, ncols = size(dt.dataset)
    nfeatures = length(dt.featureid)
    
    println(io, "DataTreatment:")
    println(io, "  Type: $(dt.aggrtype)")
    println(io, "  Dimensions: $(nrows)×$(ncols)")
    println(io, "  Features: $(nfeatures)")
    println(io, "  Reduction function: $(nameof(dt.reducefunc))")
    isnothing(dt.norm) ?
        println(io, "  Normalization: none") :
        println(io, "  Normalization: $(dt.norm.type)")
    
    if nfeatures <= 10
        println(io, "  Feature IDs:")
        for (i, fid) in enumerate(dt.featureid)
            println(io, "    $(i). ", fid)
        end
    else
        println(io, "  Feature IDs (showing first 5 and last 5):")
        for i in 1:5
            println(io, "    $(i). ", dt.featureid[i])
        end
        println(io, "    ⋮")
        for i in (nfeatures-4):nfeatures
            println(io, "    $(i). ", dt.featureid[i])
        end
    end
end

export groupby
include("groupby.jl")

end
