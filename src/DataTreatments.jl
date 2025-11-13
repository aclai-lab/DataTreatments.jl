module DataTreatments

using Statistics
using DataFrames
using Catch22

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractFeatureId end
abstract type AbstractDataTreatment end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const ValidVnames = Union{Symbol, String}

export col, row
@enum NormDim col row

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

export reducesize, aggregate
include("treatment.jl")

export zscore, sigmoid, rescale, center, unitenergy, unitpower, halfzscore, outliersuppress, minmaxclip
export element_norm, tabular_norm, ds_norm
include("normalize.jl")

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
is_multidim_dataframe(X::AbstractArray)::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

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
get_feature(fid)  # maximum
get_nwin(fid)     # 3
```

# See Also
- [`DataTreatment`](@ref): Main container using FeatureId for metadata
"""
struct FeatureId <: AbstractFeatureId
    vname :: Symbol
    feat  :: Base.Callable
    nwin  :: Int64

    function FeatureId(vname::ValidVnames, feat::Base.Callable, nwin::Int64)
        new(vname, feat, nwin)
    end
end

# value access methods
Base.getproperty(f::FeatureId, s::Symbol) = getfield(f, s)
Base.propertynames(::FeatureId)           = (:vname, :feat, :nwin)

get_vname(f::FeatureId)   = f.vname
get_feature(f::FeatureId) = f.feat
get_nwin(f::FeatureId)    = f.nwin

function Base.show(io::IO, f::FeatureId)
    feat_name = nameof(f.feat)
    if f.nwin == 1
        print(io, "$(feat_name)($(f.vname))")
    else
        print(io, "$(feat_name)($(f.vname))_w$(f.nwin)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", f::FeatureId)
    print(io, "FeatureId: ")
    show(io, f)
end

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
    vnames::Vector{<:ValidVnames},
    win::Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    reducefunc::Base.Callable=mean,
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

## `:reducesize` Mode
Applies multiple feature functions to windowed regions, preserving the dataset structure.
Each element is reduced but the matrix dimensions are maintained.

```julia
Xmatrix = fill(rand(200, 120), 100, 10)  # 100 samples, 10 variables
win = splitwindow(nwindows=4)
features = (mean, std, maximum)

dt = DataTreatment(Xmatrix, :reducesize; 
                   vnames=Symbol.("var", 1:10),
                   win=(win,), 
                   features=features)
# Each 200×120 element becomes 4×4, resulting in 100×10 output
```

## `:aggregate` Mode
Flattens multidimensional data into a single feature matrix suitable for ML models.
Applies multiple features across windows and concatenates results.

```julia
dt = DataTreatment(Xmatrix, :aggregate;
                   vnames=Symbol.("var", 1:10),
                   win=(win,),
                   features=features)
# Returns 100×(10×3×16) = 100×480 flat matrix
# 10 vars × 3 features × 16 windows (4×4 grid)
```

# Examples

## Basic Usage with DataFrame
```julia
using DataFrames

# Create dataset with multidimensional elements
df = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:1000],
    channel2 = [rand(200, 120) for _ in 1:1000],
    channel3 = [rand(200, 120) for _ in 1:1000]
)

# Define processing parameters
win = adaptivewindow(nwindows=6, overlap=0.15)
features = (mean, std, maximum, minimum, median)

# Process to tabular format
dt = DataTreatment(df, :reducesize; 
                   win=(win,), 
                   features=features)

# Access processed data
X_flat = get_dataset(dt)        # Flat feature matrix
feature_ids = get_featureid(dt) # Feature metadata
```

## Feature Selection Using Metadata
```julia
# Get all feature metadata
feature_ids = get_featureid(dt)

# Select specific features
mean_features = findall(fid -> get_feature(fid) == mean, feature_ids)
X_means = dt.dataset[:, mean_features]

# Select features from specific variable
ch1_features = findall(fid -> get_vname(fid) == :channel1, feature_ids)
X_ch1 = dt.dataset[:, ch1_features]

# Select features from specific windows
early_windows = findall(fid -> get_nwin(fid) <= 3, feature_ids)
X_early = dt.dataset[:, early_windows]
```

## Reproducibility and Documentation
```julia
# All parameters are stored for experiment reproduction
dt = DataTreatment(df, :reducesize; win=(win,), features=features)

# Extract processing metadata
aggrtype = get_aggrtype(dt)       # :reducesize
reduction = get_reducefunc(dt)    # mean
var_names = get_vnames(dt)        # [:channel1, :channel2, :channel3]
feat_funcs = get_features(dt)     # (mean, std, maximum, minimum, median)
n_windows = get_nwindows(dt)      # 6

# Document experiment
println("Processing: \$aggrtype mode")
println("Variables: \$(join(var_names, ", "))")
println("Features: \$(join(nameof.(feat_funcs), ", "))")
println("Windows: \$n_windows per dimension")
```

# Accessor Functions
- `get_dataset(dt)`: Extract the processed feature matrix
- `get_featureid(dt)`: Get feature metadata vector
- `get_reducefunc(dt)`: Get the reduction function used
- `get_aggrtype(dt)`: Get the processing mode
- `get_vnames(dt)`: Get unique variable names
- `get_features(dt)`: Get unique feature functions
- `get_nwindows(dt)`: Get maximum window number

# Indexing
DataTreatment supports array-like indexing:
```julia
dt[1, :]      # First sample (row)
dt[:, 1]      # First feature (column)
dt[1:10, :]   # First 10 samples
size(dt)      # Dataset dimensions
length(dt)    # Number of features
```

# See Also
- [`FeatureId`](@ref): Individual feature metadata
- [`aggregate`](@ref): Multi-element aggregation
- [`reducesize`](@ref): Flatten to tabular format
- [`@evalwindow`](@ref): Window evaluation macro
"""
struct DataTreatment{T, S} <: AbstractDataTreatment
    dataset    :: AbstractMatrix{T}
    featureid  :: Vector{FeatureId}
    reducefunc :: Base.Callable
    aggrtype   :: Symbol

    function DataTreatment(
        X          :: AbstractMatrix,
        aggrtype   :: Symbol;
        vnames     :: Vector{<:ValidVnames},
        win        :: Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
        features   :: Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
        reducefunc :: Base.Callable=mean
    )
        is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
            "does not contain multidimensional data."))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        win isa Base.Callable && (win = (win,))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        intervals = @evalwindow first(X) win...
        nwindows = prod(length.(intervals))

        Xresult, Xinfo = if aggrtype == :aggregate begin
            (aggregate(X, intervals; features),
            if nwindows == 1
                # single window: apply to whole time series
                [FeatureId(v, f, 1)
                    for f in features, v in vnames] |> vec
            else
                # multiple windows: apply to each interval
                [FeatureId(v, f, i)
                    for i in 1:nwindows, f in features, v in vnames] |> vec
            end
            )
        end

        elseif aggrtype == :reducesize begin
            (reducesize(X, intervals; reducefunc),
            [FeatureId(v, reducefunc, 1)
                for v in vnames] |> vec
            )
        end

        else
            error("Unknown treatment type: $treat")
        end

        new{eltype(Xresult), core_eltype(Xresult)}(Xresult, Xinfo, reducefunc, aggrtype)
    end

    function DataTreatment(
        X      :: AbstractDataFrame,
        args...;
        vnames :: Union{Vector{<:ValidVnames}, Nothing}=nothing,
        kwargs...
    )
        isnothing(vnames) && (vnames = propertynames(X))
        DataTreatment(Matrix(X), args...; vnames, kwargs...)
    end
end

# value access methods
Base.getproperty(dt::DataTreatment, s::Symbol) = getfield(dt, s)
Base.propertynames(::DataTreatment) = (:dataset, :featureid, :reducefunc, :aggrtype)

get_dataset(dt::DataTreatment)    = dt.dataset
get_featureid(dt::DataTreatment)  = dt.featureid
get_reducefunc(dt::DataTreatment) = dt.reducefunc
get_aggrtype(dt::DataTreatment)   = dt.aggrtype

# Convenience methods for common operations
get_vnames(dt::DataTreatment)   = unique([get_vname(f)   for f in dt.featureid])
get_features(dt::DataTreatment) = unique([get_feature(f) for f in dt.featureid])
get_nwindows(dt::DataTreatment) = maximum([get_nwin(f)   for f in dt.featureid])

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

export FeatureId, DataTreatment
export get_vname, get_feature, get_nwin
export get_vnames, get_features, get_nwindows
export get_dataset, get_featureid, get_reducefunc, get_aggrtype

end
