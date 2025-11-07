using Test
using DataTreatments

using Statistics

X = rand(100)
wfunc = splitwindow(nwindows=10)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals)

X = rand(100, 120)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals; reducefunc=maximum)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = reducesize(Xmatrix, intervals; reducefunc=std)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
features = (mean, maximum)
result = aggregate(Xmatrix, intervals; features)


########################################################
using SoleData.Artifacts
# fill your Artifacts.toml file;
@test_nowarn Artifacts.fillartifacts()

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

win = DataTreatments.adaptivewindow(nwindows=6, overlap=0.2)
features = (maximum, minimum, mean, std, var)

Xreduced = DataTreatment(Xts, :reducesize; win, features)
Xaggregated = DataTreatment(Xts, :aggregate; win, features)

########################################################
function _features_groupby(
    featureid::Vector{<:FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int}}
    res = Dict{Any, Vector{Int}}()
    for (i, g) in enumerate(featureid)
        key = Tuple(getproperty(g, field) for field in aggrby)
        push!(get!(res, key, Int[]), i)
    end
    return collect(values(res))  # Return the grouped indices
end

########################################################
minmax_normalize(c, args...; kwars...) = minmax_normalize!(deepcopy(c), args...; kwars...)

"""
    minmax_normalize!(X; kwargs...)
    minmax_normalize!(X, min::Real, max::Real)

Apply min-max normalization to scale values to the range [0,1], modifying the input in-place.

# Common Methods
- `minmax_normalize!(X::AbstractMatrix; kwargs...)`: Normalize a matrix
- `minmax_normalize!(df::AbstractDataFrame; kwargs...)`: Normalize a DataFrame
- `minmax_normalize!(md::MultiData.MultiDataset, frame_index::Integer; kwargs...)`: Normalize a specific frame in a multimodal dataset
- `minmax_normalize!(v::AbstractArray{<:Real}, min::Real, max::Real)`: Normalize an array using specific min/max values
- `minmax_normalize!(v::AbstractArray{<:AbstractArray{<:Real}}, min::Real, max::Real)`: Normalize an array of arrays

# Arguments
- `X`: The data to normalize (matrix, DataFrame, or MultiDataset)
- `frame_index`: For MultiDataset, the index of the frame to normalize
- `min::Real`: Minimum value for normalization (when provided directly)
- `max::Real`: Maximum value for normalization (when provided directly)

# Keyword Arguments
- `min_quantile::Real=0.0`: Lower quantile threshold for normalization
  - `0.0`: Use the absolute minimum (no outlier exclusion)
  - `> 0.0`: Use the specified quantile as minimum (e.g., 0.05 excludes bottom 5% as outliers)
- `max_quantile::Real=1.0`: Upper quantile threshold for normalization
  - `1.0`: Use the absolute maximum (no outlier exclusion)
  - `< 1.0`: Use the specified quantile as maximum (e.g., 0.95 excludes top 5% as outliers)
- `col_quantile::Bool=true`: How to calculate quantiles
  - `true`: Calculate separate quantiles for each column (column-wise normalization)
  - `false`: Calculate global quantiles across the entire dataset

# Returns
The input data, normalized in-place.

# Throws
- `DomainError`: If min_quantile < 0, max_quantile > 1, or max_quantile ≤ min_quantile

# Details
## Matrix/DataFrame normalization:
When normalizing matrices or DataFrames, this function:
1. Validates the quantile parameters
2. Determines min/max values based on the specified quantiles
3. If `col_quantile=true`, calculates separate min/max for each column
4. If `col_quantile=false`, uses the same min/max across the entire dataset
5. Applies the normalization to transform values to the [0,1] range

## Array normalization:
For direct array normalization with provided min/max values:
1. If min equals max, returns an array filled with 0.5 values
2. Otherwise, scales values to [0,1] range using the formula: (x - min) / (max - min)
"""
# function minmax_normalize!(
#     md::MultiData.MultiDataset,
#     frame_index::Integer;
#     min_quantile::Real = 0.0,
#     max_quantile::Real = 1.0,
#     col_quantile::Bool = true,
# )
#     return minmax_normalize!(
#         MultiData.modality(md, frame_index);
#         min_quantile = min_quantile,
#         max_quantile = max_quantile,
#         col_quantile = col_quantile
#     )
# end

function minmax_normalize!(
    X::AbstractMatrix;
    min_quantile::Real = 0.0,
    max_quantile::Real = 1.0,
    col_quantile::Bool = true,
)
    min_quantile < 0.0 &&
        throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
    max_quantile > 1.0 &&
        throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
    max_quantile ≤ min_quantile &&
        throw(DomainError("max_quantile must be greater then min_quantile"))

    icols = eachcol(X)

    if (!col_quantile)
        # look for quantile in entire dataset
        itdf = Iterators.flatten(Iterators.flatten(icols))
        min = StatsBase.quantile(itdf, min_quantile)
        max = StatsBase.quantile(itdf, max_quantile)
    else
        # quantile for each column
        itcol = Iterators.flatten.(icols)
        min = StatsBase.quantile.(itcol, min_quantile)
        max = StatsBase.quantile.(itcol, max_quantile)
    end
    minmax_normalize!.(icols, min, max)
    return X
end

function minmax_normalize!(
    df::AbstractDataFrame;
    kwargs...
)
    minmax_normalize!(Matrix(df); kwargs...)
end

function minmax_normalize!(
    v::AbstractArray{<:AbstractArray{<:Real}},
    min::Real,
    max::Real
)
    return minmax_normalize!.(v, min, max)
end

function minmax_normalize!(
    v::AbstractArray{<:Real},
    min::Real,
    max::Real
)
    if (min == max)
        return repeat([0.5], length(v))
    end
    min = float(min)
    max = float(max)
    max = 1 / (max - min)
    rt = StatsBase.UnitRangeTransform(1, 1, true, [min], [max])
    # This function doesn't accept Integer
    return StatsBase.transform!(rt, v)
end

# ---------------------------------------------------------------------------- #
#                               normalize dataset                              #
# ---------------------------------------------------------------------------- #
"""
    normalize_dataset(
        X::AbstractMatrix{T},
        featureid::Vector{<:SoleFeatures.FeatureId};
        min_quantile::AbstractFloat=0.00,
        max_quantile::AbstractFloat=1.00,
        group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
    ) where {T<:Number}

Normalize the dataset matrix `X` by applying min-max normalization to groups of features.

## Parameters
- `X`: The input matrix to be normalized in-place
- `featureid`: A vector of feature information objects that contain metadata about each feature
- `min_quantile`: The quantile to use as the minimum value (default: 0.00)
  - When set to 0.00, uses the absolute minimum value
  - Higher values (e.g., 0.05) ignore lower outliers by using the specified quantile instead
- `max_quantile`: The quantile to use as the maximum value (default: 1.00)
  - When set to 1.00, uses the absolute maximum value
  - Lower values (e.g., 0.95) ignore upper outliers by using the specified quantile instead
- `group`: A tuple of symbols representing fields in the `FeatureId` objects to group by (default: (:nwin, :feat))
  - Features with the same values for these fields will be normalized together
  - For example, with the default (:nwin, :feat), features from the same window and of the same type
    will be normalized as a group, preserving their relative scale

## Details
The function performs group-wise normalization, which is essential when working with features that 
should maintain their relative scales. For example, when working with time series data, different 
measures (min, max, mean) applied to the same window should be normalized together to preserve 
their relationships.
"""
function normalize_dataset(
    X::AbstractMatrix{T},
    featureid::Vector{<:FeatureId};
    min_quantile::AbstractFloat=0.0,
    max_quantile::AbstractFloat=1.0,
    group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
) where {T<:Number}
    for g in _features_groupby(featureid, group)
        minmax_normalize!(
            view(X, :, g);
            min_quantile = min_quantile,
            max_quantile = max_quantile,
            col_quantile = false
        )
    end
end

# function _normalize_dataset(Xdf::AbstractDataFrame, featureid::Vector{<:FeatureId}; kwargs...)
#     original_names = names(Xdf)
#     DataFrame(_normalize_dataset!(Matrix(Xdf), featureid; kwargs...), original_names)
# end