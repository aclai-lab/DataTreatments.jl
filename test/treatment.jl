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
@btime f_groupby = _features_groupby(get_featureid(Xaggregated), (:nwin, :feat));
# 1.181 ms (13100 allocations: 477.08 KiB)
# 876.865 μs (9500 allocations: 353.33 KiB)


function _features_groupby(
    featureid :: Vector{<:FeatureId},
    aggrby    :: Tuple{Vararg{Symbol}}
)::Vector{Vector{Int64}}
    res = Dict{Any, Vector{Int64}}()
    sizehint!(res, length(featureid) ÷ 2)
    @inbounds for i in eachindex(featureid)
        key = Tuple(getproperty(featureid[i], field) for field in aggrby)
        push!(get!(res, key, Int64[]), i)
    end
    return collect(values(res))  # return the grouped indices
end
# 1.164 ms (13100 allocations: 477.08 KiB)
# 876.865 μs (9500 allocations: 353.33 KiB)

function _features_groupby(
    featureid::Vector{<:FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int64}}
    # Pre-allocate with estimated capacity
    res = Dict{Tuple, Vector{Int64}}()
    
    @inbounds for (i, fid) in enumerate(featureid)
        key = ntuple(j -> getfield(fid, aggrby[j]), length(aggrby))
        group = get!(res, key, Int64[])
        push!(group, i)
    end
    
    return collect(values(res))
end
# 355.512 μs (8852 allocations: 275.00 KiB)

using StructArrays

function _features_groupby(
    featureid::Vector{<:FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int64}}
    # Create keys as a vector of tuples
    keys = map(featureid) do fid
        ntuple(j -> getfield(fid, aggrby[j]), length(aggrby))
    end
    
    # Group indices by keys
    groups = Dict{eltype(keys), Vector{Int64}}()
    sizehint!(groups, length(featureid) ÷ 2)
    
    @inbounds for (i, key) in enumerate(keys)
        push!(get!(groups, key, Int64[]), i)
    end
    
    return collect(values(groups))
end
# 178.830 μs (6337 allocations: 218.52 KiB)

function _features_groupby(
    featureid::Vector{<:FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int64}}
    # Create keys using map
    keys = map(fid -> ntuple(j -> getfield(fid, aggrby[j]), length(aggrby)), featureid)
    
    # Group indices by keys
    groups = Dict{eltype(keys), Vector{Int64}}()
    sizehint!(groups, length(featureid) ÷ 2)
    
    # Use map to build the groups
    map(enumerate(keys)) do (i, key)
        push!(get!(groups, key, Int64[]), i)
    end
    
    return collect(values(groups))
end
# 630.097 μs (20368 allocations: 674.34 KiB)

function _features_groupby(
    featureid::Vector{<:FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int64}}
    # Extract keys for all features
    keys = map(fid -> ntuple(j -> getfield(fid, aggrby[j]), length(aggrby)), featureid)
    
    # Build groups dictionary
    groups = Dict{eltype(keys), Vector{Int64}}()
    sizehint!(groups, length(unique(keys)))
    
    foreach(enumerate(keys)) do (i, key)
        push!(get!(groups, key, Int64[]), i)
    end
    
    return collect(values(groups))
end
# 741.499 μs (24633 allocations: 655.42 KiB)

########################################################
function _features_groupby(
    featureid::Vector{FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int64}}
    # Pre-allocate with estimated capacity
    res = Dict{Tuple, Vector{Int64}}()
    
    @inbounds for (i, fid) in enumerate(featureid)
        key = ntuple(j -> getfield(fid, aggrby[j]), length(aggrby))
        group = get!(res, key, Int64[])
        push!(group, i)
    end
    
    return collect(values(res))
end

function minmax_normalize(
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
    minmax_normalize.(icols, min, max)
    return X
end

function minmax_normalize(
    v::AbstractArray{<:AbstractArray{<:Real}},
    min::Real,
    max::Real
)
    return minmax_normalize.(v, min, max)
end

function minmax_normalize(
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
function normalize_dataset(
    X::AbstractMatrix{T},
    featureid::Vector{FeatureId};
    min_quantile::AbstractFloat=0.0,
    max_quantile::AbstractFloat=1.0,
    group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
) where {T<:Number}
    @inbounds Threads.@threads for g in _features_groupby(featureid, group)
        minmax_normalize(
            view(X, :, g);
            min_quantile = min_quantile,
            max_quantile = max_quantile,
            col_quantile = false
        )
    end
end
# 27.041 ms (34556 allocations: 66.62 MiB)
# 8.127 ms (34642 allocations: 66.62 MiB)
# 7.258 ms (34642 allocations: 66.62 MiB)


# function _normalize_dataset(Xdf::AbstractDataFrame, featureid::Vector{<:FeatureId}; kwargs...)
#     original_names = names(Xdf)
#     DataFrame(_normalize_dataset!(Matrix(Xdf), featureid; kwargs...), original_names)
# end

n = normalize_dataset(Xaggregated.dataset, get_featureid(Xaggregated))