# ---------------------------------------------------------------------------- #
#                                  groupby                                     #
# ---------------------------------------------------------------------------- #
"""
    groupby(data::DataTreatment, fields...)

Group rows in a DataTreatment dataset by one or more feature attributes.

## Purpose

The `groupby` function enables hierarchical grouping of DataTreatment columns based on 
feature properties stored in the `FeatureId` structure. This is essential for operations 
that require consistent computation across groups rather than column-by-column, preventing 
data inconsistencies and unwanted flattening.

## FeatureId Structure

Each feature in the dataset carries metadata through its `FeatureId`:
- **vname**: The name/identifier of the feature
- **nwin**: The window number associated with multidimensional levels
- **feat**: The feature used to reduce or aggregate multidimensional elements to 
  manageable computational size

## Use Cases

### Normalization Across Groups
Instead of normalizing column-by-column (which can flatten or damage the dataset), 
you can compute normalization coefficients spanning an entire group, ensuring 
consistent scaling across related features.

**Example**: Normalize all channels of a multi-channel sensor reading together, 
rather than independently per channel.

### Multi-Level Grouping
Group hierarchically by multiple attributes (e.g., first by feature name, then by 
window, then by reduction method) for complex data analysis pipelines.

## Returns

Tuple of:
- **groups**: Vector of index groups mapping to original dataset positions
- **feat_groups**: Corresponding FeatureId groups for each group of indices
"""
function groupby(df::DataTreatment, fields::Symbol...)
    # initial setup Vector{Vector} of all indexes and featureids
    featureids = get_featureid(df)

    _groupby(get_dataset(df), featureids, collect(fields))
end

# TODO
# function groupby(df::DataTreatment, fields::Vector{Vector{Symbol}})
#     colnames = propertynames(df)
#     used = unique(vcat(fields...))
#     left = setdiff(colnames, used)
#     !isempty(left) && push!(fields, left)

#     groups = Vector{DataFrame}(undef, length(fields))

#     @inbounds for i in eachindex(fields)
#         groups[i] = df[!, fields[i]]
#     end

#     return groups
# end

"""
    groupby(df::DataFrame, fields::Vector{Vector{Symbol}})

Group DataFrame columns into multiple sub-DataFrames based on pre-defined column groups.

## Arguments
- `df`: Source `DataFrame`.
- `fields`: Vector of column-name vectors (each inner vector is one group).

## Behavior
- Computes any leftover columns not listed in `fields` and appends them as a final group.
- Returns a `Vector{DataFrame}` where each element is `df[!, fields[i]]`.

## Example
```julia
using DataFrames

df = DataFrame(
    sepal_length = rand(3),
    sepal_width  = rand(3),
    petal_length = rand(3),
    petal_width  = rand(3),
)

fields = [[:sepal_length, :petal_length], [:sepal_width]]
groups = groupby(df, fields)
```
"""
function groupby(df::DataFrame, fields::Vector{Vector{Symbol}})
    colnames = propertynames(df)
    used = unique(vcat(fields...))
    left = setdiff(colnames, used)
    !isempty(left) && push!(fields, left)

    groups = Vector{DataFrame}(undef, length(fields))

    @inbounds for i in eachindex(fields)
        groups[i] = df[!, fields[i]]
    end

    return groups
end

groupby(df::DataFrame, fields::Vector{Symbol}) = groupby(df, [fields])

# ---------------------------------------------------------------------------- #
#                 internal _groupby for DataTreatment struct                   #
# ---------------------------------------------------------------------------- #
function _groupby(::Matrix{T}, f::Vector{<:AbstractDataFeature}, args...) where T
    _groupby(collect(1:length(f)), f, args...)
end

function _groupby(
    idxs::Vector{Int64},
    datafeats::Vector{<:AbstractDataFeature},
    mask::BitVector
)
    length(mask) == length(idxs) || throw(ArgumentError(
        "BitVector length ($(length(mask))) must match number of indices ($(length(idxs)))."))

    groups     = [idxs[mask], idxs[.!mask]]
    feat_groups = [datafeats[mask], datafeats[.!mask]]

    return groups, feat_groups
end

function _groupby(
    idxs::Vector{Int64},
    datafeats::Vector{<:AbstractDataFeature},
    fields::Vector{Vector{Symbol}}
)
    fnames = unique(get_vname.(datafeats))
    used = unique(vcat(fields...))

    invalid = filter(f -> f ∉ fnames, used)
    !isempty(invalid) && throw(ArgumentError(
        "The following column names were not found: $(invalid). Available: $(fnames)"))
        
    left = setdiff(fnames, used)
    !isempty(left) && push!(fields, left)

    groups     = Vector{Vector{Int}}(undef, length(fields))
    feat_groups = Vector{Vector{<:AbstractDataFeature}}(undef, length(fields))

    for (i, group_names) in enumerate(fields)
        mask = findall(fid -> get_vname(fid) ∈ group_names, datafeats)
        groups[i]      = idxs[mask]
        feat_groups[i] = datafeats[mask]
    end

    return groups, feat_groups
end

function _groupby(
    idxs::Vector{Int64},
    datafeats::Vector{<:AbstractDataFeature},
    fields::Vector{Symbol}
)
    # this function performs multi-level grouping (recursive).
    # - idxs: current groups of column indices
    # - datafeats: current groups of FeatureId metadata (aligned with idxs)
    # - fields: remaining fields to group by (e.g., [:feat, :vname, :nwin])

    isempty(fields) && return [idxs], [datafeats]

    # split by the first field
    sub_idxs, sub_datafeats = _groupby(idxs, datafeats, first(fields))

    isempty(fields[2:end]) && return sub_idxs, sub_datafeats

    # recursively group each sub-group by remaining fields
    all_groups = Vector{Vector{Int}}()
    all_feats = Vector{Vector{<:AbstractDataFeature}}()

    for (sidx, sfeat) in zip(sub_idxs, sub_datafeats)
        groups, feats = _groupby(sidx, sfeat, fields[2:end])
        append!(all_groups, groups)
        append!(all_feats, feats)
    end

    return all_groups, all_feats
end

function _groupby(
    idxs::Vector{Int64},
    datafeats::Vector{<:AbstractDataFeature},
    field::Symbol
)
    # casi speciali:
    # questo if prende su tutti i casi speciali
    # per ora si limita a catturare il caso speciale :all
    # quando il raggruppamento non deve essere columwise(default)
    # ma tutte le variabili devono essere raggruppate insieme
    # per esempio per normalizzazioni particolari
    field == :all && return [idxs], [datafeats]

    getter = Symbol(:get_, field)
    feat_type = eltype(datafeats)

    # check if the getter exists for the given feature type
    if !isdefined(DataTreatments, getter) || !hasmethod(eval(getter), Tuple{feat_type})
        throw(ArgumentError(
            "Field `$(field)` is not available for $(feat_type). "))
    end

    # dynamically construct the appropriate getter function (get_vname, get_nwin, or get_feat)
    # based on the field symbol passed as argument
    getter = @eval $(Symbol(:get_, field))

    # extract unique values of the specified field across all DataFeature objects
    # this determines how many distinct groups we'll create
    feats = unique(getter.(datafeats))

    # pre-allocate vectors to store grouped indices and their corresponding DataFeatures
    groups = Vector{Vector{Int}}(undef, length(feats))
    feat_groups = Vector{Vector{<:AbstractDataFeature}}(undef, length(feats))

    # iterate through each unique field value and partition the data
    for (i, f) in enumerate(feats)
        # find all positions where the field value matches current unique value
        mask = findall(fid -> getter(fid) == f, datafeats)
        # store the original indices that belong to this group
        groups[i] = idxs[mask]
        # store the corresponding DataFeature objects for this group
        feat_groups[i] = datafeats[mask]
    end

    return groups, feat_groups
end
