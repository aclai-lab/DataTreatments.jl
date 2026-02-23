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
#                              internal _groupby                               #
# ---------------------------------------------------------------------------- #
function _groupby(::Matrix{T}, featureids::Vector{<:AbstractFeatureId}, fields::Vector{Symbol}) where T
    # initial setup Vector{Vector} of all indexes
    idxs = [[1:length(featureids)...]]

    _groupby(idxs, [featureids], fields)
end

function _groupby(idxs::Vector{Vector{Int64}}, featureids::Vector{<:Vector{<:AbstractFeatureId}}, fields::Vector{Symbol})
    # this function performs multi-level grouping (recursive).
    # - idxs: current groups of column indices
    # - featureids: current groups of FeatureId metadata (aligned with idxs)
    # - fields: remaining fields to group by (e.g., [:feat, :vname, :nwin])

    ngroups = length(featureids)
    all_groups = Vector{Vector{Int}}()
    all_feats = Vector{Vector{<:AbstractFeatureId}}()

    for i in 1:ngroups
        # first, split the i-th group by the first field
        sub_idxs, sub_featureids = _groupby(idxs[i], featureids[i], fields[1])
        
        if length(fields) == 1
            # base case: only one field left, append the final groups
            append!(all_groups, sub_idxs)
            append!(all_feats, sub_featureids)
        else
            # recursive case: keep grouping by the remaining fields
            groups, feats = _groupby(sub_idxs, sub_featureids, fields[2:end])
            append!(all_groups, groups)
            append!(all_feats, feats)
        end
    end

    return all_groups, all_feats
end

function _groupby(idxs::Vector{Int64}, featureids::Vector{<:AbstractFeatureId}, field::Symbol)
    # dynamically construct the appropriate getter function (get_vname, get_nwin, or get_feat)
    # based on the field symbol passed as argument
    getter = @eval $(Symbol(:get_, field))

    # extract unique values of the specified field across all FeatureId objects
    # this determines how many distinct groups we'll create
    feats = unique(getter.(featureids))

    # pre-allocate vectors to store grouped indices and their corresponding FeatureIds
    groups = Vector{Vector{Int}}(undef, length(feats))
    feat_groups = Vector{Vector{<:AbstractFeatureId}}(undef, length(feats))

    # iterate through each unique field value and partition the data
    for (i, f) in enumerate(feats)
        # find all positions where the field value matches current unique value
        mask = findall(fid -> getter(fid) == f, featureids)
        # store the original indices that belong to this group
        groups[i] = idxs[mask]
        # store the corresponding FeatureId objects for this group
        feat_groups[i] = featureids[mask]
    end

    return groups, feat_groups
end