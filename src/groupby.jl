# ---------------------------------------------------------------------------- #
#                               GroupTreatment                                 #
# ---------------------------------------------------------------------------- #
struct GroupResult <: AbstractDataTreatment
    groups::Vector{Vector{Int}}
    feat_groups::Vector{Vector{FeatureId}}
end


# ---------------------------------------------------------------------------- #
#                                  groupby                                     #
# ---------------------------------------------------------------------------- #
"""
    @groupby(data, fields...)

Group rows in a DataTreatment dataset by one or more feature attributes.

## Purpose

The `@groupby` macro enables hierarchical grouping of DataTreatment columns based on 
feature properties stored in the `FeatureId` structure. This is essential for operations 
that require consistent computation across groups rather than column-by-column, preventing 
data inconsistencies and unwanted flattening.

## FeatureId Structure

Each feature in the dataset carries metadata through its `FeatureId`:
- **vname**: The name/identifier of the feature
- **win**: The window number associated with multidimensional levels
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
macro groupby(d, fields...)
    isempty(fields) && error("@atest requires at least one field")

    quote
        df = $(esc(d))
        # initial setup Vector{Vector} of all indexes and featureids
        featureids = get_featureid(df)
        idxs = [[1:length(featureids)...]]

        _groupby(idxs, [featureids], [$(esc.(fields)...)])
    end
end

function _groupby(idxs::Vector{Vector{Int64}}, featureids::Vector{Vector{FeatureId}}, fields::Vector{Symbol})
    if length(fields) == 1
        ngroups = length(featureids)
        results = Vector{Tuple}(undef, ngroups)

        for i in 1:ngroups
            # results[i] = _groupby(idxs[i], featureids[i], fields[1])
            g, f = _groupby(idxs[i], featureids[i], fields[1])
            results[i] = (g, f)
        end

        # Flatten results
        all_groups = vcat([r[1] for r in results]...)
        all_feats = vcat([r[2] for r in results]...)
        return all_groups, all_feats
    end

    ngroups = length(featureids)
    all_groups = Vector{Vector{Int}}()
    all_feats = Vector{Vector{FeatureId}}()

    for i in 1:ngroups
        sub_idxs, sub_featureids = _groupby(idxs[i], featureids[i], fields[1])
        # Recursively group and flatten as we go
        groups, feats = _groupby(sub_idxs, sub_featureids, fields[2:end])
        append!(all_groups, groups)
        append!(all_feats, feats)
    end

    return all_groups, all_feats
end

function _groupby(idxs::Vector{Int64}, featureids::Vector{FeatureId}, field::Symbol)
    # dynamically construct the appropriate getter function (get_vname, get_win, or get_feat)
    # based on the field symbol passed as argument
    getter = @eval $(Symbol(:get_, field))

    # extract unique values of the specified field across all FeatureId objects
    # this determines how many distinct groups we'll create
    feats = unique(getter.(featureids))

    # pre-allocate vectors to store grouped indices and their corresponding FeatureIds
    groups = Vector{Vector{Int}}(undef, length(feats))
    feat_groups = Vector{Vector{FeatureId}}(undef, length(feats))

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