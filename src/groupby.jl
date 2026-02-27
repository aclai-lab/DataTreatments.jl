# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
@inline field_getter(field::Symbol) =
    field == :type ? get_type :
    field == :vname ? get_vname :
    field == :nwin ? get_nwin :
    field == :feat ? get_feat :
    throw(ArgumentError("Unknown field: $field"))

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
function groupby(df::DataTreatment, args...)
    featureids = get_datafeature(df)
    groupby(featureids, args...)
end

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
groupby(df::DataFrame, fields::Symbol) = groupby(df, [[fields]])

# ---------------------------------------------------------------------------- #
#              internal groupby for AbstractDataFeature struct                 #
# ---------------------------------------------------------------------------- #
function groupby(
    datafeats::Vector{<:AbstractDataFeature},
    mask::BitVector
)
    length(mask) == length(datafeats) || throw(ArgumentError(
        "BitVector length ($(length(mask))) must match number of datafeats ($(length(datafeats)))."))

    return ((@view(datafeats[findall(m)]) for _ in (nothing,)) for m in (mask, .!mask))
end

function groupby(
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

    return (
        (@view(datafeats[findall(fid -> get_vname(fid) ∈ group_names, datafeats)]) for _ in (nothing,))
        for group_names in fields
    )
end

# ---------------------------------------------------------------------------- #
#               for multidimensional datatreatments set only                   #
# ---------------------------------------------------------------------------- #
function groupby(
    datafeats::AbstractVector{<:AbstractDataFeature},
    fields::Vector{Symbol}
)
    # this function performs multi-level grouping (recursive).
    # - idxs: current groups of column indices
    # - datafeats: current groups of FeatureId metadata (aligned with idxs)
    # - fields: remaining fields to group by (e.g., [:feat, :vname, :nwin])

    # split by the first field
    sub_idxs = groupby(datafeats, first(fields))

    isempty(fields[2:end]) && return sub_idxs

    # recursively group each sub-group by remaining fields
    all_groups = Vector{Base.Generator}()

    for i in sub_idxs
        groups = groupby(@view(datafeats[i]), fields[2:end])
        push!(all_groups, groups)
    end

    return all_groups
end

function groupby(
    datafeats::AbstractVector{<:AbstractDataFeature},
    field::Symbol
)
    field == :all && return (i for i in eachindex(datafeats))

    getter = field_getter(field)
    vals = getter.(datafeats)
    unique_vals = unique(vals)
    idxs = (findall(==(v), vals) for v in unique_vals)
    return (get_id.(@view(datafeats[i])) for i in idxs)
end