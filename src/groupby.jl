# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
@inline field_getter(field::Symbol) =
    field == :type ? get_type :
    field == :vname ? get_vname :
    field == :nwin ? get_nwin :
    field == :feat ? get_feat :
    field == :reducefunc ? get_reducefunc :
    throw(ArgumentError("Unknown field: $field"))

# ---------------------------------------------------------------------------- #
#                        groupby for DataFrame struct                          #
# ---------------------------------------------------------------------------- #
"""
    groupby(df::DataFrame, fields::Vector{Vector{Symbol}}) -> Vector{DataFrame}
    groupby(df::DataFrame, fields::Vector{Symbol})         -> Vector{DataFrame}
    groupby(df::DataFrame, fields::Symbol)                 -> Vector{DataFrame}

Group DataFrame columns into multiple sub-DataFrames based on pre-defined column groups.
Any column not listed in `fields` is collected into a final leftover group.

## Arguments
- `df`: Source `DataFrame`.
- `fields`: Column names to group. Accepts:
  - `Symbol`: a single column name → produces 2 groups (the column + leftovers)
  - `Vector{Symbol}`: a single flat group → produces 2 groups (the group + leftovers)
  - `Vector{Vector{Symbol}}`: multiple explicit groups → produces N+1 groups (each group + leftovers)

## Returns
A `Vector{DataFrame}` where each element is a sub-DataFrame with the selected columns.
The last element always contains the leftover columns not explicitly listed.

## Example
```julia
using DataFrames
df = DataFrame(a=1:3, b=4:6, c=7:9, d=10:12)

groups = groupby(df, :a)
length(groups)               # 2
propertynames(groups[1])     # [:a]
propertynames(groups[2])     # [:b, :c, :d]

groups = groupby(df, [[:a, :b], [:c]])
length(groups)               # 3
propertynames(groups[3])
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
#              external groupby for AbstractDataFeature struct                 #
# ---------------------------------------------------------------------------- #
"""
    groupby(dt::DataTreatment, args...) -> (see internal methods)

Dispatch `groupby` on the `datafeature` vector of a `DataTreatment`.
All arguments are forwarded to the appropriate internal method.

## Example
```julia
using DataTreatments, DataFrames

df = DataFrame(a=1:3, b=4:6, c=7:9, d=10:12)
y  = ["cat", "dog", "cat"]
dt = DataTreatment(df, y)

# group by BitVector
groups = DataTreatments.groupby(dt, BitVector([1, 0, 0, 1]))
length(collect(groups)) # 2

# group by column names
groups = DataTreatments.groupby(dt, [[:a, :b], [:c]])
length(collect(groups)) # 3
```
"""
function groupby(df::DataTreatment, args...)
    featureids = get_datafeature(df)
    groupby(featureids, args...)
end

# ---------------------------------------------------------------------------- #
#              internal groupby for AbstractDataFeature struct                 #
# ---------------------------------------------------------------------------- #
"""
    groupby(datafeats::Vector{<:AbstractDataFeature}, mask::BitVector)

Split a feature vector into two groups based on a `BitVector` mask.
Features at positions where `mask[i] == true` go into the first group,
the remaining into the second.

## Arguments
- `datafeats`: Vector of `AbstractDataFeature` elements.
- `mask`: A `BitVector` of the same length as `datafeats`.

## Returns
A generator of 2 elements, each being a lazy view into `datafeats`.

## Errors
Throws `ArgumentError` if `length(mask) != length(datafeats)`.

## Example
```julia
using DataTreatments, DataFrames

df    = DataFrame(a=1:3, b=4:6, c=7:9, d=10:12)
y     = ["cat", "dog", "cat"]
dt    = DataTreatment(df, y)
feats = get_datafeature(dt)

mask   = BitVector([1, 0, 0, 1])
groups = DataTreatments.groupby(feats, mask)
length(collect(groups)) # 2
```
"""
function groupby(
    datafeats::Vector{<:AbstractDataFeature},
    mask::BitVector
)
    length(mask) == length(datafeats) || throw(ArgumentError(
        "BitVector length ($(length(mask))) must match number of datafeats ($(length(datafeats)))."))

    return ((@view(datafeats[findall(m)]) for _ in (nothing,)) for m in (mask, .!mask))
end

"""
    groupby(datafeats::Vector{<:AbstractDataFeature}, fields::Vector{Vector{Symbol}})

Group a tabular feature vector by variable names.
Features whose `vname` belongs to `fields[i]` are collected into the i-th group.
Any `vname` not listed in `fields` is appended as a final leftover group.

## Arguments
- `datafeats`: Vector of `AbstractDataFeature` elements (typically `TabularFeat`).
- `fields`: Vector of symbol vectors, each defining one group by variable name.

## Returns
A generator of lazy views into `datafeats`, one per group.

## Errors
Throws `ArgumentError` if any symbol in `fields` does not match any `vname` in `datafeats`.

## Example
```julia
using DataTreatments, DataFrames

df    = DataFrame(a=1:3, b=4:6, c=7:9, d=10:12)
y     = ["cat", "dog", "cat"]
feats = get_datafeature(DataTreatment(df, y))

groups = DataTreatments.groupby(feats, [[:a, :b], [:c]])
length(collect(groups)) # 3  (:a, :b in group 1; :c in group 2; :d leftover)
```
"""
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
"""
    groupby(datafeats::AbstractVector{<:AbstractDataFeature}, field::Symbol)

Group a multidimensional feature vector by a single feature attribute.
Valid field names are: `:type`, `:vname`, `:nwin`, `:feat`, `:reducefunc`.

## Arguments
- `datafeats`: Vector of `AbstractDataFeature` elements.
- `field`: A `Symbol` naming the attribute to group by.
  Pass `:all` to return a single group with all indices.

## Returns
A generator of `Vector{Int}` (original dataset column indices), one per unique value
of the chosen attribute.

## Errors
Throws `ArgumentError` if `field` is not a recognised attribute name.

## Example
```julia
using DataTreatments, DataFrames

Xts = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:10],
    channel2 = [rand(200, 120) for _ in 1:10],
    channel3 = [rand(200, 120) for _ in 1:10]
)
yts = ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"]
win = adaptivewindow(nwindows=3, overlap=0.2)

feats = DataTreatment(Xts, yts; groups=:vname, aggrtype=:reducesize, win)

collected = get_groups(feats)
```
"""
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

"""
    groupby(datafeats::AbstractVector{<:AbstractDataFeature}, fields::Vector{Symbol})

Perform hierarchical (multi-level) grouping by applying `groupby` recursively
on each sub-group for every field in `fields`, left to right.

## Arguments
- `datafeats`: Vector of `AbstractDataFeature` elements.
- `fields`: Ordered vector of attribute symbols to group by sequentially.
  Each element must be a valid field for `field_getter` (`:type`, `:vname`,
  `:nwin`, `:feat`, `:reducefunc`).

## Returns
A `Vector{Base.Generator}`, where each element is itself a generator of
`Vector{Int}` index groups produced by the remaining fields.
When flattened (e.g. with `reduce(vcat, collect.(collect(...)))`), the result
is a `Vector{Vector{Int}}` giving every leaf group in hierarchical order.

## Example
```julia
using DataTreatments, DataFrames

Xts = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:10],
    channel2 = [rand(200, 120) for _ in 1:10],
    channel3 = [rand(200, 120) for _ in 1:10]
)
yts = ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"]
win = adaptivewindow(nwindows=3, overlap=0.2)

feats = DataTreatment(Xts, yts; aggrtype=:aggregate, win,
              groups=[:vname, :feat],
              features=(mean, maximum))

collected = get_groups(feats)
```
"""
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