# ---------------------------------------------------------------------------- #
#                              split & _groupby                                #
# ---------------------------------------------------------------------------- #
@inline field_getter(field::Symbol) =
    field == :dims ? f -> f.dims :
    field == :vname ? f -> f.vname :
    field == :nwin ? f -> f.nwin :
    field == :feat ? f -> f.feat :
    throw(ArgumentError("Unknown field: $field"))

"""
    _split_md_by_dims(ds_md::MultidimDataset) -> Vector{MultidimDataset}

Split a [`MultidimDataset`](@ref) into multiple `MultidimDataset`s, one for each
unique source dimensionality of its features.

When a `MultidimDataset` contains features originating from arrays of different
dimensionalities (e.g., 1D time series and 2D spectrograms), this function groups
them by dimensionality and returns a separate `MultidimDataset` for each group.

# Arguments
- `ds_md::MultidimDataset`: A multidimensional dataset potentially containing
  features with heterogeneous source dimensionalities.

# Returns
A `Vector{MultidimDataset}` where each element contains only features sharing the
same dimensionality. The length of the returned vector equals the number of unique
dimensionalities present in `ds_md`.
"""
function _split_md_by_dims(ds_md::MultidimDataset)
    dims = get_dims(ds_md)
    unique_dims = unique(get_dims(ds_md))

    idxs = [filter(i -> dims[i] == ud, collect(eachindex(dims))) for ud in unique_dims]

    return [ds_md[idx] for idx in idxs]
end

"""
    _groupby(datafeats::MultidimDataset{<:AggregateFeat}, fields::Vector{Symbol})

Perform hierarchical (multi-level) grouping by applying `_groupby` recursively
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
"""
function _groupby(
    datafeats::MultidimDataset,
    fields::Tuple{Vararg{Symbol}}
)
    # this function performs multi-level grouping (recursive).
    # - idxs: current groups of column indices
    # - datafeats: current groups of FeatureId metadata (aligned with idxs)
    # - fields: remaining fields to group by (e.g., [:feat, :vname, :nwin])

    # split by the first field
    sub_idxs = _groupby(datafeats, first(fields))

    remaining = fields[2:end]
    isempty(remaining) && return collect(sub_idxs)

    # recursively group each sub-group by remaining fields
    all_groups = Vector{Vector{Int}}()

    for i in sub_idxs
        groups = _groupby(@view(datafeats[i]), remaining)
        append!(all_groups, collect(groups))
    end

    return all_groups
end

"""
    _groupby(datafeats::MultidimDataset{<:AggregateFeat}, field::Symbol)

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
"""
function _groupby(
    datafeats::MultidimDataset,
    field::Symbol
)
    field == :all && return (i for i in eachindex(datafeats))

    getter = field_getter(field)
    infos = datafeats.info
    vals = [getter(infos[i]) for i in eachindex(infos)]
    unique_vals = unique(vals)
    idxs = (findall(==(v), vals) for v in unique_vals)
    return (get_idx.(@view(infos[i])) for i in idxs)
end