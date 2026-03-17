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
    _groupby(info::AbstractVector{<:AggregateFeat{T}}, fields::Tuple{Vararg{Symbol}}) where T

Perform hierarchical (multi-level) grouping by applying `_groupby` recursively
on each sub-group for every field in `fields`, left to right.

# Arguments
- `info::AbstractVector{<:AggregateFeat{T}}`: Vector of `AggregateFeat` metadata
  entries, one per feature column.
- `fields::Tuple{Vararg{Symbol}}`: Ordered tuple of attribute symbols to group by
  sequentially. Each element must be one of the symbols supported by
  [`field_getter`](@ref): `:dims`, `:vname`, `:nwin`, `:feat`.

# Returns
A `Vector{Vector{Int}}` where each inner vector contains the original column
indices belonging to one leaf group in the hierarchical grouping.

# Example
```julia
# Group first by variable name, then by feature function
groups = _groupby(info, (:vname, :feat))
# → [[1, 4], [2, 5], [3, 6]]  (example indices)
```

See also: [`_groupby(::AbstractVector{<:AggregateFeat}, ::Symbol)`](@ref),
[`field_getter`](@ref)
"""
function _groupby(
    info::AbstractVector{<:AggregateFeat{T}},
    fields::Tuple{Vararg{Symbol}}
) where T
    # this function performs multi-level grouping (recursive).
    # - idxs: current groups of column indices
    # - info: current groups of FeatureId metadata (aligned with idxs)
    # - fields: remaining fields to group by (e.g., [:feat, :vname, :nwin])

    # split by the first field
    sub_idxs = _groupby(info, first(fields))

    remaining = fields[2:end]
    isempty(remaining) && return collect(sub_idxs)

    # recursively group each sub-group by remaining fields
    all_groups = Vector{Vector{Int}}()

    for i in sub_idxs
        groups = _groupby(@view(info[i]), remaining)
        append!(all_groups, collect(groups))
    end

    return all_groups
end

"""
    _groupby(info::AbstractVector{<:AggregateFeat{T}}, field::Symbol) where T

Group feature metadata by a single attribute and return the corresponding
original column indices for each unique value.

# Arguments
- `info::AbstractVector{<:AggregateFeat{T}}`: Vector of `AggregateFeat` metadata
  entries, one per feature column.
- `field::Symbol`: The attribute to group by. Must be one of the symbols supported
  by [`field_getter`](@ref): `:dims`, `:vname`, `:nwin`, `:feat`.
  The special value `:all` returns each index as its own singleton group.

# Returns
A generator yielding `Vector{Int}` groups, where each vector contains original
column indices (obtained via `get_idx`) sharing the same value for `field`.

When `field == :all`, returns a generator yielding one index per element
(i.e., no grouping is performed).

# Errors
Throws `ArgumentError` (via [`field_getter`](@ref)) if `field` is not a
recognised attribute name.

See also: [`_groupby(::AbstractVector{<:AggregateFeat}, ::Tuple{Vararg{Symbol}})`](@ref),
[`field_getter`](@ref)
"""
function _groupby(
    info::AbstractVector{<:AggregateFeat{T}},
    field::Symbol
) where T
    field == :all && return (i for i in eachindex(info))
    getter = field_getter(field)
    vals = [getter(info[i]) for i in eachindex(info)]
    unique_vals = unique(vals)
    idxs = (findall(==(v), vals) for v in unique_vals)
    return (get_idx.(@view(info[i])) for i in idxs)
end
