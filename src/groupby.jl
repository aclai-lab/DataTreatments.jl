# ---------------------------------------------------------------------------- #
#                                  groupby                                     #
# ---------------------------------------------------------------------------- #
@inline field_getter(field::Symbol) =
    field == :dims ? f -> f.dims :
    field == :vname ? f -> f.vname :
    field == :nwin ? f -> f.nwin :
    field == :feat ? f -> f.feat :
    throw(ArgumentError("Unknown field: $field"))

"""
    _groupby(info::AbstractVector{<:AggregateFeat{T}}, fields::Tuple{Vararg{Symbol}}) where T

Perform hierarchical (multi-level) grouping by applying `_groupby` recursively
on each sub-group for every field in `fields`, left to right.

# Arguments
- `info::AbstractVector{<:AggregateFeat{T}}`: Vector of `AggregateFeat` metadata
  entries, one per feature column.
- `fields::Tuple{Vararg{Symbol}}`: Ordered tuple of attribute symbols to group by
  sequentially. Each element must be one of the symbols supported by
  [`:dims`, `:vname`, `:nwin`, `:feat`.

# Returns
A `Vector{Vector{Int}}` where each inner vector contains the original column
indices belonging to one leaf group in the hierarchical grouping.

See also: [`_groupby(::AbstractVector{<:AggregateFeat}, ::Symbol)`](@ref)
"""
function _groupby(
    info::AbstractVector{<:AggregateFeat{T}},
    fields::Tuple{Vararg{Symbol}}
) where T
    sub_idxs = _groupby(info, first(fields))

    remaining = fields[2:end]
    isempty(remaining) && return collect(sub_idxs)

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
  by [`:dims`, `:vname`, `:nwin`, `:feat`.
  The special value `:all` returns each index as its own singleton group.

# Returns
A generator yielding `Vector{Int}` groups, where each vector contains original
column indices (obtained via `get_idx`) sharing the same value for `field`.

When `field == :all`, returns a generator yielding one index per element
(i.e., no grouping is performed).

# Errors
Throws `ArgumentError` (via `field_getter` if `field` is not a
recognised attribute name.

See also: [`_groupby(::AbstractVector{<:AggregateFeat}, ::Tuple{Vararg{Symbol}})`](@ref)
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
    return (get_subid.(info[i]) for i in idxs)
end

