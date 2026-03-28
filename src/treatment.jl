# ---------------------------------------------------------------------------- #
#                             internal functions                               #
# ---------------------------------------------------------------------------- #
function _split_md_by_dims(ds_md::MultidimDataset)
    dims = get_dims(ds_md)
    unique_dims = unique(get_dims(ds_md))

    idxs = [filter(i -> dims[i] == ud, collect(eachindex(dims))) for ud in unique_dims]

    return [ds_md[idx] for idx in idxs]
end

"""
    _build_ds(
        ids::Vector{Int},
        treat::TreatmentGroup,
        data::Matrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        float_type::Type
    ) -> (ds_td, ds_tc, ds_md)

Partitions the selected columns of a dataset into three macro categories based on their data type, 
and constructs the appropriate dataset structure for each category.

# Description

Given a set of column indices and metadata, this function separates columns into:
- **Discrete data**: Columns whose element type is neither `Float` nor `AbstractArray` 
  (e.g., categorical labels, strings, integers). These are stored in a [`DiscreteDataset`](@ref).
- **Continuous data**: Columns whose element type is a subtype of `Float` (scalar numeric values). 
  These are stored in a [`ContinuousDataset`](@ref).
- **Multidimensional data**: Columns whose element type is a subtype of `AbstractArray` 
  (e.g., time series, images, spectrograms). These are stored in a [`MultidimDataset`](@ref), processed according to the aggregation function specified in the `TreatmentGroup`.

Columns with detected type `nothing` are skipped.

# Arguments

- `ids::Vector{Int}`: Indices of columns to consider.
- `treat::TreatmentGroup`: The treatment group specifying aggregation/reduction functions and optional grouping.
- `data::Matrix`: The raw data matrix.
- `vnames::Vector{String}`: Names of the columns.
- `datastruct::NamedTuple`: Metadata about the dataset (types, dimensions, validity indices, etc.).
- `float_type::Type`: The floating-point type for numeric output.

# Returns

A tuple `(ds_td, ds_tc, ds_md)`:
- `ds_td`: A [`DiscreteDataset`](@ref) or `nothing` if no discrete columns are present.
- `ds_tc`: A [`ContinuousDataset`](@ref) or `nothing` if no continuous columns are present.
- `ds_md`: A [`MultidimDataset`](@ref) or `nothing` if no multidimensional columns are present.

See also: [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref), 
[`MultidimDataset`](@ref), [`TreatmentGroup`](@ref)
"""
function _build_ds(
    ids::Vector{Int},
    treat::TreatmentGroup,
    data::Matrix,
    vnames::Vector{String},
    datastruct::NamedTuple,
    float_type::Type{T}
) where {T<:Float}
    aggrfunc = get_aggrfunc(treat)
    valtype = datastruct.datatype
    groups = get_groupby(treat)

    td_ids = ids ∩ findall(T -> !(T <: Float) && !(T <: AbstractArray), valtype)
    tc_ids = ids ∩ findall(T -> T <: Float, valtype)
    md_ids = ids ∩ findall(T -> T <: AbstractArray, valtype)

    return (
        DiscreteDataset(td_ids, data, vnames, datastruct),
        ContinuousDataset(tc_ids, data, vnames, datastruct, float_type),
        MultidimDataset(md_ids, data, vnames, datastruct, aggrfunc, float_type, groups)
    )
end

"""
    _treatments_ds(
        data::Matrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        treats::Vector{<:TreatmentGroup},
        float_type::Type
    ) -> (Vector{DiscreteDataset}, Vector{ContinuousDataset}, Vector{MultidimDataset})

Partitions and processes the dataset columns according to a set of user-defined [`TreatmentGroup`](@ref)s, 
returning the resulting datasets grouped by type.

# Description

For each treatment group in `treats`, this function:
- Selects the columns specified by the group.
- Splits them into discrete, continuous, and multidimensional categories based on their type.
- Constructs the corresponding dataset objects: [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref), 
  and [`MultidimDataset`](@ref).
- Multidimensional datasets are further split by source dimensionality (e.g., 1D, 2D) 
  using [`_split_md_by_dims`](@ref).

The function returns three vectors, each containing all datasets of a given type 
(discrete, continuous, multidimensional) across all treatment groups. Empty categories are omitted.

# Arguments

- `data::Matrix`: The raw data matrix.
- `vnames::Vector{String}`: Names of the columns.
- `datastruct::NamedTuple`: Metadata about the dataset (types, dimensions, validity indices, etc.).
- `treats::Vector{<:TreatmentGroup}`: The treatment groups specifying which columns to select and how to process them.
- `float_type::Type`: The floating-point type for numeric output.

# Returns

A tuple of vectors:
- `Vector{DiscreteDataset}`: All discrete datasets (one per treatment group with discrete columns).
- `Vector{ContinuousDataset}`: All continuous datasets (one per treatment group with continuous columns).
- `Vector{MultidimDataset}`: All multidimensional datasets, split by source dimensionality.

See also: [`TreatmentGroup`](@ref), [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref), 
[`MultidimDataset`](@ref), [`_split_md_by_dims`](@ref)
"""
function _treatments_ds(
    data::Matrix,
    vnames::Vector{String},
    datastruct::NamedTuple,
    treats::Vector{<:TreatmentGroup},
    float_type::Type
)
    ntreats = length(treats)
    ds_td = Vector{DiscreteDataset}(undef, ntreats)
    ds_tc = Vector{ContinuousDataset}(undef, ntreats)
    ds_md = Vector{MultidimDataset}(undef, ntreats)

    Threads.@threads for i in eachindex(treats)
        ds_td[i], ds_tc[i], ds_md[i] = _build_ds(
            get_ids(treats[i]),
            treats[i],
            data,
            vnames,
            datastruct,
            float_type
        )
    end

    td_filtered = filter(!isempty, ds_td)
    tc_filtered = filter(!isempty, ds_tc)
    md_filtered = filter(!isempty, ds_md)

    md_split = isempty(md_filtered) ? MultidimDataset[] : reduce(vcat, _split_md_by_dims.(md_filtered))

    return td_filtered, tc_filtered, md_split
end

"""
    _leftovers_ds(
        data::Matrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        treats::Vector{<:TreatmentGroup},
        float_type::Type
    ) -> (Vector{AbstractDataset}, Vector{AbstractDataset}, Vector{AbstractDataset})

Identifies and processes the columns of a dataset that were **not** 
selected by any user-defined [`TreatmentGroup`](@ref), 
partitioning them by data type and returning the corresponding dataset objects.

# Description

This function finds all columns not assigned to any treatment group in `treats`, and partitions them into:
- **Discrete columns**: Categorical or non-numeric columns, returned as a [`DiscreteDataset`](@ref) (if any).
- **Continuous columns**: Scalar numeric columns, returned as a [`ContinuousDataset`](@ref) (if any).
- **Multidimensional columns**: Array-valued columns, returned as one or more [`MultidimDataset`](@ref)s (if any), 
  split by source dimensionality using [`_split_md_by_dims`](@ref).

Default aggregation functions are used for multidimensional columns.

# Arguments

- `data::Matrix`: The raw data matrix.
- `vnames::Vector{String}`: Names of the columns.
- `datastruct::NamedTuple`: Metadata about the dataset (types, dimensions, validity indices, etc.).
- `treats::Vector{<:TreatmentGroup}`: The treatment groups already assigned; columns in these groups are excluded.
- `float_type::Type`: The floating-point type for numeric output.

# Returns

A tuple of vectors:
- `Vector{AbstractDataset}`: Discrete datasets for leftover columns (empty if none).
- `Vector{AbstractDataset}`: Continuous datasets for leftover columns (empty if none).
- `Vector{AbstractDataset}`: Multidimensional datasets for leftover columns, split by dimensionality (empty if none).

Empty categories are omitted from the result.

See also: [`TreatmentGroup`](@ref), [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref), 
[`MultidimDataset`](@ref), [`_split_md_by_dims`](@ref)
"""
function _leftovers_ds(
    data::Matrix,
    vnames::Vector{String},
    datastruct::NamedTuple,
    treats::Vector{<:TreatmentGroup},
    float_type::Type
)
    ids = setdiff(datastruct.id, reduce(vcat, get_ids(treats)))

    ds_td, ds_tc, ds_md = _build_ds(
        ids,
        TreatmentGroup(datastruct, vnames; aggrfunc=DefaultAggrFunc),
        data,
        vnames,
        datastruct,
        float_type
    )

    td_filtered = isempty(ds_td) ? AbstractDataset[] : AbstractDataset[ds_td]
    tc_filtered = isempty(ds_tc) ? AbstractDataset[] : AbstractDataset[ds_tc]
    md_split = isempty(ds_md) ? AbstractDataset[] : _split_md_by_dims(ds_md)

    return td_filtered, tc_filtered, md_split
end
