# ---------------------------------------------------------------------------- #
#                                     types                                    #
# ---------------------------------------------------------------------------- #
const DefaultAggrFunc = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))

# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
"""
    DataTreatment

The core structure of the `DataTreatments` package.

Its purpose is to collect all metadata useful for working with a dataset, while also
accepting user directives in the form of [`TreatmentGroup`](@ref) structures. Through
these, the user can customize how the dataset is partitioned and how multidimensional
data is handled (see [`TreatmentGroup`](@ref) and [`aggregate`](@ref) / [`reducesize`](@ref)).

`DataTreatment` stores not only the **static metadata** about the dataset — retrieved
via [`DatasetStructure`](@ref) — but also all **user-specified preferences** for
processing.

## Lazy Design

`DataTreatment` is designed to be **lazy** in order to maximize scalability. While it
provides Base methods, getters, and a set of convenience methods for extracting and
formatting the dataset contents, it is very likely that users will need to create their
own custom methods. The lazy approach makes this possible: the raw dataset and all
metadata are stored and accessible, but expensive computations (such as building the
final processed datasets) are deferred until explicitly requested.

## Fields

- `dataset::Matrix`: The raw dataset matrix as provided by the user.
- `ds_struct::DatasetStructure`: Static metadata about the dataset (types, dimensions,
  missing/NaN indices per column). See [`DatasetStructure`](@ref).
- `t_groups::Vector{TreatmentGroup}`: User-defined treatment groups that specify how
  to partition columns and how to process multidimensional data.
- `float_type::Type`: The floating-point type used for numeric processing (default: `Float64`).

## Constructors

    DataTreatment(
        dataset::Matrix,
        vnames::Vector{String},
        treatments::Base.Callable...;
        float_type::Type=Float64
    )

    DataTreatment(df::DataFrame, args...; kwargs...)

### Arguments
- `dataset::Matrix`: A matrix where each column is a feature. Elements may be scalars,
  arrays, or contain `missing` / `NaN` values.
- `vnames::Vector{String}`: Column names corresponding to each column of `dataset`.
- `treatments::Base.Callable...`: One or more [`TreatmentGroup`](@ref) constructors
  (typically in curried form). Each callable receives the internal [`DatasetStructure`](@ref)
  and returns a configured `TreatmentGroup`. Defaults to a single group that aggregates
  all multidimensional columns using `(maximum, minimum, mean)` over the whole window.
- `float_type::Type`: The floating-point type for numeric output (default: `Float64`).
- `df::DataFrame`: Alternatively, pass a `DataFrame` directly; column names are
  extracted automatically via `names(df)`.

## Examples

```julia
using DataTreatments, DataFrames, Statistics

# From a DataFrame with default treatment (aggregate with max, min, mean)
dt = DataTreatment(df)

# From a matrix with explicit column names
dt = DataTreatment(Matrix(df), names(df))

# With custom treatment groups
dt = DataTreatment(
    df,
    TreatmentGroup(dims=0),                          # scalars: no processing
    TreatmentGroup(dims=1, aggrfunc=aggregate(       # 1D arrays: custom aggregation
        win=(splitwindow(4),),
        features=(mean, std)
    )),
)

# With a specific float type
dt = DataTreatment(df; float_type=Float32)

# Lazy access — build processed datasets only when needed
datasets = get_datasets(dt)
```

See also: [`DatasetStructure`](@ref), [`TreatmentGroup`](@ref), [`aggregate`](@ref),
[`reducesize`](@ref)
"""
struct DataTreatment
    dataset::Matrix
    ds_struct::DatasetStructure
    t_groups::Vector{TreatmentGroup}
    float_type::Type

    function DataTreatment(
        dataset::Matrix,
        vnames::Vector{String},
        treatments::Base.Callable...=TreatmentGroup(
            aggrfunc=DefaultAggrFunc,
        );
        float_type::Type=Float64
    )
        ds_struct = DatasetStructure(dataset, vnames)
        t_groups = [treat(ds_struct) for treat in treatments]

        new(dataset, ds_struct, t_groups, float_type)
    end

    DataTreatment(df::DataFrame, args...; kwargs...) =
        DataTreatment(Matrix(df), names(df), args...; kwargs...)
end

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
Base.size(dt::DataTreatment) = size(dt.dataset)
Base.length(dt::DataTreatment) = size(dt.dataset, 2)
Base.eachindex(dt::DataTreatment) = Base.OneTo(length(dt))
Base.iterate(dt::DataTreatment, state=1) =
    state > length(dt) ? nothing : (@view(dt.dataset[:, state]), state + 1)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_dataset(dt::DataTreatment)

Returns the raw dataset matrix.
"""
get_dataset(dt::DataTreatment) = dt.dataset

"""
    get_ds_struct(dt::DataTreatment)

Returns the dataset structure containing metadata about the dataset.
"""
get_ds_struct(dt::DataTreatment) = dt.ds_struct

"""
    get_t_groups(dt::DataTreatment)
    get_t_groups(dt::DataTreatment, i::Int)

Returns the treatment groups. If `i` is provided, returns the `i`-th treatment group.
"""
get_t_groups(dt::DataTreatment) = dt.t_groups
get_t_groups(dt::DataTreatment, i::Int) = dt.t_groups[i]

"""
    get_float_type(dt::DataTreatment)

Returns the floating-point type used for processing.
"""
get_float_type(dt::DataTreatment) = dt.float_type

"""
    get_nrows(dt::DataTreatment)

Returns the number of rows in the dataset.
"""
get_nrows(dt::DataTreatment) = size(dt.dataset, 1)

"""
    get_ncols(dt::DataTreatment)

Returns the number of columns in the dataset.
"""
get_ncols(dt::DataTreatment) = size(dt.dataset, 2)

# ---------------------------------------------------------------------------- #
#                             internal functions                               #
# ---------------------------------------------------------------------------- #
"""
    _build_datasets(
        id::Vector,
        dataset::Matrix,
        ds_struct::DatasetStructure,
        idxs::Vector{Int},
        aggrfunc::Base.Callable,
        float_type::Type=Float64
    ) -> (ds_td, ds_tc, ds_md)

Internal function that partitions the selected columns of a dataset into three
macro categories based on their data type, and stores each partition in the
appropriate structure:

- **Discrete data**: columns whose element type is neither `AbstractFloat` nor
  `AbstractArray` (e.g., categorical labels, strings, integers). Stored in a
  [`DiscreteDataset`](@ref).
- **Continuous data**: columns whose element type is a subtype of `AbstractFloat`
  (scalar numeric values). Stored in a [`ContinuousDataset`](@ref).
- **Multidimensional data**: columns whose element type is a subtype of
  `AbstractArray` (e.g., time series, images, spectrograms). Stored in a
  [`MultidimDataset`](@ref), processed according to the provided `aggrfunc`.

Columns whose detected type is `nothing` are silently skipped.

# Arguments
- `id::Vector`: An identifier tag for the dataset partition (e.g.,
  `[:treatment_group, 1]`), propagated to each sub-dataset for traceability.
- `dataset::Matrix`: The raw dataset matrix.
- `ds_struct::DatasetStructure`: Precomputed metadata about the dataset
  (types, dimensions, validity indices). See [`DatasetStructure`](@ref).
- `idxs::Vector{Int}`: Column indices to consider (typically from a
  [`TreatmentGroup`](@ref)).
- `aggrfunc::Base.Callable`: The aggregation or reduction function to apply
  to multidimensional columns (e.g., the result of [`aggregate`](@ref) or
  [`reducesize`](@ref)).
- `float_type::Type`: The floating-point type for numeric output
  (default: `Float64`).

# Returns
A tuple of three elements `(ds_td, ds_tc, ds_md)`:
- `ds_td`: A [`DiscreteDataset`](@ref) or an empty vector `[]` if no discrete
  columns are present.
- `ds_tc`: A [`ContinuousDataset`](@ref) or an empty vector `[]` if no
  continuous columns are present.
- `ds_md`: A [`MultidimDataset`](@ref) or an empty vector `[]` if no
  multidimensional columns are present.

See also: [`DataTreatment`](@ref), [`TreatmentGroup`](@ref),
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`MultidimDataset`](@ref)
"""
function _build_datasets(
    id::Vector,
    dataset::Matrix,
    ds_struct::DatasetStructure,
    idxs::Vector{Int},
    aggrfunc::Base.Callable,
    float_type::Type=Float64
)
    valtype = get_datatype(ds_struct)

    td_cols = idxs ∩ findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    ds_td = isempty(td_cols) ?
        nothing :
        DiscreteDataset(id, dataset, ds_struct, td_cols)
    ds_tc = isempty(tc_cols) ?
        nothing :
        ContinuousDataset(id, dataset, ds_struct, tc_cols, float_type)
    ds_md = isempty(md_cols) ?
        nothing :
        MultidimDataset(id, dataset, ds_struct, md_cols, aggrfunc, float_type)

    return ds_td, ds_tc, ds_md
end

"""
    _view_md_by_dims(md_datasets, ds_struct) -> Dict{Int, Vector{@NamedTuple{md::MultidimDataset, idxs::Vector{Int}}}}

Returns a lazy view of `MultidimDataset`s grouped by the dimensionality (`dims`) of their
columns, as recorded in `ds_struct`. Unlike [`_split_md_by_dims`](@ref), this does **not**
construct new `MultidimDataset` objects — it only returns references to the originals
along with the column indices belonging to each dimensionality group.

# Returns
A `Dict` mapping each unique dimensionality to a vector of named tuples `(md, idxs)`,
where `md` is the original `MultidimDataset` and `idxs` are the column indices of that
dimensionality within it.
"""
function _split_md_by_dims(ds_md::MultidimDataset)
    dims = get_dims(ds_md)
    unique_dims = unique(get_dims(ds_md))

    idxs = [filter(i -> dims[i] == ud, eachindex(dims)) for ud in unique_dims]
    # result = Dict{Int, Vector{@NamedTuple{md::MultidimDataset, idxs::Vector{Int}}}}()

    # for md in md_datasets
    #     col_idxs = get_idxs(md)

    #     # group columns by their dimensionality
    #     dims_groups = Dict{Int, Vector{Int}}()
    #     for ci in col_idxs
    #         d = dims_info[ci]
    #         if !haskey(dims_groups, d)
    #             dims_groups[d] = Int[]
    #         end
    #         push!(dims_groups[d], ci)
    #     end

    #     for (d, group_idxs) in dims_groups
    #         entry = (md=md, idxs=group_idxs)
    #         if !haskey(result, d)
    #             result[d] = typeof(entry)[]
    #         end
    #         push!(result[d], entry)
    #     end
    # end

    # return result
    ds_md
end

# ---------------------------------------------------------------------------- #
#                             custom lazy methods                              #
# ---------------------------------------------------------------------------- #
function get_treatments_datasets(dt::DataTreatment; split=true, dataframe=false)
    treats = get_t_groups(dt)
    idxs = get_idxs(treats)

    dataset = get_dataset(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    ntreats = length(treats)
    ds_td = Vector{Union{Nothing,DiscreteDataset}}(undef, ntreats)
    ds_tc = Vector{Union{Nothing,ContinuousDataset}}(undef, ntreats)
    ds_md = Vector{Union{Nothing,MultidimDataset}}(undef, ntreats)

    Threads.@threads for i in eachindex(treats)
        ds_td[i], ds_tc[i], ds_md[i] = _build_datasets(
            [:treatment_group, i],
            dataset,
            ds_struct,
            idxs[i],
            get_aggrfunc(treats[i]),
            float_type
        )
    end

    return(
        all(isnothing, ds_td) ? nothing : filter(!isnothing, ds_td),
        all(isnothing, ds_tc) ? nothing : filter(!isnothing, ds_tc),
        all(isnothing, ds_md) ? nothing : _split_md_by_dims.(filter(!isnothing, ds_md))
    )
end

function get_leftover_datasets(dt::DataTreatment; split=true, dataframe=false)
    treats = get_t_groups(dt)
    idxs = setdiff(collect(eachindex(dt)), reduce(vcat, get_idxs(treats)))

    dataset = get_dataset(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    ds_td, ds_tc, ds_md = _build_datasets(
        [:leftover, 1],
        dataset,
        ds_struct,
        idxs,
        DefaultAggrFunc,
        float_type
    )

    return (
        isnothing(ds_td) ? nothing : [ds_td],
        isnothing(ds_tc) ? nothing : [ds_tc],
        isnothing(ds_md) ? nothing : _split_md_by_dims(ds_md)
    )
end

function get_datasets(dt::DataTreatment; split=true, dataframe=false)
    dataset = get_dataset(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    # first step: defines datasets based on treatment groups
    treats = get_t_groups(dt)
    idxs = get_idxs(treats)

    for i in eachindex(treats)
        _build_datasets(
            [:treatment_group, i],
            dataset,
            ds_struct,
            idxs[i],
            get_aggrfunc(treats[i]),
            float_type
        )
    end

    # second step: defines datasets on leftover indicies
end

function get_datasets(dt::DataTreatment, grp::TreatmentGroup; dataframe=false)

end