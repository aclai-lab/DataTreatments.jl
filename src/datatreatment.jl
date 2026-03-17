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

# Purpose

Its purpose is to collect all metadata useful for working with a dataset, while also
accepting user directives in the form of [`TreatmentGroup`](@ref) structures. Through
these, the user can customize how the dataset is partitioned and how multidimensional
data is handled (see [`TreatmentGroup`](@ref) and [`aggregate`](@ref) / [`reducesize`](@ref)).

!!! note
    TreatmentGroup directives are **not** given at the time of `DataTreatment` creation, 
    but are instead specified lazily when calling `get_dataset`. 
    This allows users to flexibly define how the dataset should be filtered, grouped, 
    and processed only when extraction is needed.

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

# Fields

- `data::Matrix`: The raw data matrix (features × samples).
- `target::Union{Nothing,TargetStructure}`: The target vector or structure, if supervised.
- `ds_struct::DatasetStructure`: Metadata about the dataset, such as types, dimensions, and missing values.
- `t_groups::Union{Nothing,Vector{TreatmentGroup}}`: User-defined treatment groups 
  for feature selection and processing (set lazily).
- `float_type::Type`: The floating-point type used for numeric processing.

# Usage

This struct is constructed automatically from a matrix or DataFrame and is used as the core object 
for all further dataset manipulations in the package.

```julia-repl
using DataTreatments, DataFrames, Statistics
```

```@example
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

See also: [`get_dataset`](@ref), [`TreatmentGroup`](@ref), [`DatasetStructure`](@ref)
"""
struct DataTreatment
    data::Matrix
    target::Union{Nothing,TargetStructure}
    ds_struct::DatasetStructure
    float_type::Type

    function DataTreatment(
        data::Matrix,
        vnames::Vector{String},
        target::Union{Nothing,AbstractVector}=nothing;
        float_type::Type=Float64
    )
        isa(target, AbstractVector) && (target = TargetStructure(target))
        ds_struct = DatasetStructure(data, vnames)

        new(data, target, ds_struct, float_type)
    end

    DataTreatment(df::DataFrame, args...; kwargs...) =
        DataTreatment(Matrix(df), names(df), args...; kwargs...)
end

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
Base.size(dt::DataTreatment) = size(dt.data)
Base.length(dt::DataTreatment) = size(dt.data, 2)
Base.eachindex(dt::DataTreatment) = Base.OneTo(length(dt))
Base.iterate(dt::DataTreatment, state=1) =
    state > length(dt) ? nothing : (@view(dt.data[:, state]), state + 1)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_data(dt::DataTreatment)

Returns the raw data matrix.
"""
get_data(dt::DataTreatment) = dt.data

"""
    get_target(dt::DataTreatment)

Returns the target vector for supervised datasets.
"""
get_target(dt::DataTreatment) = dt.target

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
        data::Matrix,
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
- `DataFrames::Matrix`: The raw DataFrames matrix.
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
    data::Matrix,
    ds_struct::DatasetStructure,
    idxs::Vector{Int},
    aggrfunc::Base.Callable;
    float_type::Type=Float64
)
    valtype = get_datatype(ds_struct)

    td_cols = idxs ∩ findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    ds_td = isempty(td_cols) ?
        nothing :
        DiscreteDataset(id, data, ds_struct, td_cols)
    ds_tc = isempty(tc_cols) ?
        nothing :
        ContinuousDataset(id, data, ds_struct, tc_cols, float_type)
    ds_md = isempty(md_cols) ?
        nothing :
        MultidimDataset(id, data, ds_struct, md_cols, aggrfunc, float_type)

    return ds_td, ds_tc, ds_md
end



"""
    _get_treatments_datasets(dt::DataTreatment) -> Vector{AbstractDataset}

Extract the processed datasets using the [`TreatmentGroup`](@ref) directives
specified by the user during the construction of the [`DataTreatment`](@ref) object.

Returns a flat `Vector{AbstractDataset}` where each element is one of:
- [`DiscreteDataset`](@ref): columns with categorical/discrete data.
- [`ContinuousDataset`](@ref): columns with scalar numeric data.
- [`MultidimDataset`](@ref): columns with array-valued data (e.g., time series,
  spectrograms), processed according to the aggregation or reduction function
  specified in each treatment group.

When multidimensional columns originate from arrays of different dimensionalities
(e.g., 1D and 2D), they are automatically split into separate `MultidimDataset`s,
one per unique dimensionality. If the user specifies a finer partitioning through
multiple `TreatmentGroup`s, this partitioning is preserved in the output.

!!! note
    Only the columns covered by the user-defined treatment groups are returned.
    Columns not assigned to any `TreatmentGroup` are **not** included; use
    [`_get_leftover_datasets`](@ref) to retrieve them, or [`get_datasets`](@ref)
    to obtain both.

# Arguments
- `dt::DataTreatment`: The `DataTreatment` object containing the raw dataset and
  the user-defined treatment groups.

# Returns
A `Vector{AbstractDataset}` containing, in order:
1. All [`DiscreteDataset`](@ref)s (one per treatment group that has discrete columns).
2. All [`ContinuousDataset`](@ref)s (one per treatment group that has continuous columns).
3. All [`MultidimDataset`](@ref)s, split by source dimensionality.

Empty categories are omitted from the result.

# Examples
```julia
using DataTreatments, DataFrames, Statistics

dt = DataTreatment(
    df,
    TreatmentGroup(dims=1, aggrfunc=aggregate(features=(mean, std))),
    TreatmentGroup(dims=0),
)

datasets = _get_treatments_datasets(dt)
# e.g., [ContinuousDataset{Float64}(100×3), MultidimDataset{Float64}(100×6, dims=[1], aggregate)]
```

See also: [`DataTreatment`](@ref), [`TreatmentGroup`](@ref),
[`_get_leftover_datasets`](@ref), [`get_datasets`](@ref),
[`_build_datasets`](@ref), [`_split_md_by_dims`](@ref)
"""
function _get_treatments_datasets(dt::DataTreatment, treats::Vector{<:TreatmentGroup})
    idxs = get_idxs(treats)

    data = get_data(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    ntreats = length(treats)
    ds_td = Vector{Union{Nothing,DiscreteDataset}}(undef, ntreats)
    ds_tc = Vector{Union{Nothing,ContinuousDataset}}(undef, ntreats)
    ds_md = Vector{Union{Nothing,MultidimDataset}}(undef, ntreats)

    Threads.@threads for i in eachindex(treats)
        ds_td[i], ds_tc[i], ds_md[i] = _build_datasets(
            [:treatment_group, i],
            data,
            ds_struct,
            idxs[i],
            get_aggrfunc(treats[i]);
            # grps=get_groupby(treats[i]),
            float_type
        )
    end

    td_filtered = filter(!isnothing, ds_td)
    tc_filtered = filter(!isnothing, ds_tc)
    md_filtered = filter(!isnothing, ds_md)

    md_split = isempty(md_filtered) ? AbstractDataset[] : reduce(vcat, _split_md_by_dims.(md_filtered))

    return AbstractDataset[td_filtered; tc_filtered; md_split]
end

"""
    _get_leftover_datasets(dt::DataTreatment) -> Vector{AbstractDataset}

Complement of [`_get_treatments_datasets`](@ref): returns the dataset columns that
were **not** selected by any user-defined [`TreatmentGroup`](@ref), formatted as
a flat `Vector{AbstractDataset}`.

Leftover columns are partitioned by data type using the same logic as
[`_get_treatments_datasets`](@ref):
- Categorical/discrete columns → [`DiscreteDataset`](@ref)
- Scalar numeric columns → [`ContinuousDataset`](@ref)
- Array-valued columns → [`MultidimDataset`](@ref), processed with the default
  aggregation function (`maximum`, `minimum`, `mean` over the whole window) and
  split by source dimensionality.

!!! note
    If every column of the dataset is covered by the user-defined treatment groups,
    the returned vector will be empty.

# Arguments
- `dt::DataTreatment`: The `DataTreatment` object containing the raw dataset and
  the user-defined treatment groups.

# Returns
A `Vector{AbstractDataset}` containing, in order:
1. A [`DiscreteDataset`](@ref) for leftover categorical columns (if any).
2. A [`ContinuousDataset`](@ref) for leftover scalar numeric columns (if any).
3. One or more [`MultidimDataset`](@ref)s for leftover array-valued columns (if any),
   split by source dimensionality.

Empty categories are omitted from the result.

# Examples
```julia
using DataTreatments, DataFrames, Statistics

# Only treat 1D columns — scalars and 2D columns become leftovers
dt = DataTreatment(
    df,
    TreatmentGroup(dims=1, aggrfunc=aggregate(features=(mean, std))),
)

leftovers = _get_leftover_datasets(dt)
# e.g., [ContinuousDataset{Float64}(100×2), MultidimDataset{Float64}(100×9, dims=[2], aggregate)]
```

See also: [`DataTreatment`](@ref), [`_get_treatments_datasets`](@ref),
[`get_datasets`](@ref), [`_build_datasets`](@ref), [`_split_md_by_dims`](@ref)
"""
function _get_leftover_datasets(dt::DataTreatment, treats::Vector{<:TreatmentGroup})
    idxs = setdiff(collect(eachindex(dt)), reduce(vcat, get_idxs(treats)))

    data = get_data(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    ds_td, ds_tc, ds_md = _build_datasets(
        [:leftover, 1],
        data,
        ds_struct,
        idxs,
        DefaultAggrFunc;
        float_type
    )

    td_filtered = isnothing(ds_td) ? AbstractDataset[] : AbstractDataset[ds_td]
    tc_filtered = isnothing(ds_tc) ? AbstractDataset[] : AbstractDataset[ds_tc]
    md_split = isnothing(ds_md) ? AbstractDataset[] : _split_md_by_dims(ds_md)

    return AbstractDataset[td_filtered; tc_filtered; md_split]
end

# ---------------------------------------------------------------------------- #
#                             custom lazy methods                              #
# ---------------------------------------------------------------------------- #
"""
    get_dataset(
        dt::DataTreatment,
        treatments::Base.Callable...;
        treatment_ds=true,
        leftover_ds=true,
        matrix=false,
        dataframe=false
    )

The core function of the DataTreatments package.
Lazily extracts processed datasets from a `DataTreatment` object according to user-specified 
treatment groups and options.

# Purpose

`get_dataset` enables flexible, on-demand extraction of datasets by applying 
user-defined filters and grouping logic (via `TreatmentGroup` and related callables). 
It supports extracting only the columns and transformations the user specifies, 
while also allowing retrieval of any leftover (unassigned) columns.

This is a convenience method that concatenates the results of
[`get_treatments_datasets`](@ref) and [`get_leftover_datasets`](@ref). The returned
vector contains, in order:
1. All datasets produced by [`get_treatments_datasets`](@ref) (discrete, continuous,
   and multidimensional columns covered by user-defined [`TreatmentGroup`](@ref)s).
2. All datasets produced by [`get_leftover_datasets`](@ref) (columns not assigned to
   any treatment group, processed with the default aggregation function).

Each element is one of:
- [`DiscreteDataset`](@ref): columns with categorical/discrete data.
- [`ContinuousDataset`](@ref): columns with scalar numeric data.
- [`MultidimDataset`](@ref): columns with array-valued data, split by source
  dimensionality.

# Arguments

- `dt::DataTreatment`: The container holding the raw dataset and all metadata.
- `treatments::Base.Callable...`: One or more treatment group callables (e.g., from `TreatmentGroup`) 
  that define how to filter, group, and process columns.
- `treatment_ds::Bool=true`: If `true`, include datasets defined by the treatment groups.
- `leftover_ds::Bool=true`: If `true`, include datasets for columns not assigned to any treatment group.
- `matrix::Bool=false`: If `true`, return the concatenated data as a matrix.
- `dataframe::Bool=false`: If `true`, return the concatenated data as a DataFrame.

# Returns

- By default, returns a vector of processed datasets (`AbstractDataset`).
- If `matrix=true`, returns a single matrix of all selected data.
- If `dataframe=true`, returns a single DataFrame of all selected data.

# Example

```julia
dt = DataTreatment(df)
ds = get_dataset(dt, TreatmentGroup(dims=0), TreatmentGroup(dims=1, aggrfunc=aggregate(features=(mean, std))))
```

See also: [`DataTreatment`](@ref), [`TreatmentGroup`](@ref), [`DatasetStructure`](@ref)
"""
function get_dataset(
    dt::DataTreatment,
    treatments::Base.Callable...=TreatmentGroup(aggrfunc=DefaultAggrFunc,);
    treatment_ds::Bool=true,
    leftover_ds::Bool=true,
    groupby_split::Bool=false,
    matrix::Bool=false,
    dataframe::Bool=false
)
    treats = [treat(get_ds_struct(dt)) for treat in treatments]

    ds = AbstractDataset[]
    treatment_ds && append!(ds, _get_treatments_datasets(dt, treats))
    leftover_ds && append!(ds, _get_leftover_datasets(dt, treats))

    isempty(ds) && return

    matrix && return get_data.(ds)
    dataframe && return DataFrame.(get_data.(ds), get_vnames.(ds))
    return ds
end
