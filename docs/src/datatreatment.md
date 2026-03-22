```@meta
CurrentModule = DataTreatments
```

# DataTreatment

The [`DataTreatment`](@ref) struct is the core object of the DataTreatments package.
It stores raw data alongside precomputed metadata, and defers all processing 
until [`get_dataset`](@ref) is called with user-defined [`TreatmentGroup`](@ref) directives.

## Construction

A `DataTreatment` can be built from a `Matrix` (with column names) or directly from a `DataFrame`:

```julia
using DataTreatments, DataFrames, Statistics

# From a matrix
dataset = Matrix{Any}([
    1.0    "hello"   missing
    2.0    "world"   4.0
    NaN    "foo"     5.0
])
vnames = ["numeric", "text", "with_missing"]
dt = DataTreatment(dataset, vnames)

# From a DataFrame
df = DataFrame(dataset, vnames)
dt = DataTreatment(df)

# With a specific float type
dt = DataTreatment(df; float_type=Float32)
```

## Dataset Extraction

Processed datasets are obtained lazily via [`get_dataset`](@ref). Treatment groups
control which columns are selected and how multidimensional data is aggregated or reduced.

### Entry Points

- [`get_tabular`](@ref): Collects all tabular-like datasets, including discrete, continuous, and aggregated multidimensional data. Especially useful for heterogeneous datasets with both tabular and multidimensional columns, where you want to aggregate multidimensional data into tabular form.
- [`get_multidim`](@ref): Collects all reduced multidimensional datasets, focusing on features that remain multidimensional after treatment.

### Examples

```julia
df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
)

dt = DataTreatment(df; float_type=Float32)

# Default treatment (aggregate with max, min, mean over whole window)
ds = get_dataset(dt)

# Custom treatment groups
ds = get_dataset(dt,
    TreatmentGroup(dims=0),
    TreatmentGroup(dims=1, aggrfunc=aggregate(
        win=(splitwindow(nwindows=3),),
        features=(mean, std)
    )),
)

# Only leftover columns (not assigned to any treatment group)
ds = get_dataset(dt, TreatmentGroup(dims=1); treatment_ds=false)

# Default treatment (aggregate with max, min, mean over whole window)
tabular_ds = get_tabular(dt)

# Custom treatment groups for tabular extraction
tabular_ds = get_tabular(dt,
    TreatmentGroup(dims=0),
    TreatmentGroup(dims=1, aggrfunc=aggregate(
        win=(splitwindow(nwindows=3),),
        features=(mean, std)
    )),
)
```

## Getters

| Method | Description |
|---|---|
| `get_data(dt)` | Raw data matrix |
| `get_target(dt)` | Target vector (or `nothing`) |
| `get_ds_struct(dt)` | [`DatasetStructure`](@ref) metadata |
| `get_float_type(dt)` | Floating-point type used for processing |
| `get_nrows(dt)` | Number of rows |
| `get_ncols(dt)` | Number of columns |

## Base Methods

| Method | Description |
|---|---|
| `size(dt)` | `(nrows, ncols)` tuple |
| `length(dt)` | Number of columns |
| `eachindex(dt)` | Column index iterator |
| `iterate(dt)` | Iterate over column views |

## API Reference

```@docs
DataTreatment

get_tabular(
    dt::DataTreatment,
    args...;
    kwargs...
)

get_multidim(
    dt::DataTreatment,
    args...;
    kwargs...
)

get_dataset(
        dt::DataTreatment,
        treatments::Vararg{Base.Callable};
        treatment_ds::Bool,
        leftover_ds::Bool,
    )
get_data(dt::DataTreatment)
get_target(dt::DataTreatment)
get_ds_struct(dt::DataTreatment)
get_float_type(dt::DataTreatment)
get_nrows(dt::DataTreatment)
get_ncols(dt::DataTreatment)
```

## Internals

```@docs
DataTreatments._build_datasets
DataTreatments._get_treatments_datasets
DataTreatments._get_leftover_datasets
```