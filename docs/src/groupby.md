```@meta
CurrentModule = DataTreatments
```
# [Grouping](@id grouping)

## Why group features?

Often you work with datasets whose columns have different units of measurement. A simple example is a medical dataset where we have audio files together with recordings of electromagnetic pulses.

Several machine learning algorithms explicitly require the input data to be normalized (typically in a 0–1 range). We could normalize column by column, but this risks flattening the dataset with a consequent loss of information.

That is why the possibility to group columns according to a certain logic is useful.

The DataTreatments package provides this functionality for multidimensional datasets via the `groupby` keyword in [`TreatmentGroup`](@ref) and the `groupby_split` keyword in [`get_dataset`](@ref).

## Supported groupby fields

Features produced by [`aggregate`](@ref) carry metadata ([`AggregateFeat`](@ref)) that can be grouped by the following attributes:

| Field | Description |
|---|---|
| `:dims` | Source dimensionality (e.g., 1 for time series, 2 for images) |
| `:vname` | Original variable name (e.g., `"ts1"`, `"img2"`) |
| `:nwin` | Number of windows used during aggregation |
| `:feat` | Feature function applied (e.g., `mean`, `maximum`) |

## Dimension splitting

Before any user-defined grouping, multidimensional datasets are automatically split by dimensionality via `_split_md_by_dims`. This ensures that 1D time series and 2D images are never mixed in the same [`MultidimDataset`](@ref).

## Single-field grouping

Pass a single `Symbol` to `groupby` in a [`TreatmentGroup`](@ref):

```julia
using DataTreatments, DataFrames, Statistics

df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5)],
)

dt = DataTreatment(df)

# Group by variable name → one dataset per variable
ds = get_dataset(dt,
    TreatmentGroup(
        aggrfunc=aggregate(features=(mean, maximum)),
        groupby=:vname,
    ),
    groupby_split=true
)
# Returns 2 DataFrames: one for ts1, one for ts2

# Group by feature function → one dataset per function
ds = get_dataset(dt,
    TreatmentGroup(
        aggrfunc=aggregate(features=(mean, maximum)),
        groupby=:feat,
    ),
    groupby_split=true
)
# Returns 2 DataFrames: one for mean columns, one for maximum columns
```

## Multi-level grouping

Pass a `Tuple` of `Symbol`s to perform hierarchical grouping. The fields are applied left to right, producing leaf groups at the deepest level:

```julia
# Group first by variable name, then by feature function
ds = get_dataset(dt,
    TreatmentGroup(
        aggrfunc=aggregate(features=(mean, maximum)),
        groupby=(:vname, :feat),
    ),
    groupby_split=true
)
# Returns 4 DataFrames: (ts1, mean), (ts1, maximum), (ts2, mean), (ts2, maximum)
```

## Combining with other options

Grouping composes with all other [`get_dataset`](@ref) options:

```julia
# Multiple TreatmentGroups with different groupby
ds = get_dataset(dt,
    TreatmentGroup(
        dims=2,
        aggrfunc=reducesize(win=(splitwindow(nwindows=3),)),
    ),
    TreatmentGroup(
        dims=1,
        aggrfunc=aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        ),
        groupby=(:vname, :feat),
    ),
    groupby_split=true,
    leftover_ds=false
)
```

!!! note
    When `groupby_split=false` (the default), the `groupby` metadata is stored but all features remain concatenated in a single [`MultidimDataset`](@ref) per dimensionality. Set `groupby_split=true` to actually split the output.

!!! note
    Grouping only applies to [`AggregateFeat`](@ref) metadata. [`ReduceFeat`](@ref) produced by [`reducesize`](@ref) does not carry the same attribute fields and is not split by `groupby`.

## API Reference

```@docs
DataTreatments._split_md_by_dims
DataTreatments._groupby(info::AbstractVector{<:AggregateFeat{T}}, fields::Tuple{Vararg{Symbol}}) where T
DataTreatments._groupby(info::AbstractVector{<:AggregateFeat{T}}, field::Symbol) where T
```