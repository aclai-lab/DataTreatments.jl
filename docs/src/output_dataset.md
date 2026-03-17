```@meta
CurrentModule = DataTreatments
```
# [Output Datasets](@id output_dataset)

Output datasets are the structured results produced by [`DataTreatment`](@ref) after processing raw data. Each dataset type wraps a data matrix together with per-column metadata ([`AbstractDataFeature`](@ref) entries), enabling downstream pipelines to access not only the transformed values but also provenance information, missing/NaN indices, and categorical levels.

All concrete dataset types are subtypes of [`AbstractDataset`](@ref).

| Type | Purpose |
|------|---------|
| [`DiscreteDataset`](@ref) | Integer-coded categorical columns |
| [`ContinuousDataset{T}`](@ref) | Numeric (floating-point) scalar columns |
| [`MultidimDataset{T}`](@ref) | Multidimensional columns (time series, images, etc.) |

---

## Dataset Types

### DiscreteDataset

```@docs
DiscreteDataset
```

### ContinuousDataset

```@docs
ContinuousDataset
```

### MultidimDataset

```@docs
MultidimDataset
```

---

## `Base` Methods

All [`AbstractDataset`](@ref) subtypes support the following `Base` methods:

### Size & Iteration

| Method | Signature | Description |
|--------|-----------|-------------|
| `size` | `size(ds::AbstractDataset)` | Returns the size of the underlying `data` matrix. |
| `size` | `size(ds::AbstractDataset, d::Int)` | Returns the size along dimension `d`. |
| `length` | `length(ds::AbstractDataset)` | Number of features (equal to `length(ds.info)`). |
| `ndims` | `ndims(ds::AbstractDataset)` | Number of dimensions of the underlying `data` array. |
| `eachindex` | `eachindex(ds::AbstractDataset)` | Returns `Base.OneTo(length(ds))`. |
| `iterate` | `iterate(ds::AbstractDataset[, state])` | Iterates over the `info` metadata entries (feature descriptors). |
| `eltype` | `eltype(ds::DiscreteDataset)` | Returns `DiscreteFeat`. |
| `eltype` | `eltype(ds::ContinuousDataset{T})` | Returns `ContinuousFeat{T}`. |
| `eltype` | `eltype(ds::MultidimDataset{T})` | Returns `Union{AggregateFeat{T}, ReduceFeat{T}}`. |

### Indexing

| Method | Signature | Description |
|--------|-----------|-------------|
| `getindex` | `ds[i::Int]` | Returns a **new** dataset containing only the `i`-th column (data is copied for single columns). |
| `getindex` | `ds[idxs::AbstractVector{Int}]` | Returns a new dataset containing columns at `idxs` (data is a `@view`). |

For [`MultidimDataset`](@ref), indexing also re-indexes the `groups` field via [`_reindex_groups`](@ref), so group membership stays consistent with the column subset.

### Views

| Method | Signature | Description |
|--------|-----------|-------------|
| `view` | `@view ds[i::Integer]` | Equivalent to `ds[Int(i)]`. |
| `view` | `@view ds[idxs::AbstractVector{<:Integer}]` | Equivalent to `ds[collect(Int, idxs)]`. |
| `view` | `@view ds[r::AbstractUnitRange{<:Integer}]` | Equivalent to `ds[collect(Int, r)]`. |
| `view` | `@view ds[:]` | Equivalent to `ds[collect(eachindex(ds))]` — returns the full dataset. |

---

## Getter Methods

### Data Access

```@docs
get_data(ds::AbstractDataset)
get_data(ds::AbstractDataset, i::Int)
get_data(ds::AbstractDataset, idxs::Vector{Int})
```

!!! note "GroupBy Split"
    For `MultidimDataset{<:AggregateFeat}`, pass `groupby_split=true` to
    `get_data` to receive a `Vector` of sub-matrices, one per group defined
    by the `groups` field.

### Metadata Access

```@docs
get_info(ds::AbstractDataset)
get_info(ds::AbstractDataset, i::Int)
get_info(ds::AbstractDataset, idxs::Vector{Int})
```

### Dimensions

```@docs
get_nrows(ds::AbstractDataset)
get_ncols(ds::AbstractDataset)
```

### Variable Names

```@docs
get_vnames(ds::AbstractDataset)
get_vnames(ds::AbstractDataset, i::Int)
get_vnames(ds::AbstractDataset, idxs::Vector{Int})
```

!!! note "Aggregate naming"
    For `MultidimDataset{<:AggregateFeat}`, `get_vnames` returns composite
    names of the form `"colname,feature,win:N"` (e.g., `"ts1,maximum,win:1"`).
    Pass `groupby_split=true` to split the names vector by group.

### Source Dimensionality (MultidimDataset only)

```@docs
get_dims(ds::MultidimDataset)
get_dims(ds::MultidimDataset, i::Int)
get_dims(ds::MultidimDataset, idxs::Vector{Int})
```

### Provenance IDs

```@docs
get_idxs(ds::AbstractDataset)
get_idxs(ds::AbstractDataset, i::Int)
get_idxs(ds::AbstractDataset, idxs::Vector{Int})
```

### Groups (MultidimDataset only)

| Method | Signature | Description |
|--------|-----------|-------------|
| `has_groups` | `has_groups(ds::MultidimDataset)` | Returns `true` if column groupings are defined (i.e., `groups !== nothing`). |
| `get_groups` | `get_groups(ds::MultidimDataset)` | Returns the `groups` field — either `nothing` or a `Vector{Vector{Int}}`. |

---

## Utility Functions

### discrete\_encode

```@docs
discrete_encode(X::Matrix)
discrete_encode(x::AbstractVector)
```

Both overloads treat `missing` and `NaN` identically: they are preserved as
`missing` in the output codes and excluded from the level labels. Internally,
values are converted to strings via `string(v)` before categorization, so
the returned `levels` are always `Vector{String}`.

### \_reindex\_groups

```@docs
_reindex_groups
```

Used internally when indexing or viewing a [`MultidimDataset`](@ref) to keep
the `groups` field consistent with the new column subset.

**Behavior:**
- `_reindex_groups(nothing, idxs)` → returns `nothing`.
- `_reindex_groups(groups::Vector{Vector{Int}}, idxs)` → filters each group to
  only contain indices present in `idxs`, re-maps them to new positions
  (`1:length(idxs)`), drops empty groups, and returns `nothing` if all groups
  are eliminated.

### \_callable\_name

```@docs
_callable_name
```

---

## Display

All dataset types provide both one-line and multi-line `Base.show` methods.

### One-line (`show(io::IO, ds)`)

```
DiscreteDataset(5×3)
ContinuousDataset{Float64}(5×5)
MultidimDataset{AggregateFeat{Float64}}(5×12, dims=1, aggregate)
```

For [`MultidimDataset`](@ref), the one-line display includes:
- The unique source dimensionality (or a vector if mixed).
- The processing mode (`aggregate` or `reducesize`).

### Multi-line (`show(io::IO, ::MIME"text/plain", ds)`)

**`DiscreteDataset`:**
```
DiscreteDataset(5 rows × 3 columns)
├─ vnames: ["str_col", "sym_col", "cat_col"]
├─ columns with missing: 2
└─ levels per column: 3, 3, 3
```

**`ContinuousDataset{Float64}`:**
```
ContinuousDataset{Float64}(5 rows × 5 columns)
├─ vnames: ["V1", "V2", "V3", "V4", "V5"]
├─ columns with missing: 1
├─ columns with NaN: 2
└─ float type: Float64
```

Lines for `columns with missing` and `columns with NaN` are only printed when
the respective count is greater than zero.

**`MultidimDataset` (aggregate mode):**
```
MultidimDataset{AggregateFeat{Float64}}(5 rows × 12 columns)
├─ mode: aggregate
├─ vnames: ["ts1", "ts2", "ts3", "ts4"]
├─ columns with missing: 2
├─ columns with NaN: 1
├─ columns with internal missing: 3
├─ columns with internal NaN: 2
├─ features: maximum, minimum, mean
└─ windows: 1
```

In aggregate mode, `vnames` shows the **unique** original column names.
Additional lines for `columns with internal missing` and `columns with internal NaN`
report the number of features whose source array elements contain corrupted entries.

**`MultidimDataset` (reducesize mode):**
```
MultidimDataset{ReduceFeat{...}}(5 rows × 4 columns)
├─ mode: reducesize
├─ vnames: ["ts1", "ts2", "ts3", "ts4"]
└─ reduce function: my_reduce_func
```

In reducesize mode, the reduce function name is obtained via [`_callable_name`](@ref)
from the first feature's metadata.