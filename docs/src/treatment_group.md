```@meta
CurrentModule = DataTreatments
```

# [TreatmentGroup](@id treatment_group)

A `TreatmentGroup` is a configuration object that selects and describes how a
subset of columns in a dataset should be processed by [`DataTreatment`](@ref).
It stores the selected column indices, their names, the dimensionality filter
used during selection, and — for multidimensional columns — the aggregation
function and optional groupby specification.

The type parameter `T` is the `typejoin` of the data types of all selected
columns (e.g., `Float64` when all columns are numeric, or `Any` when the
selection is empty or mixed).

---

## Struct

```@docs
TreatmentGroup
```

---

## Constructors

### DatasetStructure constructor

```julia
TreatmentGroup(ds_struct::DatasetStructure; kwargs...)
```

The primary constructor. Selects columns from a pre-computed
[`DatasetStructure`](@ref) using the keyword filters described below.

### DataFrame constructor

```julia
TreatmentGroup(df::DataFrame; kwargs...)
```

Convenience constructor that builds a [`DatasetStructure`](@ref) from `df`
internally.

### Matrix constructor

```julia
TreatmentGroup(ds::Matrix, vnames::Vector{String}; kwargs...)
```

Convenience constructor that builds a [`DatasetStructure`](@ref) from a raw
matrix and a vector of column names.

### Curried constructor

```julia
TreatmentGroup(; kwargs...)  # returns a Function
```

Returns a closure `x -> TreatmentGroup(x; kwargs...)` that accepts a
[`DatasetStructure`](@ref). Useful for passing to [`DataTreatment`](@ref)
without having the `DatasetStructure` available yet.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `dims` | `Int` | `-1` | Dimensionality filter. `-1` selects all dimensions; `0` selects scalar columns; `1` selects 1D (e.g., time series); `2` selects 2D (e.g., images). |
| `name_expr` | `Regex`, `Function`, or `Vector{String}` | `r".*"` | Column name filter. A `Regex` is matched against each name; a `Function` is called as a predicate; a `Vector{String}` is treated as an explicit inclusion list. |
| `datatype` | `Type` | `Any` | Filter columns by their element data type. `Any` disables the filter. |
| `aggrfunc` | `Base.Callable` | `aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))` | Aggregation or reduction function for multidimensional columns. See [`aggregate`](@ref) and [`reducesize`](@ref). |
| `groupby` | `Nothing`, `Symbol`, or `Tuple{Vararg{Symbol}}` | `nothing` | Grouping specification for output features from multidimensional processing. A single `Symbol` is automatically wrapped in a tuple. Possible keys include `:vname`, `:window`, or `:feature`. |

---

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `idxs` | `Vector{Int}` | Column indices selected by this group. |
| `dims` | `Int` | Dimensionality filter used during selection (`-1` = all). |
| `vnames` | `Vector{String}` | Names of the selected columns. |
| `aggrfunc` | `Base.Callable` | Aggregation/reduction function for multidimensional columns. |
| `groupby` | `Union{Nothing, Tuple{Vararg{Symbol}}}` | Grouping specification, or `nothing` if not set. |

---

## `Base` Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `length` | `length(tg::TreatmentGroup)` | Number of selected columns. |
| `iterate` | `iterate(tg::TreatmentGroup[, state])` | Iterates over selected column indices. |
| `eachindex` | `eachindex(tg::TreatmentGroup)` | Returns `eachindex(tg.idxs)`. |

---

## Getter Methods

```@docs
get_idxs(tg::TreatmentGroup)
get_idxs(tg::TreatmentGroup, i::Int)
get_dims(tg::TreatmentGroup)
get_vnames(tg::TreatmentGroup)
get_vnames(tg::TreatmentGroup, i::Int)
get_vnames(tg::TreatmentGroup, idxs::Vector{Int})
get_aggrfunc(tg::TreatmentGroup)
get_groupby(tg::TreatmentGroup)
has_groupby(tg::TreatmentGroup)
```

---

## Overlap Resolution

```@docs
get_idxs(tgs::Vector{<:TreatmentGroup})
```

When multiple `TreatmentGroup`s select overlapping columns, the overlap must be
resolved before building output datasets. The `get_idxs` method on a vector of
groups handles this by giving **later groups higher priority**: if column `i`
appears in groups 1 and 3, it is removed from group 1 and kept in group 3.

A warning is emitted when any group is left with zero columns after resolution.

```julia
tg1 = TreatmentGroup(df; dims=0, name_expr=["V1", "V2", "V3"])
tg2 = TreatmentGroup(df; dims=0, name_expr=["V3", "V4", "V5"])

resolved = get_idxs([tg1, tg2])
# resolved[1] contains indices for V1, V2     (V3 removed)
# resolved[2] contains indices for V3, V4, V5 (V3 kept here)
```

---

## Display

### One-line (`show(io::IO, tg)`)

```
TreatmentGroup{Float64}(5 cols, dims=0)
TreatmentGroup{Any}(18 cols, dims=all)
TreatmentGroup{Float64}(0 cols, dims=1)
```

### Multi-line (`show(io::IO, ::MIME"text/plain", tg)`)

**Scalar columns (`dims == 0`):**
```
TreatmentGroup{Float64}(5 columns selected)
├─ dims filter: 0
└─ selected indices: [6, 7, 8, 9, 10]
```

For scalar columns, `aggregation function` and `groupby` are omitted since
they are not applicable.

**Multidimensional columns (`dims != 0`):**
```
TreatmentGroup{Float64}(4 columns selected)
├─ dims filter: 1
├─ selected indices: [11, 12, 13, 14]
├─ aggregation function: Aggregate
└─ groupby: (:vname,)
```

The aggregation function name is extracted from the callable's type name.
Anonymous callables are displayed as `"anonymous callable"`.

---

## See Also

- [`DataTreatment`](@ref) — the pipeline object that consumes `TreatmentGroup`s.
- [`DatasetStructure`](@ref) — pre-computed dataset metadata used for column selection.
- [`aggregate`](@ref) — aggregation function for multidimensional columns.
- [`reducesize`](@ref) — reduction function for multidimensional columns.
- [Output Datasets](@ref output_dataset) — the dataset types produced after processing.