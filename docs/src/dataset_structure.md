```@meta
CurrentModule = DataTreatments
```

# Data Structures

This page describes the complete type hierarchy used to represent processed datasets
in `DataTreatments.jl`, from low-level metadata features up to the top-level
`DataTreatment` container.

---

## Type Hierarchy Overview

```
AbstractDataFeature
├── DiscreteFeat{T}       # metadata for a categorical column
├── ContinuousFeat{T}     # metadata for a numeric scalar column
├── AggregateFeat{T}      # metadata for a time-series → scalar aggregation
└── ReduceFeat{T}         # metadata for a time-series → smaller array reduction

AbstractDataset
├── DiscreteDataset{T}                    # integer-coded categorical matrix
├── ContinuousDataset{T}                  # float matrix
└── MultidimDataset{T, S}
     ├── S = AggregateFeat  →  tabular scalar output
     └── S = ReduceFeat     →  reduced array output

DataTreatment{T}          # top-level container (produced by load_dataset)
├── data    ::Vector{AbstractDataset}
├── target  ::AbstractVector
└── treats  ::Vector{TreatmentGroup}
```

---

## Full Pipeline

```
Raw DataFrame / Matrix
        │
        ▼  load_dataset(data, vnames, target, treatments...; float_type=Float64)
        │
        ├─ _inspecting(data)  →  datastruct::NamedTuple
        │   (inspect all columns: types, missing, NaN, dims, ...)
        │
        ├─ encode target  →  CategoricalVector
        │
        ├─ for each TreatmentGroup:
        │   ├── classify columns  →  discrete_ids, continuous_ids, multidim_ids
        │   │
        │   ├── DiscreteDataset(discrete_ids, ...)
        │   │    └── _discrete_encode  →  integer-coded matrix
        │   │
        │   ├── ContinuousDataset(continuous_ids, ...)
        │   │    └── cast to float_type  →  float matrix
        │   │
        │   └── MultidimDataset(multidim_ids, ..., aggrfunc)
        │        ├── aggrfunc = aggregate(...)   →  scalar columns (AggregateFeat)
        │        └── aggrfunc = reducesize(...)  →  smaller arrays (ReduceFeat)
        │
        └─▶  DataTreatment{T}(ds, target, treats)
```

---

## Treatment Directives

### `TreatmentGroup`

```@docs
TreatmentGroup
```

#### Column Selection Filters

```
All columns in the dataset
        │
        ├─ dims filter        keep columns with array length == dims
        │                     (skip if dims == -1)
        │
        ├─ datatype filter    :discrete  → !(T <: Float) && !(T <: AbstractArray)
        │                     :continuous → T <: Float
        │                     :multidim  → T <: AbstractArray
        │                     :all       → any type  (skip filter)
        │
        └─ name_expr filter   Regex      → match(name_expr, name) !== nothing
                              Callable   → name_expr(name)
                              Vector     → name ∈ name_expr
                              (skip if name_expr == r".*")
                                    │
                                    ▼
                            ids ::Vector{Int}
```

#### From Directive to Output Datasets

```
TreatmentGroup
│   ids = [1, 3, 5, 7, 8]   ← column indices surviving all filters
│
├─ discrete ids   → [1, 3]
│       └─▶ DiscreteDataset{T}
│            ├── data :: Matrix{Union{Int,Missing}}  (integer-coded)
│            └── info :: Vector{DiscreteFeat}
│
├─ continuous ids → [5]
│       └─▶ ContinuousDataset{T}
│            ├── data :: Matrix{Union{Missing,T}}
│            └── info :: Vector{ContinuousFeat}
│
└─ multidim ids  → [7, 8]
        │
        ├─ aggrfunc = aggregate(win=slidingwindows(2), features=(mean, std))
        │       └─▶ MultidimDataset{T, AggregateFeat}
        │            ├── data :: Matrix{T}              (nrows × n_feats×nwins)
        │            ├── info :: Vector{AggregateFeat}
        │            └── groups :: Vector{Vector{Int}}  (if groupby != nothing)
        │
        └─ aggrfunc = reducesize(256)
                └─▶ MultidimDataset{T, ReduceFeat}
                     ├── data :: Matrix{AbstractArray{T}}
                     ├── info :: Vector{ReduceFeat}
                     └── groups :: nothing
```

#### Multiple `TreatmentGroup`s

```
load_dataset(df, target, t1, t2, t3)
│
├─ t1 = TreatmentGroup(datatype=:continuous, norm=MinMaxNormalization)
│        └─▶ ContinuousDataset  (all continuous columns, normalized)
│
├─ t2 = TreatmentGroup(name_expr=r"^signal_", aggrfunc=aggregate(...), groupby=:vname)
│        └─▶ MultidimDataset{T, AggregateFeat}  (signal_* columns → scalar features)
│
└─ t3 = TreatmentGroup(name_expr=r"^audio_", aggrfunc=reducesize(256))
         └─▶ MultidimDataset{T, ReduceFeat}     (audio_* columns → downsampled arrays)

DataTreatment.data = [
    ContinuousDataset,                   # from t1
    MultidimDataset{AggregateFeat},      # from t2
    MultidimDataset{ReduceFeat}          # from t3
]
```

---

## Metadata Feature Structs

These structs carry per-column diagnostic information (valid/missing/NaN indices,
source column id, etc.). One is created for each output column.

### `DiscreteFeat{T}`

```@docs
DiscreteFeat{T}
```

---

### `ContinuousFeat{T}`

```@docs
ContinuousFeat{T}
```

---

### `AggregateFeat{T}`

```@docs
AggregateFeat{T}
```

---

### `ReduceFeat{T}`

```@docs
ReduceFeat{T}
```

---

## Dataset Structs

Each dataset wraps a processed data matrix together with a `Vector` of the
corresponding metadata feature structs.

### `DiscreteDataset{T}`

```@docs
DiscreteDataset{T}
```

---

### `ContinuousDataset{T}`

```@docs
ContinuousDataset{T}
```

---

### `MultidimDataset{T,S}`

```@docs
MultidimDataset{T,S}
```

---

## Top-level Container: `DataTreatment{T}`

```@docs
DataTreatment{T}
```


