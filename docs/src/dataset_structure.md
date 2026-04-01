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
