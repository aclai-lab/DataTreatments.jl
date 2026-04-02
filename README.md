<div align="center">
    <img src="banner.png" alt="DataTreatments" width="900">
</div>

[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PasoStudio73.github.io/DataTreatments.jl/)
[![Build Status](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/PasoStudio73/DataTreatments.jl/branch/main/graph/badge.svg?token=GqP3LgtrND)](https://codecov.io/gh/PasoStudio73/DataTreatments.jl)

## Overview

**DataTreatments.jl** provides tools for working with heterogeneous datasets that may contain mixed-type columns — discrete (categorical/symbolic), continuous (scalar numeric), and multidimensional elements (vectors, matrices, or higher-dimensional arrays). It offers:

- **Heterogeneous data handling** — seamlessly manages datasets mixing discrete, continuous, and multidimensional columns in a unified container
- **NaN and missing value support** — detects, tracks, and handles `NaN` and `missing` values across all column types
- **Imputation** via [Impute.jl](https://github.com/invenia/Impute.jl) for filling missing or NaN values with configurable strategies
- **Normalization** via [Normalization.jl](https://github.com/PasoStudio73/Normalization.jl) for scaling and standardizing continuous and multidimensional data
- **Windowing functions** for partitioning multidimensional data
- **Dimensionality reduction** using feature extraction together with windowing
- **Tabular transformation** through feature extraction to convert complex multidimensional datasets into flat feature matrices suitable for standard machine learning models
- **Group-wise operations** via `groupby` for consistent processing across related features
- **Lazy processing** — `DataTreatment` stores only raw data and metadata; all transformations happen on demand through `get_dataset`
- **Complete reproducibility** by storing all processing parameters and feature metadata

This package is particularly useful when you need to preprocess, clean, and transform heterogeneous real-world datasets — including those with missing data and mixed types — before feeding them into traditional ML algorithms that require tabular input or into modal learning pipelines that operate on structured multidimensional elements such as images, spectrograms, or time series segments.

## Installation

```julia
using Pkg
Pkg.add("DataTreatments")
```

## Quick Start

### 1. Load a dataset with `load_dataset`

`load_dataset` is the main entry point of DataTreatments.jl. It loads the input data, inspects and organizes all columns by type and dimensionality, applies user-defined `TreatmentGroup` directives, and returns a `DataTreatment` object containing all the information needed for further analysis and transformations.

```julia
using DataTreatments, DataFrames, Statistics, CategoricalArrays

df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    int_col  = Int[10, 20, 30, 40, 50],
    V1       = [NaN, missing, 3.0, 4.0, 5.6],
    V2       = [2.5, missing, 4.5, 5.5, NaN],
    ts1      = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2      = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
    img1     = [rand(6, 6) for _ in 1:5],
    img2     = [rand(6, 6) for _ in 1:5],
)

dt = load_dataset(df)
```

**`load_dataset` accepts the following parameters:**

| Parameter | Description |
|---|---|
| `data` | Input `DataFrame` or `Matrix` |
| `vnames` | Column names (auto-generated as `"V1"`, `"V2"`, ... if omitted) |
| `target` | Optional target vector (labels), automatically encoded |
| `treatments...` | Zero or more `TreatmentGroup` directives specifying which columns to select and how to process them |
| `treatment_ds` | Include datasets produced by treatment groups (default: `true`) |
| `leftover_ds` | Include columns not matched by any treatment group (default: `false`) |
| `float_type` | Floating-point precision used throughout (default: `Float64`) |

### 2. Access data from a `DataTreatment`

All methods operate on the `DataTreatment` object returned by `load_dataset`:

```julia
# Returns the tabular part (discrete + continuous + aggregated multidim)
data, vnames = get_tabular(dt)

# Returns the multidimensional part (reduced arrays)
data, vnames = get_multidim(dt)

# Access individual column types
data, vnames = get_discrete(dt)
data, vnames = get_continuous(dt)
data, vnames = get_aggregated(dt)
data, vnames = get_reduced(dt)

# Access the target vector
target = get_target(dt)
```

### 3. Custom treatment groups

`TreatmentGroup` is a user directive passed to `load_dataset` to control which columns are selected and how they are processed:

```julia
dt = load_dataset(
    df,
    TreatmentGroup(
        dims=1,
        aggrfunc=aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )
    )
)
```

### 4. Multiple treatment groups

Different directives can be applied to different column subsets:

```julia
dt = load_dataset(
    df,
    TreatmentGroup(
        dims=1,
        aggrfunc=aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )
    ),
    TreatmentGroup(
        dims=2,
        aggrfunc=reducesize(
            reducefunc=minimum,
            win=(splitwindow(nwindows=3),)
        )
    )
)
```

### 5. Filter columns by name

Use `name_expr` to select columns matching a regex pattern:

```julia
dt = load_dataset(
    df,
    TreatmentGroup(name_expr=r"^(V|i)"),
    leftover_ds=false,
)
```

## Core Concepts

### Windowing Functions

DataTreatments provides several windowing strategies for partitioning multidimensional data:

| Function | Description |
|---|---|
| `splitwindow(nwindows=3)` | Equal non-overlapping windows |
| `movingwindow(winsize=50, winstep=25)` | Fixed-size sliding windows |
| `adaptivewindow(nwindows=5, overlap=0.2)` | Windows with controlled overlap |
| `wholewindow()` | Single window covering the entire dimension |

Use the `@evalwindow` macro to apply window functions to each dimension of an array:

```julia
X = rand(200, 120)

# Same windowing on all dimensions
intervals = @evalwindow X splitwindow(nwindows=4)

# Different windowing per dimension
intervals = @evalwindow X splitwindow(nwindows=4) movingwindow(winsize=40, winstep=20)
```

### Processing Modes

#### `aggregate` — Tabular Feature Extraction

Flattens multidimensional elements into scalar columns by applying feature functions to each window. Produces a flat feature matrix suitable for standard ML.

```julia
aggrfunc = aggregate(
    features=(mean, std, maximum, minimum),
    win=(splitwindow(nwindows=3),)
)
```

#### `reducesize` — Dimensionality Reduction

Reduces the size of each multidimensional element while preserving its array structure. Useful for modal analysis and downstream tasks that expect array-valued data.

```julia
aggrfunc = reducesize(
    reducefunc=median,
    win=(adaptivewindow(nwindows=5, overlap=0.2),)
)
```

### TreatmentGroup

A `TreatmentGroup` is a user directive passed to `load_dataset` that specifies **which columns to select** and **how to process** them. Multiple `TreatmentGroup`s can be passed to `load_dataset`, each producing its own set of datasets inside the resulting `DataTreatment` object.

#### Column Selection

Columns are included when **all** active filters match:

| Parameter | Description |
|---|---|
| `dims` | Keep only columns whose array dimensionality equals `dims` (`-1` disables this filter) |
| `name_expr` | Keep columns whose name matches a `Regex`, a predicate function, or a `Vector{String}` |
| `datatype` | `:discrete`, `:continuous`, `:multidim`, or `:all` (default) |

#### Processing Directives

| Parameter | Description |
|---|---|
| `aggrfunc` | Processing function for multidimensional columns: `aggregate(...)` → scalar matrix; `reducesize(...)` → smaller arrays |
| `grouped` | `false` (default): process each column independently; `true`: process all selected columns jointly |
| `groupby` | Partition output features by `:vname`, window index, or feature type |
| `impute` | Imputation strategy (from [Impute.jl](https://github.com/invenia/Impute.jl)) applied after encoding/aggregation |
| `norm` | Normalization strategy (from [Normalization.jl](https://github.com/PasoStudio73/Normalization.jl)) applied to the output matrix |

### Access data from a `DataTreatment`

All methods operate on the `DataTreatment` object returned by `load_dataset`.

#### Composite accessors

These methods combine multiple dataset types into a single matrix for convenience:

| Method | Returns |
|---|---|
| `get_tabular(dt)` | `(Matrix, vnames)` — merges discrete, continuous, and aggregated columns into a single flat matrix |
| `get_multidim(dt)` | `(Matrix{Array}, vnames)` — collects all reduced multidimensional columns |

```julia
# All tabular data in one matrix (discrete + continuous + aggregated)
data, vnames = get_tabular(dt)

# All multidimensional data (array-valued columns)
data, vnames = get_multidim(dt)
```

#### Type-specific accessors

These methods return a single category of columns:

| Method | Returns |
|---|---|
| `get_discrete(dt)` | `(Matrix, vnames)` — categorical/symbolic columns, encoded as integers |
| `get_continuous(dt)` | `(Matrix{T}, vnames)` — scalar numeric columns |
| `get_aggregated(dt)` | `(Matrix{T}, vnames)` — multidimensional columns flattened into scalar features via `aggregate` |
| `get_reduced(dt)` | `(Matrix{Array{T}}, vnames)` — multidimensional columns downsampled via `reducesize` |
| `get_target(dt)` | `AbstractVector` — the encoded target vector |

```julia
data, vnames = get_discrete(dt)
data, vnames = get_continuous(dt)
data, vnames = get_aggregated(dt)
data, vnames = get_reduced(dt)

target = get_target(dt)
```

## Use Cases

- **Audio Processing**: Extract features from spectrograms for audio classification
- **Image Analysis**: Process image patches for computer vision tasks
- **Time Series**: Analyze segmented multivariate time series
- **Signal Processing**: Extract statistical features from signal windows
- **Medical Data**: Process multi-channel physiological signals
- **Experiment Reproducibility**: All parameters stored for exact replication

## License

MIT License

## About

Developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.