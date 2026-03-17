<div align="center">
    <img src="logo.png" alt="DataTreatments" width="600">
</div>

# DataTreatments.jl

[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PasoStudio73.github.io/DataTreatments.jl/)
[![Build Status](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/PasoStudio73/DataTreatments.jl/branch/main/graph/badge.svg?token=GqP3LgtrND)](https://codecov.io/gh/PasoStudio73/DataTreatments.jl)

A Julia package for processing datasets containing multidimensional elements through windowing and dimensionality reduction techniques.

## Overview

**DataTreatments.jl** provides tools for working with matrices or DataFrames where each element is itself a multidimensional object (vectors, matrices, or higher-dimensional arrays). It offers:

- **Windowing functions** for partitioning multidimensional data
- **Dimensionality reduction** using feature extraction together with windowing
- **Tabular transformation** through feature extraction to convert complex multidimensional datasets into flat feature matrices suitable for standard machine learning models
- **Group-wise operations** via `groupby` for consistent processing across related features
- **Lazy processing** — `DataTreatment` stores only raw data and metadata; all transformations happen on demand through `get_dataset`
- **Complete reproducibility** by storing all processing parameters and feature metadata

This package is particularly useful when you need to apply traditional ML algorithms that require tabular input to datasets containing structured multidimensional elements like images, spectrograms, or time series segments.

## Installation

```julia
using Pkg
Pkg.add("DataTreatments")
```

## Quick Start

### 1. Create a `DataTreatment` container

`DataTreatment` is a lightweight container that stores the raw dataset and its metadata. No processing happens at construction time.

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

dt = DataTreatment(df)
```

### 2. Extract processed datasets with `get_dataset`

All transformations are applied lazily when you call `get_dataset`. You pass one or more `TreatmentGroup` directives to control how columns are filtered, windowed, and aggregated.

```julia
# Default: aggregate all columns with max, min, mean over a whole window
result = get_dataset(dt)

# Return as matrices
result = get_dataset(dt, matrix=true)

# Return as DataFrames
result = get_dataset(dt, dataframe=true)
```

### 3. Custom treatment groups

Use `TreatmentGroup` to specify which columns to process and how:

```julia
# Aggregate 1D columns with custom features and windowing
result = get_dataset(
    dt,
    TreatmentGroup(
        dims=1,
        aggrfunc=aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )
    ),
    dataframe=true
)
```

### 4. Multiple treatment groups

Apply different processing to different dimensionalities:

```julia
result = get_dataset(
    dt,
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
    ),
    dataframe=true
)
```

### 5. Filter columns by name

Use `name_expr` to select columns matching a regex pattern. Set `leftover_ds=false` to exclude unmatched columns:

```julia
result = get_dataset(
    dt,
    TreatmentGroup(name_expr=r"^(V|i)"),
    leftover_ds=false,
    dataframe=true
)
```

### 6. Groupby and split

Group output columns by metadata (e.g., variable name, feature function) and optionally split the result:

```julia
result = get_dataset(
    dt,
    TreatmentGroup(
        dims=2,
        aggrfunc=aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        ),
        groupby=:vname,
    ),
    groupby_split=true,
    dataframe=true
)
```

## Core Concepts

### Lazy Architecture

`DataTreatment` is intentionally a **passive container**:

- **Construction** (`DataTreatment(df)`) only stores the raw data matrix and computes lightweight metadata (column types, dimensions, missing/NaN indices).
- **Processing** happens entirely inside `get_dataset`, which accepts `TreatmentGroup` directives specifying how to filter, window, aggregate, or reduce each subset of columns.
- This design maximizes flexibility — you can extract different views of the same dataset without rebuilding the container.

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

A `TreatmentGroup` specifies which columns to select and how to process them:

| Parameter | Description |
|---|---|
| `dims` | Filter columns by dimensionality (`0` = scalar, `1` = vector, `2` = matrix, etc.) |
| `name_expr` | Filter columns by regex on column name |
| `aggrfunc` | Processing function (`aggregate(...)` or `reducesize(...)`) |
| `groupby` | Group output columns by metadata (`:vname`, `:feat`, `:nwin`, or a tuple) |

### `get_dataset` Options

| Keyword | Default | Description |
|---|---|---|
| `treatment_ds` | `true` | Include datasets defined by treatment groups |
| `leftover_ds` | `true` | Include columns not assigned to any treatment group |
| `groupby_split` | `false` | Split multidimensional datasets by group |
| `matrix` | `false` | Return results as matrices |
| `dataframe` | `false` | Return results as DataFrames |

### Output Dataset Types

`get_dataset` returns a `Vector{AbstractDataset}` containing:

- **`DiscreteDataset`** — Categorical/discrete columns, integer-encoded
- **`ContinuousDataset`** — Scalar numeric columns
- **`MultidimDataset`** — Array-valued columns, processed via `aggregate` or `reducesize`

Each dataset carries rich metadata (`DiscreteFeat`, `ContinuousFeat`, `AggregateFeat`, `ReduceFeat`) for full traceability.

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