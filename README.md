# DataTreatments.jl

[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PasoStudio73.github.io/DataTreatments.jl/)
[![Build Status](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/PasoStudio73/DataTreatments.jl/branch/main/graph/badge.svg?token=GqP3LgtrND)](https://codecov.io/gh/PasoStudio73/DataTreatments.jl)

A Julia package for processing datasets containing multidimensional elements through windowing and dimensionality reduction techniques.

## Overview

**DataTreatments.jl** provides tools for working with matrices, or DataFrames where each element is itself a multidimensional object (vectors, matrices, or higher-dimensional arrays). It offers:

- **Windowing functions** for partitioning multidimensional data
- **Dimensionality reduction** using feature extraction together with windowing
- **Tabular transformation** through feature extraction to convert complex multidimensional datasets into flat feature matrices suitable for standard machine learning models
- **Data normalization** with multiple methods (z-score, min-max, sigmoid, etc.) for preprocessing
- **Group-wise operations** via `groupby` for consistent processing across related features
- **Complete reproducibility** by storing all processing parameters and feature metadata

This package is particularly useful when you need to apply traditional ML algorithms that require tabular input to datasets containing structured multidimensional elements like images, spectrograms, or time series segments.

## Installation

```julia
using Pkg
Pkg.add("DataTreatments")
```

# Quick Start

## Basic Usage with Matrix
```julia
using DataTreatments

# Create a dataset with multidimensional elements
Xmatrix = [rand(1:100, 4, 2) for _ in 1:10, _ in 1:5]  # 10×5 dataset where each element is a 4×2 matrix
vnames = Symbol.("auto", 1:5)

# Define processing parameters
win = splitwindow(nwindows=2)
features = (mean, std, maximum, minimum)
norm = ZScore
reducefunc = median

# Process for propositional analysis
result = DataTreatment(Xmatrix, :aggregate; vnames, win, features, norm)

# Process for modal analysis
result = DataTreatment(Xmatrix, :reducesize; vnames, win, features, reducefunc, norm)
```

## Basic Usage with DataFrame
```julia
using DataTreatments
using DataFrames

# Create dataset with multidimensional elements
df = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:1000],
    channel2 = [rand(200, 120) for _ in 1:1000],
    channel3 = [rand(200, 120) for _ in 1:1000]
)

# Define processing parameters
win = adaptivewindow(nwindows=6, overlap=0.15)
features = (mean, std, maximum, minimum, median)
norm = PNorm(p=1)
reducefunc = median

# Process for propositional analysis
result = DataTreatment(df, :aggregate; win, features, norm)

# Process for modal analysis
result = DataTreatment(df, :reducesize; win, features, reducefunc, norm)

# Access processed data
X_flat = get_dataset(result)        # Flat feature matrix
feature_ids = get_featureid(result) # Feature metadata
```

## Core Concepts

### Windowing Functions

DataTreatments provides several windowing strategies:

#### `splitwindow` - Equal Non-Overlapping Windows
```julia
win = splitwindow(nwindows=3)
# Divides data into 3 equal, non-overlapping segments
```

#### `movingwindow` - Fixed-Size Sliding Windows
```julia
win = movingwindow(winsize=50, winstep=25)
# Creates overlapping windows of size 50, advancing by 25 points
```

#### `adaptivewindow` - Windows with Controlled Overlap
```julia
win = adaptivewindow(nwindows=5, overlap=0.2)
# Creates 5 windows with 20% overlap between consecutive windows
```

#### `wholewindow` - Single Window (Entire Dimension)
```julia
win = wholewindow()
# Creates a single window covering the entire dimension
```

### Multi-Dimensional Windowing

Use the `@evalwindow` macro to apply window functions to each dimension:

```julia
X = rand(200, 120)

# Apply same windowing to all dimensions
intervals = @evalwindow X splitwindow(nwindows=4)

# Apply different windowing per dimension
intervals = @evalwindow X splitwindow(nwindows=4) movingwindow(winsize=40, winstep=20)
```

## Normalization

`DataTreatments.jl` uses **Normalization.jl** as its normalization backend.

- Core normalization algorithms are provided by `Normalization.jl`.
- `DataTreatments.jl` re-exports `fit`, `fit!`, `normalize`, and `normalize!`.
- `DataTreatments.jl` adds integration for nested/multidimensional dataset elements in `ext/NormalizationExt.jl`.

We thank the maintainers and contributors of **Normalization.jl** for their work and for making this integration possible.

### Grouping Functions

Grouping lets you partition related feature columns (e.g., by variable name, window, or feature) so that operations like normalization are applied with shared coefficients across each group instead of per-column.
This preserves consistent scaling for semantically related parts of the dataset.

```julia
dt = DataFrame([rand(1:100, 4, 2) for _ in 1:10, _ in 1:5], :auto)
win = splitwindow(nwindows=2)

grp1 = [:x1, :x2]

groups = DataTreatments.groupby(dt, grp1)

dt_norm = DataTreatment(dt, :aggregate; win, features, groups=(:vname,), norm=Scale)
```

## Data Structures

### `FeatureId` - Feature Metadata

A metadata container that stores information about each feature column for reproducibility and feature selection:

```julia
# Created automatically by DataTreatment
dt = DataTreatment(df, :reducesize; win=(win,), features=(mean, std, maximum))

# Access feature metadata
feature_ids = get_featureid(dt)

# Each FeatureId contains:
# - vname: Source variable name
# - feat: Feature function applied
# - nwin: Window number
```

### `DataTreatment` - Complete Processing Container

A comprehensive container that stores processed data along with all parameters for full reproducibility:

```julia
# Create dataset
df = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:1000],
    channel2 = [rand(200, 120) for _ in 1:1000],
    channel3 = [rand(200, 120) for _ in 1:1000]
)

# Process with full parameter storage
win = adaptivewindow(nwindows=6, overlap=0.15)
features = (mean, std, maximum, minimum, median)

dt = DataTreatment(df, :aggregate; win, features, norm=MinMax)

# Access processed data
X_flat = get_dataset(dt)        # Flat feature matrix
feature_ids = get_featureid(dt) # Feature metadata

# All parameters are stored for reproducibility
aggrtype = get_aggrtype(dt)     # :aggregate
reduction = get_reducefunc(dt)   # mean (default)
var_names = get_vnames(dt)       # [:channel1, :channel2, :channel3]
feat_funcs = get_features(dt)    # (mean, std, maximum, minimum, median)
n_windows = get_nwindows(dt)     # 36
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
