# DataTreatments.jl

[![Build Status](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PasoStudio73/DataTreatments.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/PasoStudio73/DataTreatments.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PasoStudio73/DataTreatments.jl)

A Julia package for processing datasets containing multidimensional elements through windowing and dimensionality reduction techniques.

## Overview

**DataTreatments.jl** provides tools for working with matrices, or DataFrames where each element is itself a multidimensional object (vectors, matrices, or higher-dimensional arrays). It offers:

- **Windowing functions** for partitioning multidimensional data
- **Dimensionality reduction** using feature extraction together with windowing
- **Tabular transformation** through feature extraction to convert complex multidimensional datasets into flat feature matrices suitable for standard machine learning models
- **Complete reproducibility** by storing all processing parameters and feature metadata

This package is particularly useful when you need to apply traditional ML algorithms that require tabular input to datasets containing structured multidimensional elements like images, spectrograms, or time series segments.

## Installation

```julia
using Pkg
Pkg.add("DataTreatments")
```

## Quick Start

```julia
using DataTreatments

# Create a dataset with multidimensional elements
X = rand(200, 120)  # Example: 200×120 matrix (e.g., spectrogram)
Xmatrix = fill(X, 100, 10)  # 100×10 dataset where each element is a 200×120 matrix

# Define windowing strategy
win = splitwindow(nwindows=4)  # Split into 4 equal windows per dimension

# Compute intervals for the first element
intervals = @evalwindow X win

# Apply multiple statistical features to each window
features = (mean, std, maximum, minimum)
result = aggregate(Xmatrix, intervals; features)

# Result is a 100×10 matrix where each element is reduced to 4×4
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

### Feature Extraction Functions

#### `applyfeat` - Apply Reduction to a Single Array
```julia
X = rand(200, 120)
intervals = @evalwindow X splitwindow(nwindows=4)

# Apply mean to each window
result = applyfeat(X, intervals; reducefunc=mean)
# Returns a 4×4 matrix (4 windows per dimension)
```

#### `reducesize` - Apply to Dataset Elements
```julia
Xmatrix = fill(rand(200, 120), 100, 10)  # Dataset of matrices
intervals = @evalwindow first(Xmatrix) splitwindow(nwindows=3)

# Aggregate each element using reduce feature
result = reducesize(Xmatrix, intervals; reducefunc=mean)
# Each element reduced from 200×120 to a 3×3 matrix per feature
```

#### `aggregate` - Flatten to Tabular Format
```julia
Xmatrix = fill(rand(200, 120), 100, 10)  # 100 samples, 10 variables
intervals = @evalwindow first(Xmatrix) splitwindow(nwindows=4)
features = (mean, std, maximum, minimum)

result = aggregate(Xmatrix, intervals; features)
# Returns 100×640 matrix (10 vars × 4 features × 16 windows)
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

# Use for feature selection
mean_features = filter(fid -> get_feature(fid) == mean, feature_ids)
temp_features = filter(fid -> get_vname(fid) == :temperature, feature_ids)
window1_features = filter(fid -> get_nwin(fid) == 1, feature_ids)
```

### `DataTreatment` - Complete Processing Container

A comprehensive container that stores processed data along with all parameters for full reproducibility:

```julia
using DataFrames, Statistics

# Create dataset
df = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:1000],
    channel2 = [rand(200, 120) for _ in 1:1000],
    channel3 = [rand(200, 120) for _ in 1:1000]
)

# Process with full parameter storage
win = adaptivewindow(nwindows=6, overlap=0.15)
features = (mean, std, maximum, minimum, median)

dt = DataTreatment(df, :reducesize; 
                   win=(win,), 
                   features=features)

# Access processed data
X_flat = get_dataset(dt)        # Flat feature matrix
feature_ids = get_featureid(dt) # Feature metadata

# All parameters are stored for reproducibility
aggrtype = get_aggrtype(dt)     # :reducesize
reduction = get_reducefunc(dt)   # mean (default)
var_names = get_vnames(dt)       # [:channel1, :channel2, :channel3]
feat_funcs = get_features(dt)    # (mean, std, maximum, minimum, median)
n_windows = get_nwindows(dt)     # 6

# Document experiment
println("Processing: $aggrtype mode")
println("Variables: $(join(var_names, ", "))")
println("Features: $(join(nameof.(feat_funcs), ", "))")
println("Windows: $n_windows per dimension")
```

## API Reference

### Windowing Functions
- `splitwindow(; nwindows::Int)` - Equal non-overlapping windows
- `movingwindow(; winsize::Int, winstep::Int)` - Fixed-size sliding windows
- `adaptivewindow(; nwindows::Int, overlap::Float64)` - Windows with overlap
- `wholewindow()` - Single window covering entire dimension

### Processing Functions
- `applyfeat(X, intervals; reducefunc=mean)` - Apply reduction to single array
- `aggregate(X, intervals; features=(mean,))` - Apply features to dataset elements
- `reducesize(X, intervals; features=(mean,))` - Flatten to tabular format

### Data Structures
- `DataTreatment` - Container for processed data with complete metadata
- `FeatureId` - Metadata for individual features (variable, function, window)

### Accessor Functions
- `get_dataset(dt)` - Extract processed feature matrix
- `get_featureid(dt)` - Get feature metadata vector
- `get_reducefunc(dt)` - Get reduction function used
- `get_aggrtype(dt)` - Get processing mode
- `get_vnames(dt)` - Get unique variable names
- `get_features(dt)` - Get unique feature functions
- `get_nwindows(dt)` - Get maximum window number
- `get_vname(fid)` - Get variable name from FeatureId
- `get_feature(fid)` - Get feature function from FeatureId
- `get_nwin(fid)` - Get window number from FeatureId

### Macros
- `@evalwindow(X, winfuncs...)` - Evaluate window functions for array dimensions

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
