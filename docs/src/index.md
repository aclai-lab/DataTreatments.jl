```@meta
CurrentModule = DataTreatments
```
# DataTreatments.jl

## Overview

**DataTreatments.jl** is a data preparation tool for machine learning
experiments. Its goal is to support the widest possible range of
dataset types — from plain tabular data to mixed datasets containing
images, audio, and multivariate time series — with minimal boilerplate.

One of the most time-consuming and error-prone steps in any ML project
is preparing the data. In particular:

- detecting and handling `missing` and `NaN` values
- verifying and correcting class imbalance, especially in multimedia
  datasets
- ensuring that multimedia elements (e.g. images, audio clips) have
  consistent dimensions suitable for the experiment
- detecting mixed-type datasets (e.g. tabular + images + audio) and
  routing each type to the right processing pipeline
- normalizing data when values are on incompatible scales

**DataTreatments.jl takes care of all of this:**

- **Missing and NaN handling** via
  [Impute.jl](https://github.com/invenia/Impute.jl) — not only at
  the tabular level, but also *inside* multimedia elements (vectors,
  matrices).
- **Class imbalance correction** via
  [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl) —
  supports multimedia datasets where elements may have different
  sizes.
- **Windowing and dimensionality reduction** — when multimedia
  elements are too large, DataTreatments applies windowing to reduce
  their size. Data can be n-dimensional.
- **Type-aware partitioning** — the dataset is split by data type
  (discrete tabular, continuous tabular, 1-D vectors, n-D matrices),
  and each partition is accessed safely through dedicated getters.
- **Normalization** via
  [Normalization.jl](https://github.com/PasoStudio73/Normalization.jl).
- **Multimedia-to-tabular conversion** — multimedia datasets can be
  aggregated into flat feature matrices (via windowed feature
  extraction), making them compatible with traditional ML algorithms.

The primary input format is a `DataFrame`. An optional categorical
target vector can be provided for classification experiments.

---

## Installation

```julia
using Pkg
Pkg.add("DataTreatments")
```

---

## Quick Start

### 1. Build a test dataset

```julia
using DataTreatments, DataFrames, CategoricalArrays, Statistics

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    str_col = [missing, "blue", "green", "red", "blue"],
    sym_col = [:circle, :square, :triangle, :square, missing],
    cat_col = categorical(["small", "medium", missing, "small", "large"]),
    int_col = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    ts1 = [missing, collect(2.0:7.0), missing,
        collect(4.0:9.0), collect(5.0:10.0)],
    ts2  = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5),
        collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), missing],
    img1 = [create_image(i) for i in 1:5],
    img2 = [create_image(i + 10) for i in 1:5],
)

target = ["classA", "classB", "classC", "classA", "classB"]
```

### 2. Load with no treatment (inspection only)

```julia
dt = load_dataset(df)
```

### 3. Select scalar columns and normalize

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(dims=0, datatype=:continuous, norm=MinMax),
)

X, names = get_continuous(dt)
```

### 4. Select discrete and continuous separately

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(dims=0, datatype=:discrete),
    TreatmentGroup(dims=0, datatype=:continuous),
)

X_cat, cat_names = get_discrete(dt)
X_num, num_names = get_continuous(dt)
```

### 5. Impute missing values in scalar columns

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(
        dims=0,
        datatype=:continuous,
        impute=(Interpolate(), LOCF(), NOCB()),
    ),
)
```

### 6. Aggregate time series into a tabular matrix

Extract scalar features from windowed 1-D time series and produce a
flat feature matrix compatible with any standard ML algorithm:

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(
        dims=1,
        aggrfunc=DataTreatments.aggregate(
            features=(mean, maximum),
            win=(splitwindow(nwindows=2),),
        ),
        norm=MinMax,
    ),
)

X, names = get_tabular(dt)   # flat Float64 matrix
```

### 7. Downsample images (2-D arrays)

Reduce the spatial resolution of images while preserving their
array structure:

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(
        dims=2,
        aggrfunc=reducesize(
            reducefunc=mean,
            win=(splitwindow(nwindows=2),),
        ),
    ),
)

X, names = get_multidim(dt)  # Matrix{Matrix{Float64}}
```

### 8. Mixed pipeline: scalars + time series + images

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(
        dims=0,
        datatype=:continuous,
        impute=(LOCF(), NOCB()),
        norm=ZScore,
    ),
    TreatmentGroup(
        dims=1,
        aggrfunc=DataTreatments.aggregate(
            features=(mean, std),
            win=(splitwindow(nwindows=3),),
        ),
        norm=MinMax,
    ),
    TreatmentGroup(
        dims=2,
        aggrfunc=reducesize(
            reducefunc=mean,
            win=(splitwindow(nwindows=2),),
        ),
    ),
)

X_tab, tab_names = get_tabular(dt)   # scalars + aggregated ts
X_img, img_names = get_multidim(dt)  # downsampled images
y = get_target(dt)
```

### 9. Filter columns by name

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(name_expr=r"^(V|ts)"),
)
```

### 10. Class balancing

> [!WARNING]
> **Impute before balancing!**
>
> If you intend to use the `balance` parameter, you **must** ensure
> that no `missing` or `NaN` values are present in your data, because
> [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl) **cannot
> handle missing values** and will error at runtime.
>
> It is therefore strongly recommended to always pair `balance` with
> an `impute` directive in your `TreatmentGroup`.
> **DataTreatments.jl guarantees safe execution order: imputation is
> always applied first, and only then is the dataset rebalanced.**

```julia
dt = load_dataset(
    df, target,
    TreatmentGroup(
        dims=0,
        datatype=:continuous,
        impute=(Interpolate(), LOCF(), NOCB()),
    ),
    balance=(SMOTE(k=3), TomekUndersampler()),
)
```

---

## Core Concepts

### `load_dataset` parameters

| Parameter | Description |
|:----------|:------------|
| `data` | Input `DataFrame` or `Matrix` |
| `vnames` | Column names (auto-generated if omitted) |
| `target` | Optional target vector; categorical labels are encoded |
| `treatments...` | One or more `TreatmentGroup` directives |
| `float_type` | Floating-point precision (default: `Float64`) |
| `balance` | `AbstractBalance` or tuple of them (default: `nothing`) |
| `treatment_ds` | Include treatment-matched datasets (default: `true`) |
| `leftover_ds` | Include unmatched columns (default: `false`) |

---

### Windowing Functions

| Function | Description |
|:---------|:------------|
| `splitwindow(nwindows=3)` | Equal non-overlapping windows |
| `movingwindow(winsize=50, winstep=25)` | Fixed-size sliding windows |
| `adaptivewindow(nwindows=5, overlap=0.2)` | Overlapping windows |
| `wholewindow()` | Single window over the whole element |

```julia
X = rand(200, 120)

# same windowing on all dimensions
intervals = @evalwindow X splitwindow(nwindows=4)

# different windowing per dimension
intervals = @evalwindow X splitwindow(nwindows=4) \
    movingwindow(winsize=40, winstep=20)
```

---

### Processing Modes

#### `aggregate` — feature extraction → tabular output

Applies feature functions to each window and flattens the result
into a scalar column. Use this when you want to feed time series
or images to a standard ML model.

```julia
aggrfunc = DataTreatments.aggregate(
    features=(mean, std, maximum, minimum),
    win=(splitwindow(nwindows=3),),
)
```

#### `reducesize` — dimensionality reduction → array output

Shrinks each element while preserving its array structure. Use
this for modal analysis or when downstream models expect
array-valued inputs.

```julia
aggrfunc = reducesize(
    reducefunc=median,
    win=(adaptivewindow(nwindows=5, overlap=0.2),),
)
```

---

### `TreatmentGroup` parameters

#### Column selection

| Parameter | Description |
|:----------|:------------|
| `dims` | Keep columns whose array length equals `dims` (`-1` = any) |
| `name_expr` | `Regex`, predicate, or `Vector{String}` to match names |
| `datatype` | `:discrete`, `:continuous`, `:multidim`, or `:all` |

#### Processing directives

| Parameter | Description |
|:----------|:------------|
| `aggrfunc` | `aggregate(...)` or `reducesize(...)` for multidim cols |
| `impute` | Tuple of `Impute.jl` imputors applied in order |
| `norm` | Normalization type from `Normalization.jl` |
| `groupby` | Partition output by `:vname`, window index, or feature |
| `grouped` | Process all selected columns jointly (`true`/`false`) |

---

### Accessors

| Method | Returns |
|:-------|:--------|
| `get_tabular(dt)` | merged discrete + continuous + aggregated matrix |
| `get_multidim(dt)` | matrix of downsampled arrays |
| `get_discrete(dt)` | integer-coded categorical matrix |
| `get_continuous(dt)` | float scalar matrix |
| `get_aggregated(dt)` | float matrix from `aggregate` |
| `get_reduced(dt)` | array-valued matrix from `reducesize` |
| `get_target(dt)` | encoded target vector |
| `get_treats(dt)` | list of applied `TreatmentGroup`s |
| `get_balance(dt)` | balancing strategy or `nothing` |

---

## Use Cases

- **Audio**: extract features from spectrograms for classification
- **Images**: preprocess image patches for computer vision
- **Time Series**: segment and aggregate multivariate signals
- **Medical**: process multi-channel physiological recordings
- **Mixed datasets**: route each column type to the right pipeline
- **Reproducibility**: all parameters and metadata are stored in
  the `DataTreatment` object for exact replication

---

## License

MIT License

---

## About

Developed by the [ACLAI Lab](https://aclai.unife.it/en/) @
University of Ferrara.