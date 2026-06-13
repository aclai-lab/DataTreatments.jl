```@meta
CurrentModule = DataTreatments
```

# [Working with DataTreatment](@id datatreatment)

After calling [`load_dataset`](@ref), all processed data lives inside
a [`DataTreatment`](@ref) container. This page explains how to extract
data from it, check its structure, and clean it before passing it to
a machine learning model.

---

## What is inside a `DataTreatment`?

```
DataTreatment{T}
├── data    ::Vector{AbstractDataset}   # one entry per TreatmentGroup
├── target  ::AbstractVector            # encoded labels
├── treats  ::Vector{TreatmentGroup}    # directives that built data
└── balance ::Union{Nothing, ...}       # balancing strategy or nothing
```

The `data` field is a heterogeneous list: each entry is one of
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref), or
[`MultidimDataset`](@ref). The getters below let you extract each
kind without having to filter manually.

---

## Step 1 — Check what you have

Before extracting data, it is useful to know what kinds of datasets
are present. This is especially important when a pipeline may receive
different inputs depending on the dataset.

```@docs
is_tabular
is_multidim
has_tabular
has_multidim
```

### When to use each predicate

| Predicate | Use it when you need to… |
|:----------|:-------------------------|
| `is_tabular(dt)` | assert that **all** output is tabular before |
| | passing to a tabular-only model |
| `is_multidim(dt)` | assert that **all** output is array-valued |
| | before passing to a sequence model |
| `has_tabular(dt)` | check whether any scalar features exist |
| | (mixed pipelines) |
| `has_multidim(dt)` | check whether any time-series features exist |
| | (mixed pipelines) |

```julia
dt = load_dataset(df, y, t1, t2)

if is_tabular(dt)
    X, names = get_tabular(dt)
    # feed to a decision tree, SVM, …
elseif has_multidim(dt)
    X, names = get_multidim(dt)
    # feed to a 1-D CNN, rocket, …
end
```

---

## Step 2 — Extract data

### All-in-one getters

These are the most commonly used getters. They merge all matching
sub-datasets into a single matrix.

```@docs
get_tabular
get_multidim
```

**`get_tabular`** — use this when you want a single flat feature
matrix ready for any tabular model. It merges discrete (integer-
coded), continuous (float), and aggregated time-series columns in
that order.

```julia
X, colnames = get_tabular(dt)
# X::Matrix, colnames::Vector{String}
```

**`get_multidim`** — use this when your model expects one array
per sample per channel (e.g. a recurrent network or a kernel
method for time series). It returns a matrix of vectors.

```julia
X, colnames = get_multidim(dt)
# X::Matrix{Vector{Float64}}
```

---

### Fine-grained getters

Use these when you need to access one specific kind of data, for
example to inspect it separately or feed it to different model
heads.

```@docs
get_discrete
get_continuous
get_aggregated
get_reduced
```

| Getter | Returns | Typical use |
|:-------|:--------|:------------|
| `get_discrete` | integer-coded categorical matrix | embed / one-hot |
| `get_continuous` | float scalar matrix | feed directly |
| `get_aggregated` | float scalar matrix from time series | feed directly |
| `get_reduced` | matrix of smaller arrays | further processing |

```julia
# inspect categorical columns separately
X_cat, cat_names = get_discrete(dt)

# inspect scalar columns
X_num, num_names = get_continuous(dt)

# inspect time-series features extracted by aggregate(...)
X_agg, agg_names = get_aggregated(dt)

# raw downsampled series from reducesize(...)
X_ts, ts_names = get_reduced(dt)
```

---

### Target and metadata getters

```julia
y       = get_target(dt)    # encoded label vector
treats  = get_treats(dt)    # Vector{TreatmentGroup}
balance = get_balance(dt)   # AbstractBalance or nothing
```

---

## Step 3 — Clean missing values

Real datasets often contain missing values even after imputation.
[`filter_missing`](@ref) lets you drop rows or columns that exceed
a given missing-value fraction **before** feeding data to a model
that cannot handle missings.

```@docs
filter_missing
```

### Row-wise vs column-wise

**Column-wise** (`dims=2`, default) — drop features that are mostly
empty. Use this when some sensors or variables are too unreliable.

```julia
# drop any column with more than 30% missing/NaN
dt2 = filter_missing(dt, 0.3)
```

**Row-wise** (`dims=1`) — drop samples that are mostly empty. Use
this when some observations are too incomplete to be useful.
The **same** keep-mask is applied to every sub-dataset and to the
target vector, so row alignment is always preserved.

```julia
# drop any sample with more than 10% missing/NaN across all cols
dt2 = filter_missing(dt, 0.1; dims=1)
```

!!! note "NaN handling"
    By default `NaN` values count as missing. Pass
    `include_nans=false` to count only true `missing` entries.

```julia
# count only missing, ignore NaN
dt2 = filter_missing(dt, 0.2; include_nans=false)
```

---

## Typical full workflow

```julia
using DataTreatments, Normalization, Impute

# 1. build treatment directives
t_num = TreatmentGroup(
    datatype = :continuous,
    impute   = (LOCF(), NOCB()),
    norm     = ZScore,
)

t_ts = TreatmentGroup(
    name_expr = r"^sensor_",
    datatype  = :multidim,
    aggrfunc  = aggregate(
        win      = slidingwindows(3),
        features = (mean, std),
    ),
    norm = MinMax,
)

# 2. build DataTreatment
dt = load_dataset(df, y, t_num, t_ts;
    balance = SMOTE(k=5))

# 3. drop low-quality columns
dt = filter_missing(dt, 0.2)

# 4. extract features for a tabular model
X, names = get_tabular(dt)
y        = get_target(dt)
```

---

## API reference

```@docs
DataTreatment
get_tabular
get_multidim
get_discrete
get_continuous
get_aggregated
get_reduced
is_tabular
is_multidim
has_tabular
has_multidim
filter_missing
```