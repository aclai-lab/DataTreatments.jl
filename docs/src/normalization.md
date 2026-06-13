```@meta
CurrentModule = DataTreatments
```

# [Normalization](@id normalization)

DataTreatments.jl wraps and extends
[Normalization.jl](https://github.com/JuliaAI/Normalization.jl)
to support both scalar and **multidimensional** data (e.g. time series
stored as vector-valued matrix elements). All normalizers can be passed
to [`load_dataset`](@ref) via a [`TreatmentGroup`](@ref).

## Multidimensional support

The standard Normalization.jl API (`fit`, `fit!`, `normalize`) is
extended to handle arrays whose elements are themselves arrays (e.g.
`Matrix{Union{Missing, Vector{Float64}}}`):

- `missing` entries are silently skipped during parameter estimation.
- Inner arrays are flattened before applying estimators such as
  `std`, `mad`, or `norm`.
- `normalize` returns a deep copy; the original array is never
  mutated.

```julia
X = [rand(10) for _ in 1:5, _ in 1:3]  # 5×3 matrix of time series
T = fit(ZScore, X)                       # fit globally
Y = normalize(X, T)                      # normalised copy
```

## Normalizers from Normalization.jl

The following normalizers are re-exported from Normalization.jl and
work transparently on both scalar and multidimensional data.

### Location-scale

| Normalizer | Estimator | Transform |
|:-----------|:----------|:----------|
| `ZScore` | `mean`, `std` | `(x - μ) / σ` |
| `Center` | `mean` | `x - μ` |
| `Scale` | `std` | `x / σ` |
| `ScaleMad` | `mad` (no consistency factor) | `x / mad` |
| `ScaleFirst` | `first` | `x / x[1]` |
| `MinMax` | `minimum`, `maximum` | `(x - min) / (max - min)` |

### Norm-based

| Normalizer | Estimator | Transform |
|:-----------|:----------|:----------|
| `PNorm1` | `‖x‖₁` | `x / ‖x‖₁` |
| `PNorm` | `‖x‖₂` | `x / ‖x‖₂` |
| `PNormInf` | `‖x‖∞` | `x / ‖x‖∞` |

## API

```@docs
DataTreatments.Scale
DataTreatments.ScaleMad
DataTreatments.ScaleFirst
DataTreatments.PNorm1
DataTreatments.PNorm
DataTreatments.PNormInf
```

### Internal extension API

```@docs
NormalizationExt._missingsafe
NormalizationExt.__mapdims!
```

## Usage examples

### Scalar matrix

```julia
using DataTreatments, Normalization

X = Union{Missing,Float64}[1.0 2.0; missing 4.0; 5.0 6.0]

T = fit(ZScore, X)     # fit on non-missing values
Y = normalize(X, T)    # missing entries are preserved
```

### Time series matrix

```julia
using DataTreatments, Normalization

# 5 samples × 2 channels, each element is a length-10 time series
X = [rand(10) for _ in 1:5, _ in 1:2]
X[2, 1] = missing                       # inject a missing series

T = fit(ScaleMad, X)   # MAD estimated from all non-missing series
Y = normalize(X, T)    # missing[2,1] preserved in output
```

### Column-wise fitting with `dims`

```julia
T = fit(MinMax, X; dims=2)  # fit one normalizer per column
Y = normalize(X, T)
```

## References

- **Normalization.jl repository**:
  https://github.com/JuliaAI/Normalization.jl