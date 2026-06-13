```@meta
CurrentModule = DataTreatments
```

# [Imputation](@id imputation)

Wrappers around
[Impute.jl](https://invenia.github.io/Impute.jl/stable/)
for handling missing and `NaN` values in datasets. Imputors can be
passed to [`load_dataset`](@ref) via a [`TreatmentGroup`](@ref).

## Supported Imputors

The following imputors are re-exported from Impute.jl and can be
used directly:

| Imputor | Description |
|:--------|:------------|
| [`Interpolate`](https://invenia.github.io/Impute.jl/stable/api/#Impute.Interpolate) | Linear interpolation between valid values |
| [`LOCF`](https://invenia.github.io/Impute.jl/stable/api/#Impute.LOCF) | Last Observation Carried Forward |
| [`NOCB`](https://invenia.github.io/Impute.jl/stable/api/#Impute.NOCB) | Next Observation Carried Backward |
| [`Substitute`](https://invenia.github.io/Impute.jl/stable/api/#Impute.Substitute) | Replace with a summary statistic (e.g. `mean`) |
| [`SVD`](https://invenia.github.io/Impute.jl/stable/api/#Impute.SVD) | SVD-based matrix completion |

Imputors can be chained in a tuple and are applied in order:

```julia
# fill gaps by interpolation, then carry forward/backward
# any remaining leading/trailing missings
impute = (Interpolate(), LOCF(), NOCB())
```

## Dispatch Behaviour

[`_impute`](@ref) dispatches on the type of the input data:

- **`AbstractMatrix`**: imputors are applied column-wise (`dims=2`).
- **`AbstractMatrix{T}`** where
  `T<:Union{Missing,Float,AbstractArray{<:Float}}`:
  imputors are applied without a dimension constraint.
- **Scalar or array-valued element**: imputors are applied only if
  the element is a non-missing `AbstractArray`; scalars are returned
  unchanged.

In all cases `NaN` and `"NULL"` are first declared as missing, and
`disallowmissing` is called if no missing values remain after
imputation.

## Internal API

```@docs
DataTreatments._impute
```

## References

- **Repository**: https://github.com/invenia/Impute.jl
- **Docs**: https://invenia.github.io/Impute.jl/stable/