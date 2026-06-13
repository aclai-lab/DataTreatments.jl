# ---------------------------------------------------------------------------- #
#                      @_Normalization extended methods                        #
# ---------------------------------------------------------------------------- #
scale(s) = Base.Fix2(/, s)

@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale

@_Normalization PNorm1 ((x)->norm(x, 1),) scale
@_Normalization PNorm ((x)->norm(x, 2),) scale
@_Normalization PNormInf ((x)->norm(x, Inf),) scale

@doc """
    Scale

Normalizes by dividing by the standard deviation.
Equivalent to z-score normalization without mean centering.

Defined via `@_Normalization` using `std` as the estimator.

# See Also
- [`ScaleMad`](@ref), [`ScaleFirst`](@ref)
- [`Normalization.@_Normalization`](https://github.com/JuliaAI/Normalization.jl)
""" Scale

@doc """
    ScaleMad

Normalizes by dividing by the Median Absolute Deviation (MAD),
computed without the standard normal consistency factor
(`normalize=false`).

Defined via `@_Normalization` using `mad` as the estimator.

# See Also
- [`Scale`](@ref), [`ScaleFirst`](@ref)
""" ScaleMad

@doc """
    ScaleFirst

Normalizes by dividing by the first element of the series.
Useful for relative scaling with respect to an initial reference
value.

Defined via `@_Normalization` using `first` as the estimator.

# See Also
- [`Scale`](@ref), [`ScaleMad`](@ref)
""" ScaleFirst

@doc """
    PNorm1

Normalizes by dividing by the L1-norm (sum of absolute values) of
the series.

Defined via `@_Normalization` using `x -> norm(x, 1)`.

# See Also
- [`PNorm`](@ref), [`PNormInf`](@ref)
""" PNorm1

@doc """
    PNorm

Normalizes by dividing by the L2-norm (Euclidean norm) of the
series.

Defined via `@_Normalization` using `x -> norm(x, 2)`.

# See Also
- [`PNorm1`](@ref), [`PNormInf`](@ref)
""" PNorm

@doc """
    PNormInf

Normalizes by dividing by the L∞-norm (maximum absolute value) of
the series.

Defined via `@_Normalization` using `x -> norm(x, Inf)`.

# See Also
- [`PNorm1`](@ref), [`PNorm`](@ref)
""" PNormInf