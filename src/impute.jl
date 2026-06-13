# ---------------------------------------------------------------------------- #
#                             impute wrappers                                  #
# ---------------------------------------------------------------------------- #
"""
    _impute(data::AbstractArray, impute::Tuple{Vararg{<:Impute.Imputor}})
    _impute(
        data::AbstractMatrix{T},
        impute::Tuple{Vararg{<:Impute.Imputor}}
    ) where {T<:Union{Missing,Float,AbstractArray{<:Float}}}
    _impute(
        data::T,
        impute::Tuple{Vararg{<:Impute.Imputor}}
    ) where {T<:Union{Missing,Float,AbstractArray{<:Float}}}

Internal wrapper around
[Impute.jl](https://invenia.github.io/Impute.jl/stable/)
that handles missing and NaN value imputation. Dispatches on the
type of `data`:

- `AbstractArray`: applies each imputor column-wise (`dims=2`).
- `AbstractMatrix{T}` where `T<:Union{Missing,Float,AbstractArray}`:
  applies each imputor without specifying a dimension.
- Scalar or array-valued float element: applies imputors only if
  `data` is a non-missing `AbstractArray`, otherwise returns
  unchanged.

In all cases, `NaN` and `"NULL"` are first declared as missing via
[`Impute.declaremissings`](https://invenia.github.io/Impute.jl/\
stable/api/#Impute.declaremissings), then each imputor in the tuple
is applied in order via
[`Impute.impute!`](https://invenia.github.io/Impute.jl/\
stable/api/#Impute.impute!).
If no missing values remain after imputation, `disallowmissing` is
called to tighten the element type.

# Arguments
- `data`: Input data to impute. Can be an `AbstractArray`, an
  `AbstractMatrix` of floats, or a single scalar/array-valued
  float element.
- `impute::Tuple{Vararg{<:Impute.Imputor}}`: Ordered tuple of
  imputors (e.g. `(Impute.Interpolate(), Impute.Fill(0.0))`).

# Returns
- The imputed data, with missing values removed from the type if
  possible.

# See Also
- [Impute.jl docs](https://invenia.github.io/Impute.jl/stable/)
"""
function _impute(data::AbstractArray, impute::Tuple{Vararg{<:Impute.Imputor}})
    Impute.declaremissings(data; values=(NaN, "NULL"))

    for im in impute
        Impute.impute!(data, im; dims=2)
    end
    any(ismissing.(data)) || (data = disallowmissing(data))

    return data
end

function _impute(
    data::AbstractMatrix{T},
    impute::Tuple{Vararg{<:Impute.Imputor}}
) where {T<:Union{Missing,Float,AbstractArray{<:Float}}}
    Impute.declaremissings(data; values=(NaN, "NULL"))

    for im in impute
        Impute.impute!(data, im)
    end
    any(ismissing.(data)) || (data = disallowmissing(data))

    return data
end

function _impute(
    data::T,
    impute::Tuple{Vararg{<:Impute.Imputor}}
) where {T<:Union{Missing,Float,AbstractArray{<:Float}}}   
    if !ismissing(data) && typeof(data) <: AbstractArray
        Impute.declaremissings(data; values=(NaN, "NULL"))

        for im in impute
            Impute.impute!(data, im)
        end
        any(ismissing.(data)) || (data = disallowmissing(data))
    end

    return data
end
