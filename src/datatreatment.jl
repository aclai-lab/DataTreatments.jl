# ---------------------------------------------------------------------------- #
#                          getter methods collection                           #
# ---------------------------------------------------------------------------- #
"""
    get_discrete(dt::DataTreatment)
        -> (Matrix{Union{Missing,CategoricalValue}}, Vector{String})

Collect all [`DiscreteDataset`](@ref)s from `dt` and return their
data matrix and column names.

Returns an empty `(0×0 matrix, String[])` if no discrete datasets
are present.

# See Also
[`get_continuous`](@ref), [`get_tabular`](@ref)
"""
function get_discrete(
    dt::DataTreatment
)
    ds = collect(filter(d -> d isa DiscreteDataset, dt.data))
    return if isempty(ds)
        Matrix{Union{Missing,CategoricalValue}}(undef, 0, 0), String[]
    else
        get_data(ds), reduce(vcat, get_vnames.(ds))
    end
end

"""
    get_continuous(dt::DataTreatment{T})
        -> (Matrix{T}, Vector{String})

Collect all [`ContinuousDataset`](@ref)s from `dt` and return their
data matrix and column names.

Returns an empty `(0×0 matrix, String[])` if no continuous datasets
are present.

# Type Parameters
- `T<:Float`: Floating-point type of the dataset.

# See Also
[`get_discrete`](@ref), [`get_tabular`](@ref)
"""
function get_continuous(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = collect(filter(d -> d isa ContinuousDataset{T}, dt.data))
    return if isempty(ds)
        Matrix{T}(undef, 0, 0), String[]
    else
        get_data(ds), reduce(vcat, get_vnames.(ds))
    end
end

"""
    get_aggregated(dt::DataTreatment{T})
        -> (Matrix{T}, Vector{String})

Collect all [`MultidimDataset`](@ref)s whose features are
[`AggregateFeat`](@ref) (i.e. produced by [`aggregate`](@ref))
and return their tabular scalar matrix and column names.

Returns an empty `(0×0 matrix, String[])` if no aggregated
datasets are present.

# Type Parameters
- `T<:Float`: Floating-point type of the dataset.

# See Also
[`get_reduced`](@ref), [`get_tabular`](@ref)
"""
function get_aggregated(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa AggregateFeat{T}, get_info(d)), dt.data)
    return if isempty(ds)
        Matrix{T}(undef, 0, 0), String[]
    else
        (get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

"""
    get_reduced(dt::DataTreatment{T})
        -> (Matrix{VecOrMat{T}}, Vector{String})

Collect all [`MultidimDataset`](@ref)s whose features are
[`ReduceFeat`](@ref) (i.e. produced by [`reducesize`](@ref))
and return their array-valued matrix and column names.

Returns an empty `(0×0 matrix, String[])` if no reduced datasets
are present.

# Type Parameters
- `T<:Float`: Floating-point type of the inner arrays.

# See Also
[`get_aggregated`](@ref), [`get_multidim`](@ref)
"""
function get_reduced(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa ReduceFeat, get_info(d)), dt.data)
    return if isempty(ds)
        (Matrix{VecOrMat{T}}(undef, 0, 0), String[])
    else
        (get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

"""
    is_tabular(dt::DataTreatment) -> Bool

Return `true` if **all** datasets in `dt` are tabular (i.e. every
dataset satisfies `is_tabular`). Returns `false` if any dataset is
multidimensional.

# See Also
[`is_multidim`](@ref), [`has_tabular`](@ref)
"""
is_tabular(dt::DataTreatment) = all(is_tabular.(dt.data))

"""
    is_multidim(dt::DataTreatment) -> Bool

Return `true` if **all** datasets in `dt` are multidimensional
(i.e. every dataset satisfies `is_multidim`).

# See Also
[`is_tabular`](@ref), [`has_multidim`](@ref)
"""
is_multidim(dt::DataTreatment) = all(is_multidim.(dt.data))

"""
    has_tabular(dt::DataTreatment) -> Bool

Return `true` if **at least one** dataset in `dt` is tabular.

# See Also
[`is_tabular`](@ref), [`has_multidim`](@ref)
"""
has_tabular(dt::DataTreatment) = any(is_tabular.(dt.data))

"""
    has_multidim(dt::DataTreatment) -> Bool

Return `true` if **at least one** dataset in `dt` is
multidimensional.

# See Also
[`is_multidim`](@ref), [`has_tabular`](@ref)
"""
has_multidim(dt::DataTreatment) = any(is_multidim.(dt.data))

# ---------------------------------------------------------------------------- #
#                             get tabular method                               #
# ---------------------------------------------------------------------------- #
"""
    get_tabular(dt::DataTreatment{T})
        -> (Matrix, Vector{String})

Merge all tabular-like datasets from `dt` into a single matrix and
return it together with the concatenated column names.

Collects discrete, continuous, and aggregated multidimensional data
(in that order) and horizontally concatenates them. If no missing
values remain after merging, `disallowmissing` is called on the
result.

Returns an empty `(0×0 Matrix{T}, String[])` if no tabular data is
present.

!!! note
    The element type of the output matrix is the union of the
    element types of the collected sub-matrices (e.g.
    `Union{Missing, Int, Float64}`). If no `missing` values are
    present, the type is narrowed automatically.

# See Also
[`get_discrete`](@ref), [`get_continuous`](@ref),
[`get_aggregated`](@ref), [`get_multidim`](@ref)
"""
@inline function get_tabular(
    dt::DataTreatment{T}
) where {T<:Float}
    mats = get_discrete(dt), get_continuous(dt), get_aggregated(dt)
    idxs = findall(x -> !(isempty(x)), map(first, mats))

    isempty(idxs) && return(
        (Matrix{T}(undef, 0,0), String[])
    )

    X = collect(zip(mats[idxs]...))
    Tnew = unique(eltype.(X[1]))
    data = Matrix{Union{Tnew...}}(reduce(hcat, X[1]))
    any(ismissing.(data)) || (data = disallowmissing(data))

    return (data, reduce(vcat, X[2]))
end

# ---------------------------------------------------------------------------- #
#                            get multidim method                               #
# ---------------------------------------------------------------------------- #
"""
    get_multidim(dt::DataTreatment{T}; kwargs...)
        -> (Matrix{VecOrMat{T}}, Vector{String})

Collect all reduced multidimensional datasets from `dt` and return
their array-valued matrix and column names.

Delegates to [`get_reduced`](@ref) and calls `disallowmissing` on
the result if no missing values are present.

Returns an empty `(0×0 matrix, String[])` if no reduced datasets
are present.

# See Also
[`get_reduced`](@ref), [`get_tabular`](@ref)
"""
@inline function get_multidim(
    dt::DataTreatment{T};
    kwargs...
) where {T<:Float}
    data, vnames = get_reduced(dt; kwargs...)
    any(ismissing.(data)) || (data = disallowmissing(data))

    return data, vnames
end

# ---------------------------------------------------------------------------- #
#                         filter missing by percentage                         #
# ---------------------------------------------------------------------------- #
"""
    filter_missing(dt::DataTreatment{T}, perc::Real;
                   include_nans::Bool=true,
                   dims::Int=2) -> DataTreatment{T}

Remove rows or columns from all datasets in `dt` that exceed the
missing-value threshold `perc`.

# Arguments
- `dt`: Source [`DataTreatment`](@ref).
- `perc`: Maximum allowed fraction of missing values in `[0.0, 1.0]`.
  Rows/columns with a higher fraction are dropped.

# Keyword Arguments
- `include_nans::Bool=true`: If `true`, `NaN` values are counted as
  missing when computing the fraction (ignored for
  [`DiscreteDataset`](@ref)).
- `dims::Int=2`: Filtering direction.
  - `1` — **row-wise**: a row is dropped if the fraction of missing
    values across *all* columns (summed over all sub-datasets)
    exceeds `perc`. The same keep-mask is applied to every
    sub-dataset and to the target vector, preserving row alignment.
  - `2` — **column-wise** (default): each column is independently
    dropped if its missing fraction exceeds `perc`. Sub-datasets
    are filtered independently.

# Returns
A new `DataTreatment{T}` with the same structure as `dt` but with
offending rows or columns removed. Metadata feature structs
([`DiscreteFeat`](@ref), [`ContinuousFeat`](@ref), etc.) are
re-indexed to reflect the new row/column numbering via
[`_reindex_feat`](@ref).

# Throws
- `AssertionError`: if `perc` is outside `[0.0, 1.0]`.

# Examples

```julia
# drop columns with more than 20% missing values
dt2 = filter_missing(dt, 0.2)

# drop rows with more than 10% missing values (including NaNs)
dt2 = filter_missing(dt, 0.1; dims=1)

# drop rows based on missing only (ignore NaNs)
dt2 = filter_missing(dt, 0.1; dims=1, include_nans=false)
```

# See Also
[`DataTreatment`](@ref), [`_reindex_feat`](@ref)
"""
function filter_missing(
    dt::DataTreatment{T},
    perc::Real;
    include_nans::Bool=true,
    dims::Int=2
) where T
    @assert 0.0 ≤ perc ≤ 1.0 "perc must be between 0.0 and 1.0, got $perc"

    if dims == 1  # row-wise: global keep mask across ALL sub-datasets
        n = nrows(dt)
        total_cols = sum(ncols(d) for d in dt.data)
        row_badcount = zeros(Int, n)

        for d in dt.data
            missings = get_missingidxs.(d.info)
            missings = include_nans && !isa(d, DiscreteDataset) ?
                union.(missings, get_nanidxs.(d.info)) :
                missings
            foreach(idxs -> (row_badcount[idxs] .+= 1), missings)
        end

        keep = findall((row_badcount ./ total_cols) .≤ perc)

        data = map(dt.data) do d
            new_info = _reindex_feat.(d.info, Ref(keep))

            isa(d, MultidimDataset{<:Any, AggregateFeat}) ?
                typeof(d).name.wrapper(d.data[keep, :], new_info, d.groups) :
            isa(d, MultidimDataset{<:Any, ReduceFeat}) ?
                typeof(d).name.wrapper(d.data[keep, :], new_info, d.groups) :
                typeof(d).name.wrapper(d.data[keep, :], new_info)
        end

        target = get_target(dt)
        target = isempty(target) ? target : target[keep]

        return DataTreatment{T}(data, target, get_treats(dt), get_balance(dt))

    else  # col-wise: independent per sub-dataset (no row consistency issue)
        data = map(dt.data) do d
            missings = get_missingidxs.(d.info)
            missings = include_nans && !isa(d, DiscreteDataset) ?
                union.(missings, get_nanidxs.(d.info)) :
                missings

            n = nrows(d)
            keep = (length.(missings) ./ n) .≤ perc

            isa(d, MultidimDataset{<:Any, AggregateFeat}) ?
                typeof(d).name.wrapper(
                    d.data[:, keep], d.info[keep], d.groups) :
                typeof(d).name.wrapper(
                    d.data[:, keep], d.info[keep])
        end

        return DataTreatment{T}(
            data, get_target(dt), get_treats(dt), get_balance(dt))
    end
end

