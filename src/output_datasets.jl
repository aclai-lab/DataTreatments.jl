# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
"""
    DiscreteFeat{T} <: AbstractDataFeature

Metadata for a **discrete (categorical)** feature in a dataset.

# Fields
- `id::Int`: Column index in the source data.
- `vname::String`: Original column name.
- `valididxs::Vector{Int}`: Row indices with valid (non-missing) values.
- `missingidxs::Vector{Int}`: Row indices with `missing` values.
- `datatype::Type`: Original column element type.

# Example

```
Source column "color" (id=3, 10 rows, rows 4 and 7 missing)

Row:  1       2       3       4        5       6       7
Val: "red"  "blue" "green" missing  "red"  "blue"  missing ...

→ id          = 3
→ vname       = "color"
→ valididxs   = [1, 2, 3, 5, 6, 8, 9, 10]
→ missingidxs = [4, 7]
→ datatype    = String
```

# See Also
[`DiscreteDataset`](@ref), [`ContinuousFeat`](@ref)
"""
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    datatype::Type
end

"""
    ContinuousFeat{T} <: AbstractDataFeature

Metadata for a **continuous (numeric scalar)** feature in a dataset.

# Fields
- `id::Int`: Column index in the source data.
- `vname::String`: Original column name.
- `valididxs::Vector{Int}`: Row indices with valid (non-missing,
  non-NaN) values.
- `missingidxs::Vector{Int}`: Row indices with `missing` values.
- `nanidxs::Vector{Int}`: Row indices with `NaN` values.

# Example

```
Source column "temperature" (id=2, 8 rows)
Row:  1      2      3        4      5     6      7      8
Val: 22.1  19.5  missing  24.0   21.3   NaN   20.8   23.1

→ id          = 2
→ vname       = "temperature"
→ valididxs   = [1, 2, 4, 5, 7, 8]
→ missingidxs = [3]
→ nanidxs     = [6]
```

# See Also
[`ContinuousDataset`](@ref), [`DiscreteFeat`](@ref)
"""
struct ContinuousFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
end

"""
    AggregateFeat{T} <: AbstractDataFeature

Metadata for an **aggregated scalar** feature derived from a
multidimensional column (e.g. a time series) via a feature function
applied over a sliding window.

Each source column produces `nfeatures × nwindows` output columns,
each represented by one `AggregateFeat`.

# Fields
- `id::Int`: Source column index in the original data.
- `subid::Int`: Position of this feature in the flattened output.
- `vname::String`: Original column name.
- `dims::Int`: Length of the source arrays.
- `feat::Base.Callable`: Aggregation function (e.g. `mean`, `std`).
- `nwin::Int`: Window index this feature was computed on.
- `valididxs::Vector{Int}`: Row indices with valid entries.
- `missingidxs::Vector{Int}`: Row indices where the cell is `missing`.
- `nanidxs::Vector{Int}`: Row indices where the scalar result is `NaN`.
- `hasmissing::Vector{Int}`: Row indices where the source array
  contains internal `missing` values.
- `hasnans::Vector{Int}`: Row indices where the source array contains
  internal `NaN` values.

# Example

```
Source column "signal" (id=1, dims=100)
  Row 1: [0.1, 0.3, ..., 0.9]      (valid)
  Row 2: [0.2, missing, ..., 0.7]  (internal missing)
  Row 3: missing                   (whole cell missing)

Apply features=[mean, std] over nwindows=2
→ 4 output columns:

  subid=1  feat=mean  nwin=1  id=1
  subid=2  feat=std   nwin=1  id=1
  subid=3  feat=mean  nwin=2  id=1
  subid=4  feat=std   nwin=2  id=1
```

# See Also
[`MultidimDataset`](@ref), [`ReduceFeat`](@ref),
[`aggregate`](@ref)
"""
struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    subid::Int
    vname::String
    dims::Int
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}
end

"""
    ReduceFeat{T} <: AbstractDataFeature

Metadata for a **reduced-size** feature derived from a
multidimensional column. Unlike [`AggregateFeat`](@ref), the output
preserves the array structure but reduces its size (e.g. downsampling
a 10 000-point time series to 256 points).

# Fields
- `id::Int`: Source column index in the original data.
- `vname::String`: Original column name.
- `dims::Int`: Length of the source arrays.
- `reducefunc::Base.Callable`: Reduction/downsampling function.
- `valididxs::Vector{Int}`: Row indices with valid entries.
- `missingidxs::Vector{Int}`: Row indices where the cell is `missing`.
- `nanidxs::Vector{Int}`: Row indices where the reduced result has
  `NaN`.
- `hasmissing::Vector{Int}`: Row indices where the source array has
  internal `missing` values.
- `hasnans::Vector{Int}`: Row indices where the source array has
  internal `NaN` values.

# Example

```
Source column "audio" (id=5, dims=10000)
  Row 1: [0.1, 0.2, ..., 0.9]   (valid)
  Row 2: [0.3, NaN, ..., 0.5]   (internal NaN)
  Row 3: missing                (whole cell missing)

Apply reducefunc = downsample(256)
  Row 1: [0.12, 0.34, ..., 0.88]  (length 256) ← valid
  Row 2: [0.31, 0.49, ..., 0.50]  (length 256) ← hasnans=[2]
  Row 3: missing                               ← missingidxs=[3]
```

# See Also
[`MultidimDataset`](@ref), [`AggregateFeat`](@ref),
[`reducesize`](@ref)
"""
struct ReduceFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    dims::Int
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}
end

# Base.eltype(::ContinuousFeat{T}) where T = T
Base.eltype(::AggregateFeat{T}) where T = T
Base.eltype(::ReduceFeat{T}) where T = T

get_subid(f::AggregateFeat) = f.subid
get_dims(f::Union{AggregateFeat,ReduceFeat}) = f.dims

get_vnames(d::AbstractDataFeature) = d.vname
get_missingidxs(d::AbstractDataFeature) = d.missingidxs
get_nanidxs(d::AbstractDataFeature) = d.nanidxs

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
function _get_features(a::Base.Callable)
    features = a.features
    return features isa Base.Callable ?
        (features=(features,)) : features
end
_get_reducefunc(r::Base.Callable) = r.reducefunc

"""
    _reindex_groups(groups, idxs) -> Vector{Vector{Int}}

Re-map group indices from the original column space to the new subset
`idxs`. Groups with no members in `idxs` are dropped.

# Arguments
- `groups::Vector{Vector{Int}}`: Original column groupings.
- `idxs::AbstractVector{Int}`: New column subset (sorted).

# Returns
- Remapped groups containing only indices present in `idxs`,
  renumbered to the new contiguous range `1:length(idxs)`.
  Returns an empty `Vector{Vector{Int}}` if `groups` is `nothing`.
"""
function _reindex_groups(groups::Vector{Vector{Int}}, idxs::AbstractVector{Int})
    idx_set = Set(idxs)
    # Build reverse mapping: old index -> new index
    old_to_new = Dict(old => new for (new, old) in enumerate(idxs))
    
    new_groups = Vector{Vector{Int}}()

    for grp in groups
        new_grp = [old_to_new[i] for i in grp if i in idx_set]
        if !isempty(new_grp)
            push!(new_groups, new_grp)
        end
    end
    
    return new_groups
end

_reindex_groups(
    groups::Nothing,
    idxs::AbstractVector{Int}
) = Vector{Vector{Int}}()

"""
    _reindex_feat(f, keep) -> AbstractDataFeature

Return a copy of feature metadata `f` with all row-index vectors
filtered and remapped to the new row subset `keep`.

Useful when rows are dropped (e.g. after class balancing) and the
stored index vectors must reflect the new row numbering.

# Arguments
- `f::AbstractDataFeature`: Any metadata feature struct.
- `keep::AbstractVector{Int}`: Sorted vector of original row indices
  to retain.

# Returns
- A new instance of the same type as `f` with all `*idxs` fields
  remapped to the new row space.
"""
function _reindex_feat(f::DiscreteFeat{T}, keep::AbstractVector{Int}) where T
    old_to_new = Dict(old => new for (new, old) in enumerate(keep))
    keep_set   = Set(keep)
    DiscreteFeat{T}(
        f.id, f.vname,
        [old_to_new[i] for i in f.valididxs   if i in keep_set],
        [old_to_new[i] for i in f.missingidxs  if i in keep_set],
        f.datatype
    )
end

function _reindex_feat(f::ContinuousFeat{T}, keep::AbstractVector{Int}) where T
    old_to_new = Dict(old => new for (new, old) in enumerate(keep))
    keep_set   = Set(keep)
    ContinuousFeat{T}(
        f.id, f.vname,
        [old_to_new[i] for i in f.valididxs   if i in keep_set],
        [old_to_new[i] for i in f.missingidxs  if i in keep_set],
        [old_to_new[i] for i in f.nanidxs      if i in keep_set],
    )
end

function _reindex_feat(f::AggregateFeat{T}, keep::AbstractVector{Int}) where T
    old_to_new = Dict(old => new for (new, old) in enumerate(keep))
    keep_set   = Set(keep)
    AggregateFeat{T}(
        f.id, f.subid, f.vname, f.dims, f.feat, f.nwin,
        [old_to_new[i] for i in f.valididxs   if i in keep_set],
        [old_to_new[i] for i in f.missingidxs  if i in keep_set],
        [old_to_new[i] for i in f.nanidxs      if i in keep_set],
        [old_to_new[i] for i in f.hasmissing   if i in keep_set],
        [old_to_new[i] for i in f.hasnans      if i in keep_set],
    )
end

function _reindex_feat(f::ReduceFeat{T}, keep::AbstractVector{Int}) where T
    old_to_new = Dict(old => new for (new, old) in enumerate(keep))
    keep_set   = Set(keep)
    ReduceFeat{T}(
        f.id, f.vname, f.dims, f.reducefunc,
        [old_to_new[i] for i in f.valididxs   if i in keep_set],
        [old_to_new[i] for i in f.missingidxs  if i in keep_set],
        [old_to_new[i] for i in f.nanidxs      if i in keep_set],
        [old_to_new[i] for i in f.hasmissing   if i in keep_set],
        [old_to_new[i] for i in f.hasnans      if i in keep_set],
    )
end

# ---------------------------------------------------------------------------- #
#                           output dataset structs                             #
# ---------------------------------------------------------------------------- #
"""
    DiscreteDataset{T} <: AbstractDataset

Output dataset for **discrete (categorical)** columns collected by
[`load_dataset`](@ref).

Categorical columns are integer-encoded via `_discrete_encode` so
that downstream ML models receive a plain numeric matrix.

# Fields
- `data::AbstractMatrix`: Integer-coded matrix
  (`nrows × nfeatures`). Values are `Int` level codes or `missing`.
- `info::Vector{<:DiscreteFeat}`: One [`DiscreteFeat`](@ref) per
  output column.

# Layout

```
Source columns (ids=[1,2,3])
 ┌──────────┬──────────┬──────────┐
 │ color    │ shape    │ size     │
 ├──────────┼──────────┼──────────┤
 │ "red"    │ "circle" │ "small"  │
 │ "blue"   │ missing  │ "large"  │
 │ "red"    │ "square" │ "small"  │
 └──────────┴──────────┴──────────┘
         │  _discrete_encode
 ┌──────────┬──────────┬──────────┐
 │  col_1   │  col_2   │  col_3   │
 ├──────────┼──────────┼──────────┤
 │    2     │    1     │    2     │
 │    1     │ missing  │    1     │
 │    2     │    3     │    2     │
 └──────────┴──────────┴──────────┘
```

# Constructors
- `DiscreteDataset(data, info)`: Direct constructor from a
  pre-built matrix and metadata vector.
- `DiscreteDataset(ids, data, vnames, datastruct, impute)`:
  Internal constructor — selects `ids` from `data`, encodes
  categorically, applies optional `impute`, and builds
  [`DiscreteFeat`](@ref) metadata.

# Arguments (internal constructor)
- `ids::Vector{Int}`: Column indices to include.
- `data::AbstractMatrix`: Full raw dataset matrix.
- `vnames::Vector{String}`: All column names.
- `datastruct::NamedTuple`: Per-column metadata from `_inspecting`.
- `impute`: `nothing` or a `Tuple` of `Impute.Imputor`s.

# See Also
[`ContinuousDataset`](@ref), [`MultidimDataset`](@ref),
[`DiscreteFeat`](@ref)
"""
mutable struct DiscreteDataset{T} <: AbstractDataset
    data::AbstractMatrix
    info::Vector{<:DiscreteFeat}

    DiscreteDataset(
        data::AbstractMatrix,
        info::Vector{<:DiscreteFeat{T}}
    ) where T = new{T}(data, info)
    
    function DiscreteDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}}
    )
        codes = _discrete_encode(@views(data[:, ids]))
        vnames = vnames[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        datatype = datastruct.datatype[ids]

        data = isempty(codes) ?
            Matrix{eltype(codes)}(undef, 0, 0) :
            stack(codes)

        isnothing(impute) || (data = _impute(data, impute))

        return new{eltype(codes)}(
            data,
            [DiscreteFeat{eltype(codes)}(
                ids[i],
                vnames[i],
                valid[i],
                miss[i],
                datatype[i]
            )
                for i in eachindex(ids)]
        )
    end
end

"""
    ContinuousDataset{T} <: AbstractDataset

Output dataset for **continuous (numeric scalar)** columns produced
by [`load_dataset`](@ref).

Numeric columns are cast to `float_type`, then optionally imputed
and normalized column-wise.

# Fields
- `data::AbstractMatrix`: Float matrix (`nrows × nfeatures`).
  Values are `Union{Missing, T}` before imputation, or `T`
  afterwards.
- `info::Vector{<:ContinuousFeat}`: One [`ContinuousFeat`](@ref)
  per output column.

# Layout

```
Source columns (ids=[4,5,6])
 ┌────────────┬──────────┬──────────┐
 │temperature │ pressure │ humidity │
 ├────────────┼──────────┼──────────┤
 │  22.1      │  1013.0  │   55.3   │
 │  missing   │   987.5  │   NaN    │
 │  19.8      │  1001.2  │   60.1   │
 └────────────┴──────────┴──────────┘
         │  cast to Float64
         │  normalize! via NaNSafe{MinMax{Float64}}
 ┌────────────┬──────────┬──────────┐
 │   col_4    │  col_5   │  col_6   │
 ├────────────┼──────────┼──────────┤
 │   0.82     │   1.00   │   0.72   │
 │  missing   │   0.00   │   NaN    │
 │   0.00     │   0.54   │   1.00   │
 └────────────┴──────────┴──────────┘
```

!!! note "Processing order"
    1. Cast each element to `float_type` (preserving `missing`).
    2. Apply `impute` (if provided).
    3. Apply `norm` column-wise via `NaNSafe{norm{T}}` (if provided).
    `NaNSafe` ignores `NaN` when estimating normalization parameters.

# Type Parameter
- `T`: Floating-point type (e.g. `Float64`, `Float32`).

# Constructors
- `ContinuousDataset(data, info)`: Direct constructor from a
  pre-built matrix and metadata vector.
- `ContinuousDataset(ids, data, vnames, datastruct,
  impute, norm, float_type)`: Internal constructor.

# Arguments (internal constructor)
- `ids::Vector{Int}`: Column indices to include.
- `data::AbstractMatrix`: Full raw dataset matrix.
- `vnames::Vector{String}`: All column names.
- `datastruct::NamedTuple`: Per-column metadata from `_inspecting`.
- `impute`: `nothing` or a `Tuple` of `Impute.Imputor`s.
- `norm`: `nothing` or a `Type{<:AbstractNormalization}`
  (e.g. `MinMax`, `ZScore`).
- `float_type::Type`: Target floating-point type.

# See Also
[`DiscreteDataset`](@ref), [`MultidimDataset`](@ref),
[`ContinuousFeat`](@ref)
"""
mutable struct ContinuousDataset{T} <: AbstractDataset
    data::AbstractMatrix
    info::Vector{<:ContinuousFeat}

    ContinuousDataset(
        data::AbstractMatrix,
        info::Vector{<:ContinuousFeat{T}}
    ) where T = new{T}(data, info)

    function ContinuousDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}},
        norm::Union{Type{<:AbstractNormalization},Nothing},
        float_type::Type
    )
        vnames = vnames[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        nan = datastruct.nanidxs[ids]

        data = if isempty(ids)
            Matrix{float_type}(undef, 0, 0)
        else
            reduce(hcat, [map(x -> ismissing(x) ?
                missing :
                float_type(x), @view data[:, id])
                for id in ids]
            )
        end

        isnothing(impute) || (data = _impute(data, impute))

        if !isnothing(norm)
            data = Impute.replace(data; values=NaN)
            normalize!(data, NaNSafe{norm{float_type}}, dims=1)
        end

        return new{float_type}(
            data,
            [ContinuousFeat{float_type}(
                ids[i],
                vnames[i],
                valid[i],
                miss[i],
                nan[i]
            )
                for i in eachindex(ids)]
        )
    end
end

"""
    MultidimDataset{T,S} <: AbstractDataset

Output dataset for **multidimensional** columns (e.g. time series,
spectrograms) produced by [`load_dataset`](@ref).

The type parameter `S` determines the output layout:

| `S`            | `aggrfunc`     | Output layout              |
|:---------------|:---------------|:---------------------------|
| `AggregateFeat`| `aggregate`    | tabular scalar matrix      |
| `ReduceFeat`   | `reducesize`   | matrix of smaller arrays   |

# Fields
- `data::AbstractArray`: Processed data matrix.
- `info::Vector{<:Union{AggregateFeat,ReduceFeat}}`: One metadata
  entry per output column.
- `groups::Union{Nothing,Vector{Vector{Int}}}`: Optional column
  groupings (e.g. `[[1,2,3,4],[5,6,7,8]]` means columns 1–4 come
  from source column A and 5–8 from source column B).

## Strategy 1 — `aggregate` → `S = AggregateFeat`

```
Source column "signal" (dims=1000)
  Row 1: [0.1, 0.2, ..., 0.9]
  Row 2: [0.3, 0.4, ..., 0.8]
  Row 3: missing

Apply features=[mean, std], nwindows=2
 ┌──────────┬─────────┬──────────┬─────────┐
 │mean,win1 │std,win1 │mean,win2 │std,win2 │
 ├──────────┼─────────┼──────────┼─────────┤
 │  0.312   │  0.098  │  0.601   │  0.112  │
 │  0.421   │  0.077  │  0.699   │  0.085  │
 │ missing  │ missing │ missing  │ missing │
 └──────────┴─────────┴──────────┴─────────┘
```

## Strategy 2 — `reducesize` → `S = ReduceFeat`

```
Source column "audio" (dims=10000)
  Row 1: [0.1, 0.2, ..., 0.9]  (valid)
  Row 2: [0.3, NaN, ..., 0.8]  (internal NaN)
  Row 3: missing

Apply reducefunc = downsample(256)
  Row 1: [0.12, ..., 0.88]  (length 256)
  Row 2: [0.31, ..., 0.50]  (length 256)
  Row 3: missing
```

!!! note "Processing order"
    1. Apply `aggrfunc` (aggregation or downsampling).
    2. Apply `impute` (if provided).
    3. Apply `norm` via `NaNSafe{norm{T}}` (if provided).
       For `AggregateFeat`: normalization is applied per group.
       For `ReduceFeat`: normalization is applied element-wise.

# Type Parameters
- `T`: Element type of the inner arrays (e.g. `Float64`).
- `S`: `AggregateFeat` or `ReduceFeat`.

# Constructors
- `MultidimDataset(data, info::Vector{<:AggregateFeat}, groups)`:
  Direct constructor for aggregated data.
- `MultidimDataset(data, info::Vector{<:ReduceFeat}, groups)`:
  Direct constructor for reduced data.
- `MultidimDataset(ids, data, vnames, datastruct, aggrfunc,
  impute, norm, float_type, groups)`: Internal constructor.

# Arguments (internal constructor)
- `ids::Vector{Int}`: Column indices to include.
- `data::AbstractMatrix`: Full raw dataset matrix.
- `vnames::Vector{String}`: All column names.
- `datastruct::NamedTuple`: Per-column metadata from `_inspecting`.
- `aggrfunc::Base.Callable`: `aggregate(...)` or `reducesize(...)`.
- `impute`: `nothing` or a `Tuple` of `Impute.Imputor`s.
- `norm`: `nothing` or a `Type{<:AbstractNormalization}`.
- `float_type::Type{T}`: Target floating-point type.
- `groups`: `nothing` or a `Tuple{Vararg{Symbol}}` for grouping.

# See Also
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`AggregateFeat`](@ref), [`ReduceFeat`](@ref),
[`aggregate`](@ref), [`reducesize`](@ref)
"""
mutable struct MultidimDataset{T,S} <: AbstractDataset
    data::AbstractArray
    info::Vector{<:Union{AggregateFeat,ReduceFeat}}
    groups::Union{Nothing,Vector{Vector{Int}}}

    MultidimDataset(
        data::AbstractArray,
        info::Vector{<:AggregateFeat{T}},
        groups::Union{Nothing,Vector{Vector{Int}}}
    ) where T = new{T,AggregateFeat}(data, info, groups)

    MultidimDataset(
        data::AbstractArray,
        info::Vector{<:ReduceFeat{T}},
        groups::Union{Nothing,Vector{Vector{Int}}}=nothing
    ) where T = new{T,ReduceFeat}(data, info, groups)

    function MultidimDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        aggrfunc::F,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}},
        norm::Union{Nothing,Type{<:AbstractNormalization}},
        float_type::Type{T},
        groups::Union{Nothing, Tuple{Vararg{Symbol}}}
    ) where {T<:Float, F<:Base.Callable}
        data = @view data[:, ids]
        vnames = vnames[ids]
        dims = datastruct.dims[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        nan = datastruct.nanidxs[ids]
        hasmiss = datastruct.hasmissing[ids]
        hasnan = datastruct.hasnans[ids]

        md, nwindows = aggrfunc(data, valid, float_type)

        if !isnothing(impute) && !isempty(md)
            md = _impute(md, impute)
        end

        md_feats, md, grouped = if hasfield(typeof(aggrfunc), :features)
            tuples = Iterators.flatten((
                ((c, f, n) for f in _get_features(aggrfunc) 
                for n in 1:nwindows[c])
                for c in eachindex(ids)
            ))

            md_feats = [AggregateFeat{float_type}(
                ids[c],
                j,
                vnames[c],
                dims[c],
                f,
                n,
                valid[c],
                miss[c],
                nan[c],
                hasmiss[c],
                hasnan[c]
            ) for (j, (c, f, n)) in enumerate(tuples)]

            grouped = isnothing(groups) ?
                _groupby(md_feats, (:vname,)) :
                _groupby(md_feats, groups)

            if !isnothing(norm) && !isempty(md)
                Impute.replace!(md; values=NaN)
                for i in grouped
                    md[:,i] = normalize(md[:,i], NaNSafe{norm{float_type}})
                end
            end

            md_feats, md, grouped
        else
            md_feats = [ReduceFeat{AbstractArray{float_type}}(
                ids[i],
                vnames[c],
                dims[c],
                _get_reducefunc(aggrfunc),
                valid[c],
                miss[c],
                nan[c],
                hasmiss[c],hasnan[c]
            ) for (i, c) in enumerate(axes(md,2))]

            if !isnothing(norm) && !isempty(md)
                md = normalize(md, NaNSafe{norm{float_type}}; dims=1)
            end

            md_feats, md, nothing
        end

        new{float_type,eltype(md_feats)}(md, md_feats, grouped)
    end
end

# ---------------------------------------------------------------------------- #
#                                  methods                                     #
# ---------------------------------------------------------------------------- #
Base.eltype(::DiscreteDataset{T}) where T = T
Base.eltype(::ContinuousDataset{T}) where T = T
Base.eltype(::MultidimDataset{T})where T = T

nrows(d::AbstractDataset) = size(d.data, 1)
ncols(d::AbstractDataset) = size(d.data, 2)
Base.ndims(d::AbstractDataset) = ndims(d.data)
Base.size(d::AbstractDataset) = size(d.data)
Base.size(d::AbstractDataset, i::Int) = size(d.data, i)
Base.length(d::AbstractDataset) = length(d.data)

Base.getindex(ds::MultidimDataset, idxs::AbstractVector{Int}) =
    MultidimDataset(
        @view(ds.data[:, idxs]),
        ds.info[idxs],
        _reindex_groups(ds.groups, idxs)
    )

Base.isempty(ds::AbstractDataset) = isempty(ds.data)

get_dims(d::MultidimDataset) = [get_dims(f) for f in d.info]

get_data(d::Vector{<:AbstractDataset}) = reduce(hcat, get_data.(d))
get_data(d::DiscreteDataset)= d.data
get_data(d::ContinuousDataset)= d.data
get_data(d::MultidimDataset)= d.data

get_info(d::Vector{<:AbstractDataset}) = reduce(vcat, get_info.(d))
get_info(d::AbstractDataset) = d.info

is_tabular(d::AbstractDataset) = d isa Union{
    ContinuousDataset,
    DiscreteDataset,
    MultidimDataset{<:Any, AggregateFeat}
}
is_multidim(d::AbstractDataset) = isa(d, MultidimDataset{<:Any, ReduceFeat})

get_vnames(d::AbstractDataset) = get_vnames.(d.info)

function get_vnames(
    ds::MultidimDataset{<:Any,<:AggregateFeat};
    groupby_split::Bool=false
)
    ["$(f.vname)_$(f.feat)_win_$(f.nwin)" for f in ds.info]
end
