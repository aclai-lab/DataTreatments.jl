# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
"""
    DiscreteFeat{T} <: AbstractDataFeature

Metadata for a **discrete (categorical)** feature in a dataset.

## Structure

```
DiscreteFeat{T}
├── id          ::Int           # column index in the source data
├── vname       ::String        # original column name
├── valididxs   ::Vector{Int}   # row indices with valid (non-missing) values
├── missingidxs ::Vector{Int}   # row indices with missing values
└── datatype    ::Type          # original column datatype
```

## Example

Given a column `"color"` at position `3` in the source data with 10 rows,
where rows 4 and 7 are missing:

```
Source data column "color" (id=3)
Row:  1       2       3       4        5       6       7        8       9       10
Val: "red"  "blue" "green" missing  "red"  "blue"  missing  "green" "red"  "blue"

→ id          = 3
→ vname       = "color"
→ valididxs   = [1, 2, 3, 5, 6, 8, 9, 10]
→ missingidxs = [4, 7]
→ datatype    = String
```
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

## Structure

```
ContinuousFeat{T}
├── id          ::Int           # column index in the source data
├── vname       ::String        # original column name
├── valididxs   ::Vector{Int}   # row indices with valid (non-missing, non-NaN) values
├── missingidxs ::Vector{Int}   # row indices with missing values
└── nanidxs     ::Vector{Int}   # row indices with NaN values
```

## Example

Given a column `"temperature"` at position `2` in the source data with 8 rows,
where row 3 is missing and row 6 contains NaN:

```
Source data column "temperature" (id=2)
Row:  1      2      3        4      5      6      7      8
Val: 22.1   19.5  missing  24.0   21.3   NaN   20.8   23.1

→ id          = 2
→ vname       = "temperature"
→ valididxs   = [1, 2, 4, 5, 7, 8]
→ missingidxs = [3]
→ nanidxs     = [6]
```
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

Metadata for an **aggregated scalar** feature derived from a multidimensional column
(e.g., a time series) via a feature function applied over a sliding window.

## Structure

```
AggregateFeat{T}
├── id          ::Int            # source column index in the original data
├── subid       ::Int            # position of this feature in the flattened output
├── vname       ::String         # original column name
├── dims        ::Int            # dimensionality of the source arrays
├── feat        ::Base.Callable  # aggregation function (e.g., mean, std)
├── nwin        ::Int            # window index this feature was computed on
├── valididxs   ::Vector{Int}    # row indices with valid entries
├── missingidxs ::Vector{Int}    # row indices where the cell is missing
├── nanidxs     ::Vector{Int}    # row indices where the scalar result is NaN
├── hasmissing  ::Vector{Int}    # row indices where the source array has internal `missing`
└── hasnans     ::Vector{Int}    # row indices where the source array has internal `NaN`
```

## Example

A column `"signal"` (id=1) contains time series of length 100.
We apply `[mean, std]` over `2` sliding windows → 4 output columns:

```
Source column "signal" (id=1, dims=100)
       ┌───────────────────────────────────────────────┐
Row 1: │ [0.1, 0.3, ..., 0.9]     (length 100)         │
Row 2: │ [0.2, missing, ..., 0.7] (has internal miss)  │
Row 3: │ missing                  (whole cell missing) │
       └───────────────────────────────────────────────┘
           │
           ▼  apply [mean, std] × 2 windows
┌──────────────────────────────────────────────────────┐
│  subid=1         subid=2         subid=3   subid=4   │
│  feat=mean,win=1 feat=std,win=1  mean,win2 std,win2  │
│  id=1            id=1            id=1      id=1      │
└──────────────────────────────────────────────────────┘
```
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

Metadata for a **reduced-size** feature derived from a multidimensional column.
Unlike [`AggregateFeat`](@ref), the output preserves the array structure but
reduces its size (e.g., downsampling a 10 000-point time series to 256 points).

## Structure

```
ReduceFeat{T}
├── id          ::Int            # source column index in the original data
├── vname       ::String         # original column name
├── dims        ::Int            # dimensionality of the source arrays
├── reducefunc  ::Base.Callable  # reduction/downsampling function
├── valididxs   ::Vector{Int}    # row indices with valid entries
├── missingidxs ::Vector{Int}    # row indices where the cell is missing
├── nanidxs     ::Vector{Int}    # row indices where the reduced result has NaN
├── hasmissing  ::Vector{Int}    # row indices where source array has internal `missing`
└── hasnans     ::Vector{Int}    # row indices where source array has internal `NaN`
```

## Example

```
Source column "audio" (id=5, dims=10000)
       ┌────────────────────────────────────────────┐
Row 1: │ [0.1, 0.2, ..., 0.9]  (length 10 000)      │
Row 2: │ [0.3, NaN, ..., 0.5]  (has internal NaN)   │
Row 3: │ missing               (whole cell missing) │
       └────────────────────────────────────────────┘
           │
           ▼  apply reducefunc (e.g., downsample to 256)
       ┌────────────────────────────────────────────┐
Row 1: │ [0.12, 0.34, ..., 0.88]  (length 256)      │  ← valid
Row 2: │ [0.31, 0.49, ..., 0.50]  (length 256)      │  ← hasnans=[2]
Row 3: │ missing                                    │  ← missingidxs=[3]
       └────────────────────────────────────────────┘
```
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

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
_get_features(a::Base.Callable) = a.features
_get_reducefunc(r::Base.Callable) = r.reducefunc

"""
    _reindex_groups(groups, idxs) -> Union{Nothing, Vector{Vector{Int}}}

Re-map group indices from the original column space to the new subset `idxs`.
Only groups that have at least one member in `idxs` are kept.
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

# ---------------------------------------------------------------------------- #
#                           output dataset structs                             #
# ---------------------------------------------------------------------------- #
"""
    DiscreteDataset{T} <: AbstractDataset

Output dataset for **discrete (categorical)** columns collected by `DataTreatment`.

## Structure

```
DiscreteDataset{T}
├── data  ::AbstractMatrix          # integer-coded matrix (nrows × nfeatures)
│                                   # values are Int level codes or missing
└── info  ::Vector{<:DiscreteFeat}  # one DiscreteFeat entry per column
```

## Layout

```
Source DataFrame (mixed types)
 ┌──────────┬──────────┬──────────┐
 │ color    │ shape    │ size     │  ← discrete columns selected (ids=[1,2,3])
 ├──────────┼──────────┼──────────┤
 │ "red"    │ "circle" │ "small"  │
 │ "blue"   │ missing  │ "large"  │
 │ "red"    │ "square" │ "small"  │
 └──────────┴──────────┴──────────┘
         │
         ▼  _discrete_encode
 ┌──────────┬──────────┬──────────┐
 │ col_1    │ col_2    │ col_3    │   data::AbstractMatrix
 ├──────────┼──────────┼──────────┤   (integer-coded, Union{Int,Missing})
 │    2     │    1     │    2     │
 │    1     │  miss.   │    1     │
 │    2     │    3     │    2     │
 └──────────┴──────────┴──────────┘

 info = [DiscreteFeat(id=1, vname="color",  ...),
         DiscreteFeat(id=2, vname="shape",  ...),
         DiscreteFeat(id=3, vname="size",   ...)]
```

# Constructors

- `DiscreteDataset(data, info)`: Direct constructor from pre-built matrix and metadata.
- `DiscreteDataset(ids, data, vnames, datastruct, impute)`:
  Internal constructor — selects columns `ids` from `data`, encodes them
  categorically, optionally applies `impute`, and builds
  [`DiscreteFeat`](@ref) metadata from `datastruct`.

## Arguments for internal constructor
- `ids::Vector{Int}`: Column indices to include.
- `data::AbstractMatrix`: Full raw dataset matrix.
- `vnames::Vector{String}`: Names of all columns.
- `datastruct::NamedTuple`: Pre-computed dataset metadata from `_inspecting`.
- `impute`: `nothing` or a tuple of `Impute.Imputor`s to fill missing values.

See also: [`ContinuousDataset`](@ref), [`MultidimDataset`](@ref), [`DiscreteFeat`](@ref)
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

Output dataset for **continuous (numeric scalar)** columns produced by `DataTreatment`.

## Structure

```
ContinuousDataset{T}
├── data  ::AbstractMatrix            # float matrix (nrows × nfeatures)
│                                     # values are Union{Missing, T}
└── info  ::Vector{<:ContinuousFeat}  # one ContinuousFeat entry per column
```

## Layout

```
Source DataFrame (mixed types)
 ┌────────────┬──────────┬───────────┐
 │temperature │ pressure │ humidity  │  ← continuous columns (ids=[4,5,6])
 ├────────────┼──────────┼───────────┤
 │  22.1      │  1013.0  │   55.3    │
 │  missing   │  987.5   │   NaN     │
 │  19.8      │  1001.2  │   60.1    │
 └────────────┴──────────┴───────────┘
         │
         ▼  cast to float_type (e.g. Float64)
 ┌────────────┬──────────┬───────────┐
 │  col_4     │  col_5   │  col_6    │   data::Matrix{Union{Missing,Float64}}
 ├────────────┼──────────┼───────────┤
 │  22.1      │  1013.0  │   55.3    │
 │  missing   │   987.5  │   NaN     │
 │  19.8      │  1001.2  │   60.1    │
 └────────────┴──────────┴───────────┘
         │
         ▼  normalize! (if norm != nothing)  e.g. NaNSafe{MinMaxNormalization{Float64}}
 ┌────────────┬──────────┬───────────┐
 │  col_4     │  col_5   │  col_6    │   data::Matrix{Float64}  (column-wise)
 ├────────────┼──────────┼───────────┤
 │   0.82     │   1.00   │   0.72    │
 │  missing   │   0.00   │   NaN     │
 │   0.00     │   0.54   │   1.00    │
 └────────────┴──────────┴───────────┘

 info = [ContinuousFeat(id=4, vname="temperature", nanidxs=[],  missingidxs=[2], ...),
         ContinuousFeat(id=5, vname="pressure",    nanidxs=[],  missingidxs=[],  ...),
         ContinuousFeat(id=6, vname="humidity",    nanidxs=[2], missingidxs=[],  ...)]
```

!!! note "Normalization order"
    Imputation (if any) is applied **before** normalization.
    NaN values produced by missing imputation are handled transparently via
    `NaNSafe{norm{T}}`, which ignores NaNs when computing statistics.

# Type Parameter
- `T`: The floating-point type used for numeric conversion (e.g., `Float64`, `Float32`).

# Constructors

- `ContinuousDataset(data, info)`: Direct constructor from pre-built matrix and metadata.
- `ContinuousDataset(ids, data, vnames, datastruct, impute, norm, float_type)`:
  Internal constructor — selects columns `ids` from `data`, converts each element
  to `float_type` (preserving `missing`), optionally applies `impute` and/or
  `norm`, and builds [`ContinuousFeat`](@ref) metadata from `datastruct`.

## Arguments for internal constructor
- `ids::Vector{Int}`: Column indices to include.
- `data::AbstractMatrix`: Full raw dataset matrix.
- `vnames::Vector{String}`: Names of all columns.
- `datastruct::NamedTuple`: Pre-computed dataset metadata from `_inspecting`.
- `impute`: `nothing` or a tuple of `Impute.Imputor`s to fill missing values.
- `norm`: `nothing` or a `Type{<:AbstractNormalization}` (e.g., `MinMax`,
  `ZScore`) applied column-wise after imputation. NaN values
  are handled via `NaNSafe{norm{T}}`.
- `float_type::Type`: Target floating-point type for numeric conversion.

See also: [`DiscreteDataset`](@ref), [`MultidimDataset`](@ref), [`ContinuousFeat`](@ref)
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

Output dataset for **multidimensional** columns (e.g., time series, spectrograms)
produced by `DataTreatment`. The layout depends on the chosen aggregation strategy:

---

## Strategy 1 — `aggregate` → `S = AggregateFeat`

Each multidimensional element is condensed into multiple **scalar columns**
via feature functions applied over sliding windows.

```
Source column "signal" (dims=1000)
 ┌────────────────────────────────────┐
 │ Row 1: [0.1, 0.2, ..., 0.9]        │
 │ Row 2: [0.3, 0.4, ..., 0.8]        │
 │ Row 3: missing                     │
 └────────────────────────────────────┘
         │
         ▼  features=[mean, std], nwindows=2
 ┌──────────┬─────────┬──────────┬─────────┐
 │mean,win1 │std,win1 │mean,win2 │std,win2 │  data::Matrix{Float64}
 ├──────────┼─────────┼──────────┼─────────┤  (tabular, nrows × n_feats×nwins)
 │  0.312   │  0.098  │  0.601   │  0.112  │
 │  0.421   │  0.077  │  0.699   │  0.085  │
 │ missing  │ missing │ missing  │ missing │
 └──────────┴─────────┴──────────┴─────────┘
         │
         ▼  normalize! (if norm != nothing)  e.g. NaNSafe{MinMaxNormalization{Float64}}
 ┌──────────┬─────────┬──────────┬─────────┐
 │mean,win1 │std,win1 │mean,win2 │std,win2 │  (column-wise normalization)
 ├──────────┼─────────┼──────────┼─────────┤
 │  0.00    │  0.28   │  0.00    │  0.00   │
 │  1.00    │  0.00   │  1.00    │  1.00   │
 │ missing  │ missing │ missing  │ missing │
 └──────────┴─────────┴──────────┴─────────┘
 info = [AggregateFeat(subid=1, feat=mean, nwin=1, ...),
         AggregateFeat(subid=2, feat=std,  nwin=1, ...),
         AggregateFeat(subid=3, feat=mean, nwin=2, ...),
         AggregateFeat(subid=4, feat=std,  nwin=2, ...)]
```

---

## Strategy 2 — `reducesize` → `S = ReduceFeat`

Each multidimensional element is **downsampled** to a smaller array,
preserving the array structure.

```
Source column "audio" (dims=10000)
 ┌────────────────────────────────────────┐
 │ Row 1: [0.1, 0.2, ..., 0.9]  (10000)   │
 │ Row 2: [0.3, NaN, ..., 0.8]  (10000)   │
 │ Row 3: missing                         │
 └────────────────────────────────────────┘
         │
         ▼  reducefunc = downsample(256)
 ┌────────────────────────────────────────┐
 │ Row 1: [0.12, 0.34, ..., 0.88]  (256)  │  data::Matrix{AbstractArray{Float64}}
 │ Row 2: [0.31, 0.49, ..., 0.50]  (256)  │
 │ Row 3: missing                         │
 └────────────────────────────────────────┘
         │
         ▼  normalize! (if norm != nothing)  applied element-wise per array
 ┌────────────────────────────────────────┐
 │ Row 1: [0.00, 0.23, ..., 1.00]  (256)  │
 │ Row 2: [0.00, 0.19, ..., 0.96]  (256)  │
 │ Row 3: missing                         │
 └────────────────────────────────────────┘
 info = [ReduceFeat(id=5, vname="audio", dims=10000, reducefunc=..., ...)]
```

---

## Structure

```
MultidimDataset{T, S}
├── data    ::AbstractArray                              # processed data matrix
├── info    ::Vector{<:Union{AggregateFeat,ReduceFeat}}  # per-column metadata
└── groups  ::Union{Nothing, Vector{Vector{Int}}}        # optional column groupings
             │
             └─ e.g. groups = [[1,2,3,4], [5,6,7,8]]
                means columns 1-4 belong to group 1 (source col A),
                and columns 5-8 belong to group 2 (source col B)
```

!!! note "Normalization order"
    Processing steps are applied in this order:
    1. `aggrfunc`  — aggregation or downsampling
    2. `impute`    — fill missing values (if provided)
    3. `norm`      — column-wise normalization via `NaNSafe{norm{T}}` (if provided)

# Type Parameters
- `T`: Element type of the inner arrays (e.g., `Float64`).
- `S`: Either `AggregateFeat` (tabular output) or `ReduceFeat` (array output).

# Constructors

- `MultidimDataset(data, info::Vector{<:AggregateFeat}, groups)`: Direct constructor for aggregated data.
- `MultidimDataset(data, info::Vector{<:ReduceFeat}, groups)`: Direct constructor for reduced data.
- `MultidimDataset(ids, data, vnames, datastruct, aggrfunc, impute, norm, float_type, groups)`:
  Internal constructor — selects columns `ids`, applies `aggrfunc`, optionally
  applies `impute` and/or `norm`, and builds the corresponding metadata from `datastruct`.

## Arguments for internal constructor
- `ids::Vector{Int}`: Column indices to include.
- `data::Matrix`: Full raw dataset matrix.
- `vnames::Vector{String}`: Names of all columns.
- `datastruct::NamedTuple`: Pre-computed metadata from `_inspecting`.
- `aggrfunc::Base.Callable`: Strategy struct (e.g., `aggregate(...)` or `reducesize(...)`).
- `impute`: `nothing` or a tuple of `Impute.Imputor`s.
- `norm`: `nothing` or a `Type{<:AbstractNormalization}` (e.g., `MinMax`,
  `ZScore`) applied column-wise after imputation via `NaNSafe{norm{T}}`.
- `float_type::Type{T}`: Target floating-point type.
- `groups`: `nothing` to disable grouping, or a `Tuple{Vararg{Symbol}}` to enable it.

See also: [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`AggregateFeat`](@ref), [`ReduceFeat`](@ref), [`aggregate`](@ref), [`reducesize`](@ref)
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
        data::Matrix,
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
                ((c, f, n) for f in _get_features(aggrfunc) for n in 1:nwindows[c])
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

get_vnames(d::AbstractDataset)::Vector{String} = get_vnames.(d.info)

function get_vnames(
    ds::MultidimDataset{<:Any,<:AggregateFeat};
    groupby_split::Bool=false
)
    names =
        ["$(f.vname),$(f.feat),win:$(f.nwin)" for f in ds.info]
    # groupby_split && has_groups(ds) ?
    #     [names[g] for g in get_groups(ds)] :
    #     names
end

