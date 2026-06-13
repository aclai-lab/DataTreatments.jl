# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const TreatType = Dict(
    :discrete => T -> !(T <: Float) && !(T <: AbstractArray),
    :continuous => T -> T <: Float,
    :multidim => T -> T <: AbstractArray,
    :all => T -> true
)

# ---------------------------------------------------------------------------- #
#                            TreatmentGroup struct                             #
# ---------------------------------------------------------------------------- #
"""
    TreatmentGroup

A user-defined directive that specifies **which columns to select**
from a dataset and **how to process** them inside
[`load_dataset`](@ref).

Each `TreatmentGroup` encodes two orthogonal concerns:

1. **Column selection** — filter columns by name, dimensionality,
   or data type.
2. **Processing directives** — how to handle multidimensional
   columns, missing values, normalization, and output grouping.

Multiple `TreatmentGroup`s can be passed to `load_dataset`; each
produces its own set of [`AbstractDataset`](@ref) entries inside
the resulting [`DataTreatment`](@ref).

---

## Column Selection

Columns are included when **all** active filters match:

| Field       | Type                                    | Default  |
|:------------|:----------------------------------------|:---------|
| `dims`      | `Int`                                   | `-1`     |
| `name_expr` | `Regex` / `Base.Callable` / `Vector`    | `r".*"`  |
| `datatype`  | `Symbol`                                | `:all`   |

- `dims`: Keep only columns whose array length equals `dims`.
  `-1` disables the filter.
- `name_expr`: Keep columns whose name matches the regex,
  satisfies the predicate, or is in the string vector.
- `datatype`: One of `:discrete`, `:continuous`, `:multidim`,
  or `:all`.

```
All columns in the dataset
        │
        ├─ dims filter        (skip if dims == -1)
        ├─ datatype filter    (skip if datatype == :all)
        └─ name_expr filter   (skip if name_expr == r".*")
                │
                ▼
        ids ::Vector{Int}
```

---

## Processing Directives

| Field      | Type                                         | Default  |
|:-----------|:---------------------------------------------|:---------|
| `aggrfunc` | `Base.Callable`                              | see below|
| `grouped`  | `Bool`                                       | `false`  |
| `groupby`  | `Nothing` / `Tuple{Vararg{Symbol}}`          | `nothing`|
| `impute`   | `Nothing` / `Tuple{Vararg{Imputor}}`         | `nothing`|
| `norm`     | `Nothing` / `Type{<:AbstractNormalization}`  | `nothing`|

- `aggrfunc`: Applied to multidimensional columns.
  - [`aggregate`](@ref)`(...)` → scalar tabular output.
  - [`reducesize`](@ref)`(...)` → smaller array output.
  - Default: `aggregate(win=(wholewindow(),),
    features=(maximum, minimum, mean))`.
- `grouped`: If `true`, all selected columns are processed
  jointly rather than independently.
- `groupby`: Partition output features by `:vname`, window
  index, or feature type. Accepts a single `Symbol` or a
  `Tuple{Vararg{Symbol}}`.
- `impute`: Imputation chain applied after encoding or
  aggregation (e.g. `(LOCF(), NOCB())`).
- `norm`: Normalization type applied to the output matrix
  (e.g. `MinMax`, `ZScore`).

---

## Fields

```
TreatmentGroup
├── ids      ::Vector{Int}
├── dims     ::Int
├── vnames   ::Vector{String}
├── aggrfunc ::Base.Callable
├── grouped  ::Bool
├── groupby  ::Union{Nothing,Tuple{Vararg{Symbol}}}
├── impute   ::Union{Nothing,Tuple{Vararg{Imputor}}}
├── norm     ::Union{Nothing,Type{<:AbstractNormalization}}
└── datatype ::Type
```

---

## Constructors

### Direct constructor

```julia
TreatmentGroup(datastruct, vnames; kwargs...)
```

Called internally by `load_dataset`. Runs the three column
filters on `datastruct` (output of `_inspecting`) and builds
the group.

### Curried constructor

```julia
TreatmentGroup(; kwargs...)
```

Returns a `(datastruct, vnames) -> TreatmentGroup` closure.
This is the form expected by `load_dataset`:

```julia
dt = load_dataset(df, y, TreatmentGroup(datatype=:continuous))
```

---

## Output mapping

```
TreatmentGroup
        │
        ├─ discrete columns
        │    └─▶ DiscreteDataset{T}
        │
        ├─ continuous columns
        │    └─▶ ContinuousDataset{T}
        │
        └─ multidim columns
             ├─ aggrfunc = aggregate(...)
             │    └─▶ MultidimDataset{T, AggregateFeat}
             └─ aggrfunc = reducesize(...)
                  └─▶ MultidimDataset{T, ReduceFeat}
```

---

## Examples

```julia
# normalize all continuous columns with MinMax
t1 = TreatmentGroup(datatype=:continuous, norm=MinMax)

# aggregate signal columns with mean/std over 3 sliding windows
t2 = TreatmentGroup(
    name_expr = r"^signal_",
    datatype  = :multidim,
    aggrfunc  = aggregate(
        win      = slidingwindows(3),
        features = (mean, std),
    ),
    groupby = :vname,
)

# downsample audio columns to 256 points
t3 = TreatmentGroup(
    name_expr = r"^audio_",
    datatype  = :multidim,
    aggrfunc  = reducesize(256),
)

dt = load_dataset(df, target, t1, t2, t3)
```

# See Also
[`load_dataset`](@ref), [`DataTreatment`](@ref),
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`MultidimDataset`](@ref), [`aggregate`](@ref),
[`reducesize`](@ref)
"""
struct TreatmentGroup
    ids::Vector{Int}
    dims::Int
    vnames::Vector{String}
    aggrfunc::Base.Callable
    grouped::Bool
    groupby::Union{Nothing,Tuple{Vararg{Symbol}}}
    impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}}
    norm::Union{Nothing,Type{<:AbstractNormalization}}
    datatype::Type

    function TreatmentGroup(
        datastruct::NamedTuple,
        vnames::Vector{String};
        dims::Int=-1,
        name_expr::Union{Regex,Base.Callable,Vector{String}}=r".*",
        aggrfunc::F=aggregate(
            win=(wholewindow(),), features=(maximum, minimum, mean)),
        grouped::Bool=false,
        groupby::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}=nothing,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}}=nothing,
        norm::Union{Nothing,Type{<:AbstractNormalization}}=nothing,
        datatype::Symbol=:all
    ) where {F<:Base.Callable}
        all_dims = datastruct.dims
        all_types = datastruct.datatype
        groupby isa Symbol && (groupby = (groupby,))

        name_match = if name_expr isa Regex
            n -> match(name_expr, n) !== nothing
        elseif name_expr isa Vector{String}
            name_set = Set(name_expr)
            n -> n in name_set
        else
            name_expr
        end

        ids = Int[]

        for i in datastruct.id
            (dims != -1 && all_dims[i] != dims) && continue
            !TreatType[datatype](all_types[i]) && continue
            name_match(vnames[i]) || continue
            push!(ids, i)
        end

        col_types = all_types[ids]
        t = isempty(col_types) ? Any : mapreduce(identity, typejoin, col_types)
        isconcretetype(t) || (t = Any)

        new(ids, dims, vnames[ids], aggrfunc, grouped, groupby, impute, norm, t)
    end
end

TreatmentGroup(; kwargs...) = (d, v) -> TreatmentGroup(d, v; kwargs...)

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
get_ids(t::Vector{<:TreatmentGroup}) = get_ids.(t)
get_ids(t::TreatmentGroup) = t.ids

get_aggrfunc(t::TreatmentGroup) = t.aggrfunc

has_groupby(t::TreatmentGroup) = !isnothing(t.groupby)
get_groupby(t::TreatmentGroup) = t.groupby

get_impute(t::TreatmentGroup) = t.impute
get_norm(t::TreatmentGroup) = t.norm
