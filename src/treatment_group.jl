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

A user-defined directive that specifies **which columns to select** from a dataset
and **how to process** them inside `load_dataset`.

Each `TreatmentGroup` encodes two orthogonal concerns:

1. **Column selection** — filter columns by name, dimensionality, or data type.
2. **Processing directives** — how to handle multidimensional columns, missing
   values, normalization, and output grouping.

Multiple `TreatmentGroup`s can be passed to `load_dataset`; each produces its own
set of [`AbstractDataset`](@ref) entries inside the resulting [`DataTreatment`](@ref).

---

## Column Selection

Columns are included when **all** active filters match:

| Field        | Type                                          | Effect                                             |
|--------------|-----------------------------------------------|----------------------------------------------------|
| `dims`       | `Int`                                         | Keep only columns whose array length equals `dims`; `-1` disables this filter |
| `name_expr`  | `Regex` / `Base.Callable` / `Vector{String}`  | Keep columns whose name matches the pattern/predicate/list |
| `datatype`   | `Symbol`                                      | `:discrete`, `:continuous`, `:multidim`, or `:all` |

```
All columns in the dataset
        │
        ├─ dims filter        (skip if dims == -1)
        ├─ datatype filter    (skip if datatype == :all)
        └─ name_expr filter   (skip if name_expr == r".*")
                │
                ▼
        ids ::Vector{Int}   ← indices of the surviving columns
```

---

## Processing Directives

Once columns are selected, the following fields govern how they are processed:

| Field      | Type                                        | Effect                                                                                                    |
|------------|---------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `aggrfunc` | `Base.Callable`                             | Applied to multidimensional columns: `aggregate(...)` → scalar matrix; `reducesize(...)` → smaller arrays |
| `grouped`  | `Bool`                                      | `false` (default): process each column independently; `true`: process all selected columns jointly        |
| `groupby`  | `Nothing` / `Tuple{Vararg{Symbol}}`         | Partition output features by `:vname`, window index, or feature type                                      |
| `impute`   | `Nothing` / `Tuple{Vararg{Imputor}}`        | Imputation strategy applied after encoding/aggregation                                                    |
| `norm`     | `Nothing` / `Type{<:AbstractNormalization}` | Normalization applied to the output matrix                                                                |

---

## Fields

```
TreatmentGroup
├── ids       ::Vector{Int}                                  # selected column indices
├── dims      ::Int                                          # dimensionality filter used
├── vnames    ::Vector{String}                               # names of selected columns
├── aggrfunc  ::Base.Callable                                # multidim processing strategy
├── grouped   ::Bool                                         # joint vs columnwise processing
├── groupby   ::Union{Nothing,Tuple{Vararg{Symbol}}}         # output feature partitioning
├── impute    ::Union{Nothing,Tuple{Vararg{Imputor}}}        # imputation strategy
├── norm      ::Union{Nothing,Type{<:AbstractNormalization}} # normalization
└── datatype  ::Type                                         # typejoin of selected column types
```

---

## Usage

`TreatmentGroup` is most conveniently created in **curried form** and passed
directly to [`load_dataset`](@ref):

```julia
# Select all continuous columns and normalize them
t1 = TreatmentGroup(datatype=:continuous, norm=MinMaxNormalization)

# Select time-series columns matching "signal_*", aggregate with mean/std over 3 windows
t2 = TreatmentGroup(
    name_expr  = r"^signal_",
    datatype   = :multidim,
    aggrfunc   = aggregate(win=slidingwindows(3), features=(mean, std)),
    groupby    = :vname
)

# Select audio columns and downsample to 256 points
t3 = TreatmentGroup(
    name_expr = r"^audio_",
    datatype  = :multidim,
    aggrfunc  = reducesize(256)
)

dt = load_dataset(df, target, t1, t2, t3)
```

---

## How Each Group Maps to Output Datasets

```
TreatmentGroup (user directive)
        │
        ├─ selected discrete columns
        │       └─▶ DiscreteDataset{T}
        │
        ├─ selected continuous columns
        │       └─▶ ContinuousDataset{T}
        │
        └─ selected multidim columns
                ├─ aggrfunc = aggregate(...)
                │       └─▶ MultidimDataset{T, AggregateFeat}  (tabular)
                └─ aggrfunc = reducesize(...)
                        └─▶ MultidimDataset{T, ReduceFeat}     (array)
```

The curried form `TreatmentGroup(; kwargs...)` returns a `(datastruct, vnames) -> TreatmentGroup`
closure, which is the expected input signature for `load_dataset`.

See also: [`load_dataset`](@ref), [`DataTreatment`](@ref),
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref), [`MultidimDataset`](@ref),
[`aggregate`](@ref), [`reducesize`](@ref)
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
        aggrfunc::F=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
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