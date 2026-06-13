# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
"""
    DataTreatment{T}

Top-level container produced by [`load_dataset`](@ref). Holds all
output datasets derived from a source table, a target vector, and
the applied treatment groups.

## Structure

```
DataTreatment{T}  (T = float_type, e.g. Float64)
├── data    ::Vector{AbstractDataset}
│    ├── DiscreteDataset{...}       # from discrete (categorical) cols
│    ├── ContinuousDataset{T}       # from continuous (scalar) cols
│    └── MultidimDataset{T, ...}    # from multidimensional cols
│         ├── AggregateFeat         # → tabular scalar output
│         └── ReduceFeat            # → array output
├── target  ::AbstractVector        # encoded target vector
├── treats  ::Vector{TreatmentGroup}# user directives
└── balance ::Union{Nothing,        # balancing strategy or nothing
                    AbstractBalance,
                    Tuple{Vararg{<:AbstractBalance}}}
```

## Full pipeline overview

```
Raw DataFrame / Matrix
        │
        ▼  load_dataset(data, vnames, target, treatments...;
        │               float_type, balance)
        │
        ├─ _inspecting(data)  →  datastruct::NamedTuple
        │   (inspect columns: types, missing, NaN, dims, ...)
        │
        ├─ encode target  →  CategoricalVector
        │
        ├─ for each TreatmentGroup:
        │   ├── classify columns
        │   │    → discrete_ids, continuous_ids, multidim_ids
        │   ├── DiscreteDataset(discrete_ids, ...)
        │   ├── ContinuousDataset(continuous_ids, ...)
        │   └── MultidimDataset(multidim_ids, ..., aggrfunc)
        │
        ├─ if balance is provided:
        │   │  (single AbstractBalance wrapped into a 1-tuple)
        │   └── for each dataset d in ds:
        │        reduce over balance chain:
        │          (d.data, target) = bal₁ ∘ bal₂ ∘ … ∘ balₙ
        │
        └── DataTreatment{T}(ds, target, treats, balance)
```

## Accessor summary

| Method               | Returns                                    |
|:---------------------|:-------------------------------------------|
| `get_discrete(dt)`   | `(Matrix, vnames)` for categorical cols    |
| `get_continuous(dt)` | `(Matrix{T}, vnames)` for scalar cols      |
| `get_aggregated(dt)` | `(Matrix{T}, vnames)` for aggregated series|
| `get_reduced(dt)`    | `(Matrix{Array{T}}, vnames)` for series    |
| `get_tabular(dt)`    | merged tabular matrix + all column names   |
| `get_multidim(dt)`   | reduced multidim matrix + column names     |
| `get_target(dt)`     | the encoded target vector                  |
| `get_treats(dt)`     | the applied `TreatmentGroup` list          |
| `get_balance(dt)`    | the balancing strategy or `nothing`        |

## Reprocessing

A fitted `DataTreatment` can be re-treated by passing it back to
`load_dataset`, which reconstructs the raw matrix from all stored
datasets and applies a new set of treatment groups:

```julia
dt2 = load_dataset(dt, new_treatment)
```

# Type Parameter
- `T`: Floating-point type used throughout (e.g. `Float64`, `Float32`).

# Fields
- `data`: Ordered list of output datasets (discrete, continuous,
  multidimensional).
- `target`: Encoded target vector (labels).
- `treats`: User-specified treatment groups that drove column
  classification.
- `balance`: Optional balancing strategy, or `nothing`. Can be a
  single `AbstractBalance` or a `Tuple{Vararg{<:AbstractBalance}}`
  to chain multiple strategies sequentially (e.g.
  `(SMOTE(k=5), TomekUndersampler())`). When a tuple is provided,
  each balancer is applied in order, passing the `(data, target)`
  output of one step as input to the next.

# See Also
[`load_dataset`](@ref), [`TreatmentGroup`](@ref),
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`MultidimDataset`](@ref), [`AbstractBalance`](@ref)
"""
mutable struct DataTreatment{T}
    data::Vector{AbstractDataset}
    target::AbstractVector
    treats::Vector{TreatmentGroup}
    balance::Union{Nothing,AbstractBalance,Tuple{Vararg{<:AbstractBalance}}}
end

nrows(dt::DataTreatment) = size(first(dt.data).data, 1)
ncols(dt::DataTreatment) = size(first(dt.data).data, 2)

get_target(dt::DataTreatment) = dt.target
get_treats(dt::DataTreatment) = dt.treats
get_balance(dt::DataTreatment) = dt.balance

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
"""
    load_dataset(data, [vnames], [target], treatments...;
                 float_type=Float64, balance=nothing,
                 treatment_ds=true, leftover_ds=false)
    load_dataset(df::DataFrame, [target], treatments...; kwargs...)
    load_dataset(dt::DataTreatment, treatments...; kwargs...)

Build a [`DataTreatment`](@ref) from a raw data source by inspecting
columns, applying treatment groups, and optionally rebalancing classes.

---

## Dispatch 1 — Matrix source

```julia
load_dataset(
    data      :: AbstractMatrix,
    vnames    :: Vector{String}  = ["V1", "V2", ...],
    target    :: Union{Nothing, AbstractVector} = nothing,
    treatments:: Vararg{Base.Callable} = DefaultTreatmentGroup;
    float_type :: Type   = Float64,
    balance    :: Union{Nothing, AbstractBalance,
                        Tuple{Vararg{<:AbstractBalance}}} = nothing,
    treatment_ds :: Bool = true,
    leftover_ds  :: Bool = false,
)
```

Core method. All other dispatches delegate here.

### Arguments
- `data`: Input matrix. Columns may be scalar, categorical, or
  array-valued (time series).
- `vnames`: Column names. Defaults to `["V1", "V2", ...]`.
- `target`: Optional label vector. Categorical labels are
  automatically encoded via `_discrete_encode`. Float targets are
  passed through unchanged. Pass `nothing` (default) for
  unsupervised use.
- `treatments`: One or more [`TreatmentGroup`](@ref) callables that
  classify and process columns. Defaults to
  `DefaultTreatmentGroup`.

### Keyword Arguments
- `float_type::Type=Float64`: Floating-point type for all numeric
  output (`Float32` or `Float64`).
- `balance`: Rebalancing strategy applied after dataset construction.
  Accepts:
  - `nothing` (default) — no rebalancing.
  - A single `AbstractBalance` (e.g. `SMOTE()`).
  - A `Tuple` of `AbstractBalance` objects applied in order:
    `(SMOTE(), TomekUndersampler())`.
  Requires a non-empty `target`.
- `treatment_ds::Bool=true`: If `true`, build datasets from columns
  matched by the treatment groups.
- `leftover_ds::Bool=false`: If `true`, also build datasets from
  columns not matched by any treatment group.

---

## Dispatch 2 — DataFrame source

```julia
load_dataset(df::DataFrame, target::AbstractVector, args...; kwargs...)
load_dataset(df::DataFrame, args...; kwargs...)
```

Convenience wrappers that extract `Matrix(df)` and `names(df)` and
forward to Dispatch 1. The targetless overload passes an empty
`CategoricalVector[]` as `target`.

### Arguments
- `df`: Source `DataFrame`. Column names are taken from `names(df)`.
- `target`: Label vector (optional). Omit for unsupervised use.
- `args...`, `kwargs...`: Forwarded to Dispatch 1.

---

## Dispatch 3 — Re-treat an existing DataTreatment

```julia
load_dataset(dt::DataTreatment, treatments...; kwargs...)
```

Reconstructs the raw matrix from all datasets stored in `dt` and
applies a fresh set of treatment groups. Useful for re-processing
already-treated data with different parameters.

Internally:
```
data   = hcat(d.data   for d in dt.data)
vnames = vcat(vnames(d) for d in dt.data)
target = get_target(dt)
→ load_dataset(data, vnames, target, treatments...; kwargs...)
```

### Arguments
- `dt`: A previously built [`DataTreatment`](@ref).
- `treatments`: New treatment group callables.
- `kwargs...`: Forwarded to Dispatch 1.

---

## Pipeline summary

```
data / df / dt
      │
      ▼  _inspecting(data)
      │   → per-column metadata (type, missing, NaN, dims)
      │
      ▼  encode target  →  CategoricalVector
      │
      ▼  for each TreatmentGroup:
      │   ├─ classify → discrete_ids, continuous_ids, multidim_ids
      │   ├─ DiscreteDataset(...)
      │   ├─ ContinuousDataset(...)
      │   └─ MultidimDataset(..., aggrfunc)
      │
      ▼  if balance provided:
      │   └─ Threads.@threads for each dataset:
      │       (data, target) = bal₁ ∘ … ∘ balₙ(data, target)
      │
      └─ DataTreatment{float_type}(ds, target, treats, balance)
```

## Returns
- [`DataTreatment{T}`](@ref) where `T = float_type`.

## Examples

```julia
# from a DataFrame with classification target
dt = load_dataset(df, y, TreatmentGroup(dims=0))

# from a DataFrame, chain oversampling + undersampling
dt = load_dataset(df, y;
    balance=(SMOTE(k=5), TomekUndersampler()))

# re-treat with different parameters
dt2 = load_dataset(dt, TreatmentGroup(dims=1, impute=(LOCF(),)))
```

## See Also
[`DataTreatment`](@ref), [`TreatmentGroup`](@ref),
[`AbstractBalance`](@ref)
"""
function load_dataset(
    data::AbstractMatrix{T},
    vnames::Vector{String}=["V$i" for i in 1:size(data, 2)],
    target::Union{Nothing,AbstractVector}=nothing,
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    balance::Union{
        Nothing,AbstractBalance,Tuple{Vararg{<:AbstractBalance}}}=nothing,
    treatment_ds::Bool=true,
    leftover_ds::Bool=false,
    float_type::Type=Float64
) where T
    datastruct = _inspecting(data)

    if isnothing(target)
        target = CategoricalVector[]
    elseif !isnothing(target) && !(eltype(target) <: AbstractFloat)
        target = _discrete_encode(target)
    end

    treats = [treat(datastruct, vnames) for treat in treatments]

    ds = AbstractDataset[]

    if treatment_ds
        ds_td, ds_tc, ds_md = _treatments_ds(
            data,
            vnames,
            datastruct,
            treats, 
            float_type
        )
        !isempty(ds_td) && append!(ds, ds_td)
        !isempty(ds_tc) && append!(ds, ds_tc)
        !isempty(ds_md) && append!(ds, ds_md)
    end

    if leftover_ds
        ds_td, ds_tc, ds_md = _leftovers_ds(
            data,
            vnames,
            datastruct,
            treats,
            float_type
        )
        !isempty(ds_td) && append!(ds, ds_td)
        !isempty(ds_tc) && append!(ds, ds_tc)
        !isempty(ds_md) && append!(ds, ds_md)
    end

    if !isnothing(target) && !isnothing(balance)
        balance isa AbstractBalance && (balance=(balance,))
        balfuncs = _get_balfunc.(balance)
        balkws = _get_balkw.(balance)

        Threads.@threads for d in ds
            d.data, target = reduce(
                (acc, i) -> balfuncs[i](acc...; balkws[i]...),
                eachindex(balfuncs);
                init=(d.data, target)
            )
        end
    end

    return DataTreatment{float_type}(ds, target, treats, balance)
end

load_dataset(df::DataFrame, target::AbstractVector, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), target, args...; kwargs...)

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), CategoricalVector[], args...; kwargs...)

function load_dataset(
    dt::DataTreatment{T},
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    kwargs...
) where T
    data = reduce(hcat, d.data for d in dt.data)
    vnames = reduce(vcat, get_vnames(d) for d in dt.data)
    target = get_target(dt)

    load_dataset(data, vnames, target, treatments...; kwargs...)
end