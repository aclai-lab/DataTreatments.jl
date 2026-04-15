# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
"""
    DataTreatment{T}

Top-level container produced by [`load_dataset`](@ref). Holds all output datasets
derived from a source table, a target vector, and the applied treatment groups.

## Structure

```
DataTreatment{T}  (T = float_type, e.g. Float64)
├── data    ::Vector{AbstractDataset}           # ordered list of output datasets
│    ├── DiscreteDataset{...}                   # from discrete (categorical) columns
│    ├── ContinuousDataset{T}                   # from continuous (scalar) columns
│    └── MultidimDataset{T, ...}                # from multidimensional columns
│         ├── AggregateFeat variant             # → tabular scalar output
│         └── ReduceFeat variant                # → array output
├── target  ::AbstractVector                    # encoded target vector (labels)
├── treats  ::Vector{TreatmentGroup}            # user directives
└── balance ::Union{Nothing,AbstractBalance,    # balancing strategy (or nothing)
                    Tuple{Vararg{<:AbstractBalance}}}
```

## Full pipeline overview

```
Raw DataFrame / Matrix
        │
        ▼  load_dataset(data, vnames, target, treatments...; float_type, balance)
        │
        ├─ _inspecting(data)  →  datastruct::NamedTuple
        │   (inspect all columns: types, missing, NaN, dims, ...)
        │
        ├─ encode target  →  CategoricalVector
        │
        ├─ for each TreatmentGroup:
        │   ├── classify columns  →  discrete_ids, continuous_ids, multidim_ids
        │   ├── DiscreteDataset(discrete_ids, ...)
        │   ├── ContinuousDataset(continuous_ids, ...)
        │   └── MultidimDataset(multidim_ids, ..., aggrfunc)
        │
        ├─ if balance is provided:
        │   │  (single AbstractBalance is wrapped into a 1-tuple)
        │   └── for each dataset d in ds:
        │        reduce over balance chain:
        │          (d.data, target) = bal₁ ∘ bal₂ ∘ … ∘ balₙ
        │
        └── DataTreatment{T}(ds, target, treats, balance)
```

## Accessor summary

| Method              | Returns                                      |
|---------------------|----------------------------------------------|
| `get_discrete(dt)`  | `(Matrix, vnames)` for categorical columns   |
| `get_continuous(dt)`| `(Matrix{T}, vnames)` for scalar columns     |
| `get_aggregated(dt)`| `(Matrix{T}, vnames)` for aggregated series  |
| `get_reduced(dt)`   | `(Matrix{Array{T}}, vnames)` for series      |
| `get_tabular(dt)`   | merged tabular matrix + all column names     |
| `get_multidim(dt)`  | reduced multidim matrix + column names       |
| `get_target(dt)`    | the encoded target vector                    |

# Type Parameter
- `T`: The floating-point type used throughout (e.g., `Float64`, `Float32`).

# Fields
- `data`: Ordered list of output datasets (discrete, continuous, multidimensional).
- `target`: Encoded target vector (labels).
- `treats`: User-specified treatment groups that drove column classification.
- `balance`: Optional balancing strategy applied to the dataset, or `nothing` if no
  balancing is requested. Can be a single `AbstractBalance` or a
  `Tuple{Vararg{<:AbstractBalance}}` to chain multiple strategies sequentially
  (e.g., `(SMOTE(k=5), TomekUndersampler())`). When a tuple is provided, each
  balancer is applied in order, passing the `(data, target)` output of one step
  as input to the next.

See also: [`load_dataset`](@ref), [`DiscreteDataset`](@ref),
[`ContinuousDataset`](@ref), [`MultidimDataset`](@ref), [`TreatmentGroup`](@ref),
[`AbstractBalance`](@ref)
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