# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
get_discrete(dt::TreatmentOutput) = filter(d -> d isa DiscreteDataset, dt)
get_continuous(dt::TreatmentOutput) = filter(d -> d isa ContinuousDataset, dt)
get_multidim(dt::TreatmentOutput) = filter(d -> d isa MultidimDataset, dt)

get_aggregated(dt::TreatmentOutput) = filter(d -> d isa MultidimDataset && 
    eltype(getfield(d, :info)) <: AggregateFeat, dt)

get_reduced(dt::TreatmentOutput) = filter(d -> d isa MultidimDataset && 
    eltype(getfield(d, :info)) <: ReduceFeat, dt)

# ---------------------------------------------------------------------------- #
#                             get tabular method                               #
# ---------------------------------------------------------------------------- #
"""
    get_tabular(dt::DataTreatment, args...; groupby_split=true, output_type=standard, kwargs...) -> TreatmentOutput

Convenience function to collect all tabular-like datasets from a `DataTreatment` object, including discrete, continuous, and aggregated multidimensional data.

# Purpose

`get_tabular` is especially useful when working with datasets containing heterogeneous data types (e.g., a mix of tabular and multidimensional columns). It extracts all columns that can be represented in tabular form, including those multidimensional columns that have been aggregated (via `TreatmentGroup` settings) into a tabular structure.

This function is helpful because the `DataTreatments` package allows flexible treatment of multidimensional data: you can aggregate such data (e.g., by computing summary statistics) or reduce it in other ways. Aggregated multidimensional data is suitable for inclusion in tabular datasets, which is what `get_tabular` collects.

# Arguments

- `dt::DataTreatment`: The data treatment object.
- `args...`: One or more treatment group callables (e.g., from `TreatmentGroup`). These specify how to group or aggregate columns.
- `groupby_split::Bool=true`: Whether to split multidimensional datasets by group.
- `output_type`: Output format function (`standard`, `matrix`, or `dataframe`).
- `kwargs...`: Additional keyword arguments passed to `get_dataset`, such as:
    - `treatment_ds::Bool=true`: Include datasets defined by the treatment groups.
    - `leftover_ds::Bool=true`: Include datasets for columns not assigned to any treatment group.

# Returns

A `TreatmentOutput` vector containing all discrete, continuous, and aggregated multidimensional datasets in tabular form.

See also: [`get_multidim`](@ref), [`get_dataset`](@ref)
"""
@inline function get_tabular(
    dt::DataTreatment,
    args...;
    groupby_split::Bool=true,
    output_type::Base.Callable=standard, # standard, matrix, dataframe
    kwargs...
)::TreatmentOutput
    data = get_dataset(dt::DataTreatment, args...; kwargs...)
    isempty(data) && return data
    
    selected = vcat(get_discrete(data), get_continuous(data), get_aggregated(data))

    return output_type(selected, groupby_split)
end

# ---------------------------------------------------------------------------- #
#                            get multidim method                               #
# ---------------------------------------------------------------------------- #
"""
    get_multidim(dt::DataTreatment, args...; output_type=standard, kwargs...) -> TreatmentOutput

Convenience function to collect all reduced multidimensional datasets from a `DataTreatment` object.

# Purpose

`get_multidim` is useful for extracting only those columns that remain multidimensional after treatment, particularly when the dataset contains both tabular and multidimensional data. This is important because `DataTreatments` allows you to specify, via `TreatmentGroup`, how multidimensional data should be handled: it can be aggregated (for tabular output) or reduced (for multidimensional output).

Use `get_multidim` when you want to focus on the reduced multidimensional features, as opposed to the aggregated/tabular ones.

# Arguments

- `dt::DataTreatment`: The data treatment object.
- `args...`: One or more treatment group callables (e.g., from `TreatmentGroup`). These specify how to group or reduce columns.
- `output_type`: Output format function (`standard`, `matrix`, or `dataframe`).
- `kwargs...`: Additional keyword arguments passed to `get_dataset`, such as:
    - `treatment_ds::Bool=true`: Include datasets defined by the treatment groups.
    - `leftover_ds::Bool=true`: Include datasets for columns not assigned to any treatment group.

# Returns

A `TreatmentOutput` vector containing all reduced multidimensional datasets.

See also: [`get_tabular`](@ref), [`get_dataset`](@ref)
"""
@inline function get_multidim(
    dt::DataTreatment,
    args...;
    output_type::Base.Callable=standard, # standard, matrix, dataframe
    kwargs...
)::TreatmentOutput
    data = get_dataset(dt::DataTreatment, args...; kwargs...)
    isempty(data) && return data

    return output_type(get_reduced(data), false)
end
