# ---------------------------------------------------------------------------- #
#                            TreatmentGroup struct                             #
# ---------------------------------------------------------------------------- #
"""
    TreatmentGroup{T}

A configuration object for selecting and processing columns in a dataset passed to `DataTreatment`.

The type parameter `T` is the `typejoin` of the data types of all selected columns.

## Selection Parameters

Columns are selected based on:
- **`dims::Int`**: Dimensionality filter (`-1` selects all dimensions)
- **`name_expr`**: Column name filter, can be:
  - `Regex`: matches column names against the pattern
  - `Function`: predicate function applied to column names
  - `Vector{String}`: explicit list of column names to include
- **`datatype::Type`**: Filter columns by data type (default: `Any` means no filter)

## Processing Parameters (for multidimensional columns)

- **`aggrfunc::Base.Callable`**: Aggregation function applied to multidimensional elements:
  - `aggregate`: tabularizes multidimensional data into a flat matrix
  - `reducesize`: resizes multidimensional data while preserving dimensionality

- **`grouped::Bool`**: If `true`, further processing is performed on all selected columns together (jointly), rather than the default columnwise processing. If `false` (default), processing is applied to each column independently.

- **`groupby::Tuple{Vararg{Symbol}}`**: Further partitioning of output features from multidimensional processing.
  Possible grouping keys include `:vname` (column name), window index, or feature type applied.

## Fields

- `ids::Vector{Int}`: Column indices selected by this group
- `dims::Int`: Dimensionality filter used
- `vnames::Vector{String}`: Names of selected columns
- `aggrfunc::Base.Callable`: Aggregation function for multidimensional columns
- `grouped::Bool`: Whether to process all columns together (`true`) or columnwise (`false`)
- `groupby::Tuple{Vararg{Symbol}}`: Grouping specification for output features

The curried form `TreatmentGroup(; kwargs...)` returns a function that accepts a
`DataStructure` and forwards `kwargs`, useful for passing to `DataTreatment`.
"""
struct TreatmentGroup
    ids::Vector{Int}
    dims::Int
    vnames::Vector{String}
    aggrfunc::Base.Callable
    grouped::Bool
    groupby::Union{Nothing,Tuple{Vararg{Symbol}}}
    datatype::Type

    function TreatmentGroup(
        datastruct::NamedTuple,
        vnames::Vector{String};
        dims::Int=-1,
        name_expr::Union{Regex,Base.Callable,Vector{String}}=r".*",
        datatype::Type=Any,
        aggrfunc::Base.Callable=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
        grouped::Bool=false,
        groupby::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}=nothing
    )
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
            (datatype != Any && all_types[i] != datatype) && continue
            name_match(vnames[i]) || continue
            push!(ids, i)
        end

        col_types = all_types[ids]
        T = isempty(col_types) ? Any : mapreduce(identity, typejoin, col_types)
        isconcretetype(T) || (T = Any)

        new(ids, dims, vnames[ids], aggrfunc, grouped, groupby, T)
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