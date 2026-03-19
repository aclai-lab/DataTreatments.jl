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
  
- **`groupby::Tuple{Vararg{Symbol}}`**: Further partitioning of output features from multidimensional processing.
  Possible grouping keys include `:vname` (column name), window index, or feature type applied.

## Fields

- `idxs::Vector{Int}`: Column indices selected by this group
- `dims::Int`: Dimensionality filter used
- `vnames::Vector{String}`: Names of selected columns
- `aggrfunc::Base.Callable`: Aggregation function for multidimensional columns
- `groupby::Tuple{Vararg{Symbol}}`: Grouping specification for output features

## Constructors

    TreatmentGroup(ds_struct::DatasetStructure; kwargs...)
    TreatmentGroup(ds::Matrix, vnames::Vector{String}; kwargs...)
    TreatmentGroup(df::DataFrame; kwargs...)
    TreatmentGroup(; kwargs...)  # curried form, returns a callable

The curried form `TreatmentGroup(; kwargs...)` returns a function that accepts a
`DatasetStructure` and forwards `kwargs`, useful for passing to `DataTreatment`.

## Examples

```julia
# Select all columns with dimensionality 0 (scalars) — curried form
TreatmentGroup(dims=0)

# Select columns matching a regex pattern
TreatmentGroup(ds_struct, name_expr=r"^V")

# Select specific columns by name
TreatmentGroup(ds_struct, name_expr=["col1", "col2"])

# Select continuous columns with custom aggregation
TreatmentGroup(ds_struct, datatype=Float64, aggrfunc=aggregate(...))
```
"""
struct TreatmentGroup{T}
    idxs::Vector{Int}
    dims::Int
    vnames::Vector{String}
    aggrfunc::Base.Callable
    groupby::Union{Nothing,Tuple{Vararg{Symbol}}}

    function TreatmentGroup(
        ds_struct::DatasetStructure;
        dims::Int=-1,
        name_expr::Union{Regex,Base.Callable,Vector{String}}=r".*",
        datatype::Type=Any,
        aggrfunc::Base.Callable=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
        groupby::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}=nothing
    )
        vnames = get_vnames(ds_struct)
        all_dims = get_dims(ds_struct)
        all_types = get_datatype(ds_struct)
        groupby isa Symbol && (groupby = (groupby,))

        # build name matcher once
        name_match = if name_expr isa Regex
            n -> match(name_expr, n) !== nothing
        elseif name_expr isa Vector{String}
            name_set = Set(name_expr)
            n -> n in name_set
        else
            name_expr
        end

        # single pass: collect indices matching all filters
        idxs = Int[]
        for i in eachindex(ds_struct)
            (dims != -1 && all_dims[i] != dims) && continue
            (datatype != Any && all_types[i] != datatype) && continue
            name_match(vnames[i]) || continue
            push!(idxs, i)
        end

        col_types = get_datatype(ds_struct, idxs)
        T = isempty(col_types) ? Any : mapreduce(identity, typejoin, col_types)

        new{T}(idxs, dims, vnames[idxs], aggrfunc, groupby)
    end

    TreatmentGroup(ds::Matrix, vnames::Vector{String}; kwargs...) =
        TreatmentGroup(DatasetStructure(ds, vnames); kwargs...)
    TreatmentGroup(df::DataFrame; kwargs...) =
        TreatmentGroup(DatasetStructure(df); kwargs...)
end

TreatmentGroup(; kwargs...) = x -> TreatmentGroup(x; kwargs...)

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
Base.length(tg::TreatmentGroup) = length(tg.idxs)
Base.iterate(tg::TreatmentGroup, state=1) = state > length(tg) ? nothing : (tg.idxs[state], state + 1)
Base.eachindex(tg::TreatmentGroup) = eachindex(tg.idxs)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_idxs(tg::TreatmentGroup)
    get_idxs(tg::TreatmentGroup, i::Int)

Returns the column indices selected by this group. If `i` is provided, returns the `i`-th index.
"""
get_idxs(tg::TreatmentGroup) = tg.idxs
get_idxs(tg::TreatmentGroup, i::Int) = tg.idxs[i]

"""
    get_dims(tg::TreatmentGroup)

Returns the dimensionality filter used to select columns (`-1` means no filter).
"""
get_dims(tg::TreatmentGroup) = tg.dims

"""
    get_vnames(tg::TreatmentGroup)
    get_vnames(tg::TreatmentGroup, i::Int)
    get_vnames(tg::TreatmentGroup, idxs::Vector{Int})

Returns the column names selected by this group. If `i` is provided, returns the `i`-th name.
If a vector of indices is provided, returns a view of the names for those positions.
"""
get_vnames(tg::TreatmentGroup) = tg.vnames
get_vnames(tg::TreatmentGroup, i::Int) = tg.vnames[i]
get_vnames(tg::TreatmentGroup, idxs::Vector{Int}) = @views tg.vnames[idxs]

"""
    get_aggrfunc(tg::TreatmentGroup)

Returns the aggregation function used by this group.
"""
get_aggrfunc(tg::TreatmentGroup) = tg.aggrfunc

"""
    get_groupby(tg::TreatmentGroup)

Returns the `groupby` tuple of symbols used to partition output features.
"""
get_groupby(tg::TreatmentGroup) = tg.groupby

"""
    has_groupby(tg::TreatmentGroup)

Returns `true` if a `groupby` specification is set for this group.
"""
has_groupby(tg::TreatmentGroup) = !isnothing(tg.groupby)


# ---------------------------------------------------------------------------- #
#                             custom lazy methods                              #
# ---------------------------------------------------------------------------- #
"""
    get_idxs(tgs::Vector{<:TreatmentGroup})

Returns a vector of index vectors, one for each treatment group.

When the same index appears in multiple groups, **later groups take precedence**:
the duplicated index is kept in the last group where it appears and removed from all
previous groups.

In other words, the output groups are made pairwise disjoint by filtering overlaps
from left to right, giving priority to groups with higher position in `tgs`.
"""
function get_idxs(tgs::Vector{<:TreatmentGroup})
    seen = Set{Int}()
    no_intersect_groups = reverse(map(reverse(tgs)) do tg
        unique_idxs = filter(∉(seen), get_idxs(tg))
        union!(seen, unique_idxs)
        unique_idxs
    end)

    any(isempty.(no_intersect_groups)) &&
        @warn "One or more TreatmentGroups have no columns after resolving overlaps " *
        "(all their indices were claimed by later groups)"

    return no_intersect_groups
end

# ---------------------------------------------------------------------------- #
#                                 show method                                  #
# ---------------------------------------------------------------------------- #
# one-line
function Base.show(io::IO, tg::TreatmentGroup{T}) where {T}
    dims_str = tg.dims == -1 ? "all" : string(tg.dims)
    print(io, "TreatmentGroup{$T}(", length(tg.idxs), " cols, dims=", dims_str, ")")
end

# multi-line
function Base.show(io::IO, ::MIME"text/plain", tg::TreatmentGroup{T}) where {T}
    n_selected = length(tg.idxs)
    dims_str = tg.dims == -1 ? "all" : string(tg.dims)

    aggr_label = try
        ftype_name = String(Base.unwrap_unionall(typeof(tg.aggrfunc)).name.name)
        startswith(ftype_name, "#") ? "anonymous callable" : ftype_name
    catch
        string(typeof(tg.aggrfunc))
    end

    println(io, "TreatmentGroup{$T}($n_selected columns selected)")
    println(io, "├─ dims filter: $dims_str")

    if tg.dims != 0
        println(io, "├─ selected indices: $(tg.idxs)")
        println(io, "├─ aggregation function: $aggr_label")
        print(io,   "└─ groupby: $(tg.groupby)")
    else
        print(io,   "└─ selected indices: $(tg.idxs)")
    end
end
