# ---------------------------------------------------------------------------- #
#                            TreatmentGroup struct                             #
# ---------------------------------------------------------------------------- #
"""
    TreatmentGroup{T}

A configuration object for selecting and processing columns in a dataset passed to `DataTreatment`.

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

## Examples

```julia
# Select all columns with dimensionality 0 (scalars)
TreatmentGroup(dims=0)

# Select columns matching a regex pattern
TreatmentGroup(name_expr=r"^V")

# Select specific columns by name
TreatmentGroup(name_expr=["col1", "col2"])

# Select continuous columns with custom aggregation
TreatmentGroup(datatype=Float64, aggrfunc=aggregate(...))
```
"""
struct TreatmentGroup{T}
    idxs::Vector{Int}
    dims::Int
    vnames::Vector{String}
    aggrfunc::Base.Callable
    groupby::Tuple{Vararg{Symbol}}

    function TreatmentGroup(
        ds_struct::DatasetStructure;
        dims::Int=-1,
        name_expr::Union{Regex,Base.Callable,Vector{String}}=r".*",
        datatype::Type=Any,
        aggrfunc::Base.Callable=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
        groupby::Tuple{Vararg{Symbol}}=(:vname,)
    )
        # filter by dims
        idxs = dims == -1 ?
            collect(1:length(ds_struct)) :
            findall(get_dims(ds_struct) .== dims)

        # filter by datatype
        datatype != Any && (idxs= idxs ∩ findall(get_datatype(ds_struct) .== datatype))

        vnames = get_vnames(ds_struct)
        valid_names = name_expr isa Regex ?
            filter(item -> match(name_expr, item) !== nothing, vnames) :
            name_expr isa Vector{String} ?
                name_expr :
                filter(name_expr, vnames)

        idxs = idxs ∩ findall(n -> n in valid_names, vnames)

        # get types
        col_types = get_datatype(ds_struct, idxs)
        T = isempty(col_types) ? Any : mapreduce(identity, typejoin, col_types)

        new{T}(idxs, dims, vnames[idxs], aggrfunc, groupby)
    end

    TreatmentGroup(ds::Matrix, vnames::Vector{String}; kwargs...) =
        TreatmentGroup(get_dataset_structure(ds, vnames); kwargs...)
    TreatmentGroup(df::DataFrame; kwargs...) =
        TreatmentGroup(get_dataset_structure(df); kwargs...)
end

TreatmentGroup(; kwargs...) = x -> TreatmentGroup(x; kwargs...)

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
"""
    Base.length(tg::TreatmentGroup)

Returns the number of columns selected by this group.
"""
Base.length(tg::TreatmentGroup) = length(tg.idxs)

"""
    Base.iterate(tg::TreatmentGroup, state=1)

Iterates over the selected column indices.
"""
Base.iterate(tg::TreatmentGroup, state=1) = state > length(tg) ? nothing : (tg.idxs[state], state + 1)

"""
    Base.eachindex(tg::TreatmentGroup)

Returns the indices of the selected columns vector.
"""
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
    idxs = Vector{Vector{Int}}(undef, length(tgs))

    for (i, tg) in enumerate(tgs)
        idxs[i] = get_idxs(tg)
        idxs[1:i-1] = map(v -> filter(∉(idxs[i]), v), idxs[1:i-1])
    end

    return idxs
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

    # safer label for anonymous callables/closures
    ftype_name = String(Base.unwrap_unionall(typeof(tg.aggrfunc)).name.name)
    aggr_label = startswith(ftype_name, "#") ? "anonymous callable" : ftype_name

    println(io, "TreatmentGroup{$T}($n_selected columns selected)")
    println(io, "├─ dims filter: $dims_str")

    if tg.dims > 0
        println(io, "├─ selected indices: $(tg.idxs)")
        println(io, "├─ aggregation function: $aggr_label")
        print(io,   "└─ groupby: $(tg.groupby)")
    else
        print(io,   "└─ selected indices: $(tg.idxs)")
    end
end
