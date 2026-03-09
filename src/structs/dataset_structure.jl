# ---------------------------------------------------------------------------- #
#                           DatasetStructure struct                            #
# ---------------------------------------------------------------------------- #
"""
    DatasetStructure

A structure used by `DataTreatment` to perform an initial analysis of all metadata 
useful for constructing `DataTreatment` objects.

Analyzes the dataset column by column to:
- Verify the data type of each column
- Retrieve indices of possible `missing` and `NaN` values
- Verify the dimensionality of elements in each column

All information is stored in this structure and is essential for the correct 
functioning of `TreatmentGroup` structures, which allow users to pre-partition 
the dataset.

# Fields

- `vnames::Vector{String}`: Column names of the dataset
- `datatype::Vector{<:Type}`: Data type for each column
- `dims::Vector{Int}`: Dimensionality of elements for each column
- `valididxs::Vector{Vector{Int}}`: Indices of valid values for each column
- `missingidxs::Vector{Vector{Int}}`: Indices of `missing` values for each column
- `nanidxs::Vector{Vector{Int}}`: Indices of `NaN` values for each column
- `hasmissing::Vector{Vector{Int}}`: Indices of elements containing `missing` (for vectors/matrices)
- `hasnans::Vector{Vector{Int}}`: Indices of elements containing `NaN` (for vectors/matrices)

# Constructors

    DatasetStructure(
        dataset::Matrix,
        vnames::Union{Vector{String},Nothing}=["V\$i" for i in 1:size(dataset, 2)]
    ) -> DatasetStructure

    DatasetStructure(df::DataFrame) -> DatasetStructure

Scans the dataset column by column (in parallel via `Threads.@threads`) and builds
a `DatasetStructure` containing all metadata needed by `DataTreatment`.

For each column, [`_get_column_structure`](@ref) is called to extract:
- the element type
- the dimensionality
- the indices of valid, missing, NaN, and internally-corrupt elements

## Arguments
- `dataset::Matrix`: A matrix where each column is a feature. Elements may be scalars,
  arrays, or matrices, and may contain `missing` or `NaN` values.
- `vnames::Vector{String}`: Column names associated with each column of `dataset`.
- `df::DataFrame`: Converted to `Matrix` before processing. Column names are
  automatically extracted via `names(df)` and stored in the resulting `DatasetStructure`.

## Example
```julia
# from a Matrix with explicit names
ds = DatasetStructure(Matrix(df), names(df))
get_vnames(ds)        # ["col1", "col2", ...]
get_datatype(ds)      # Vector of types, one per column
get_valididxs(ds, 3)  # Valid row indices for column 3

# from a DataFrame (names extracted automatically)
ds = DatasetStructure(df)
get_vnames(ds)        # column names from the DataFrame
```
"""
struct DatasetStructure
    vnames::Vector{String}
    datatype::Vector{<:Type}
    dims::Vector{Int}
    valididxs::Vector{Vector{Int}}
    missingidxs::Vector{Vector{Int}}
    nanidxs::Vector{Vector{Int}}
    hasmissing::Vector{Vector{Int}}
    hasnans::Vector{Vector{Int}}

    function DatasetStructure(
        vnames::Vector{String},
        datatype::Vector{<:Type},
        dims::Vector{Int},
        valididxs::Vector{Vector{Int}},
        missingidxs::Vector{Vector{Int}},
        nanidxs::Vector{Vector{Int}},
        hasmissing::Vector{Vector{Int}},
        hasnans::Vector{Vector{Int}}
    )
        validate_vector_lengths(
            vnames,
            datatype,
            dims,
            valididxs,
            missingidxs,
            nanidxs,
            hasmissing,
            hasnans
        )

        new(vnames, datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end

    function DatasetStructure(
        dataset::Matrix,
        vnames::Union{Vector{String},Nothing}=["V$i" for i in 1:size(dataset, 2)],)
        ncols = size(dataset, 2)

        datatype = Vector{Type}(undef, ncols)
        dims = Vector{Int}(undef, ncols)
        valididxs = Vector{Vector{Int}}(undef, ncols)
        missingidxs = Vector{Vector{Int}}(undef, ncols)
        nanidxs = Vector{Vector{Int}}(undef, ncols)
        hasmissing = Vector{Vector{Int}}(undef, ncols)
        hasnans = Vector{Vector{Int}}(undef, ncols)

        Threads.@threads for i in axes(dataset, 2)
            datatype[i], dims[i], valididxs[i], missingidxs[i], nanidxs[i], hasmissing[i], hasnans[i] =
                _get_column_structure(@view(dataset[:, i]))
        end

        return DatasetStructure(vnames, datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end

    DatasetStructure(df::DataFrame; kwargs...) = DatasetStructure(Matrix(df), names(df))
end

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
"""
    Base.size(ds::DatasetStructure)

Returns the size of the structure as a tuple `(ncols,)`.
"""
Base.size(ds::DatasetStructure) = (length(ds.datatype),)

"""
    Base.length(ds::DatasetStructure)

Returns the number of columns in the structure.
"""
Base.length(ds::DatasetStructure) = length(ds.datatype)

"""
    Base.ndims(ds::DatasetStructure)

Returns the number of dimensions (always 1).
"""
Base.ndims(ds::DatasetStructure) = 1

"""
    Base.iterate(ds::DatasetStructure, state=1)

Supports iteration over the structure. Iterates over the data types of each column.
"""
Base.iterate(ds::DatasetStructure, state=1) = state > length(ds) ? nothing : (ds.datatype[state], state + 1)

"""
    Base.eachindex(ds::DatasetStructure)

Returns the indices of the columns.
"""
Base.eachindex(ds::DatasetStructure) = eachindex(ds.datatype)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_vnames(ds::DatasetStructure)
    get_vnames(ds::DatasetStructure, i::Int)
    get_vnames(ds::DatasetStructure, idxs::Vector{Int})

Returns the column names stored in the structure. If an index `i` is provided,
returns the name of column `i`. If a vector of indices is provided, returns a
view of the names for those columns.
"""
get_vnames(ds::DatasetStructure) = ds.vnames
get_vnames(ds::DatasetStructure, i::Int) = ds.vnames[i]
get_vnames(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.vnames[idxs]

"""
    get_datatype(ds::DatasetStructure)
    get_datatype(ds::DatasetStructure, i::Int)
    get_datatype(ds::DatasetStructure, idxs::Vector{Int})

Returns the data types. If an index `i` is provided, returns the type of column `i`.
If a vector of indices is provided, returns a view of the types for those columns.
"""
get_datatype(ds::DatasetStructure) = ds.datatype
get_datatype(ds::DatasetStructure, i::Int) = ds.datatype[i]
get_datatype(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.datatype[idxs]

"""
    get_dims(ds::DatasetStructure)
    get_dims(ds::DatasetStructure, i::Int)
    get_dims(ds::DatasetStructure, idxs::Vector{Int})

Returns the dimensionalities. If an index `i` is provided, returns the dimensionality of column `i`.
If a vector of indices is provided, returns a view of the dimensionalities for those columns.
"""
get_dims(ds::DatasetStructure) = ds.dims
get_dims(ds::DatasetStructure, i::Int) = ds.dims[i]
get_dims(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.dims[idxs]

"""
    get_valididxs(ds::DatasetStructure)
    get_valididxs(ds::DatasetStructure, i::Int)
    get_valididxs(ds::DatasetStructure, idxs::Vector{Int})

Returns the indices of valid values. If an index `i` is provided, returns the valid indices of column `i`.
If a vector of indices is provided, returns a view of the valid indices for those columns.
"""
get_valididxs(ds::DatasetStructure) = ds.valididxs
get_valididxs(ds::DatasetStructure, i::Int) = ds.valididxs[i]
get_valididxs(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.valididxs[idxs]

"""
    get_missingidxs(ds::DatasetStructure)
    get_missingidxs(ds::DatasetStructure, i::Int)
    get_missingidxs(ds::DatasetStructure, idxs::Vector{Int})

Returns the indices of `missing` values. If an index `i` is provided, returns the `missing` indices of column `i`.
If a vector of indices is provided, returns a view of the missing indices for those columns.
"""
get_missingidxs(ds::DatasetStructure) = ds.missingidxs
get_missingidxs(ds::DatasetStructure, i::Int) = ds.missingidxs[i]
get_missingidxs(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.missingidxs[idxs]

"""
    get_nanidxs(ds::DatasetStructure)
    get_nanidxs(ds::DatasetStructure, i::Int)
    get_nanidxs(ds::DatasetStructure, idxs::Vector{Int})

Returns the indices of `NaN` values. If an index `i` is provided, returns the `NaN` indices of column `i`.
If a vector of indices is provided, returns a view of the NaN indices for those columns.
"""
get_nanidxs(ds::DatasetStructure) = ds.nanidxs
get_nanidxs(ds::DatasetStructure, i::Int) = ds.nanidxs[i]
get_nanidxs(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.nanidxs[idxs]

"""
    get_hasmissing(ds::DatasetStructure)
    get_hasmissing(ds::DatasetStructure, i::Int)
    get_hasmissing(ds::DatasetStructure, idxs::Vector{Int})

Returns the indices of elements containing `missing` values internally. If an index `i` is provided, 
returns the indices for column `i`. If a vector of indices is provided, returns a view of the indices for those columns.
"""
get_hasmissing(ds::DatasetStructure) = ds.hasmissing
get_hasmissing(ds::DatasetStructure, i::Int) = ds.hasmissing[i]
get_hasmissing(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.hasmissing[idxs]

"""
    get_hasnans(ds::DatasetStructure)
    get_hasnans(ds::DatasetStructure, i::Int)
    get_hasnans(ds::DatasetStructure, idxs::Vector{Int})

Returns the indices of elements containing `NaN` values internally. If an index `i` is provided, 
returns the indices for column `i`. If a vector of indices is provided, returns a view of the indices for those columns.
"""
get_hasnans(ds::DatasetStructure) = ds.hasnans
get_hasnans(ds::DatasetStructure, i::Int) = ds.hasnans[i]
get_hasnans(ds::DatasetStructure, idxs::Vector{Int}) = @views ds.hasnans[idxs]

# ---------------------------------------------------------------------------- #
#                              structure method                                #
# ---------------------------------------------------------------------------- #
"""
    _isnanval(v) -> Bool

Returns `true` if `v` is an `AbstractFloat` and is `NaN`, `false` otherwise.
"""
_isnanval(v) = v isa AbstractFloat && isnan(v)

"""
    _isarray(v) -> Bool

Returns `true` if `v` is an `AbstractArray` with `AbstractFloat` elements, `false` otherwise.
"""
_isarray(v) = v isa AbstractArray{<:AbstractFloat}

"""
    _get_column_structure(col::AbstractVector) -> NTuple{7}

Analyzes a single column and returns a tuple of metadata:

- `datatype::Type`: The common supertype of all valid values (`typejoin` over valid elements).
  Returns `Any` if the column has no valid values.
- `dims::Int`: The maximum dimensionality of array-valued elements (`ndims`).
  Returns `0` for scalar columns or empty columns.
- `valididxs::Vector{Int}`: Indices of elements that are neither `missing` nor `NaN`.
- `missingidxs::Vector{Int}`: Indices of `missing` elements.
- `nanidxs::Vector{Int}`: Indices of top-level `NaN` elements (scalar floats that are `NaN`).
- `hasmissing::Vector{Int}`: Among `valididxs`, indices of array elements that contain
  at least one `missing` internally.
- `hasnans::Vector{Int}`: Among `valididxs`, indices of array elements that contain
  at least one `NaN` internally.

!!! note
    `hasmissing` and `hasnans` are indices into `valididxs`, not global row indices.
"""
function _get_column_structure(col::AbstractVector)
    valididxs = findall(i -> !ismissing(col[i]) && !_isnanval(col[i]), eachindex(col))
    missingidxs = findall(i -> ismissing(col[i]), eachindex(col))
    nanidxs = findall(i -> !ismissing(col[i]) && _isnanval(col[i]), eachindex(col))

    valid_vals = @view col[valididxs]

    hasmissing = findall(i -> _isarray(col[i]) && any(ismissing, col[i]), valididxs)
    hasnans = findall(i -> _isarray(col[i]) && any(_isnanval, col[i]), valididxs)

    datatype = isempty(valid_vals) ? Any : mapreduce(typeof, typejoin, valid_vals)
    dims = isempty(valid_vals) ? 0 : maximum(v -> _isarray(v) ? ndims(v) : 0, valid_vals)

    return datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans
end

# ---------------------------------------------------------------------------- #
#                                 show method                                  #
# ---------------------------------------------------------------------------- #
function get_structure(ds::DatasetStructure)
    ncols = length(ds.datatype)
    
    # Group columns by datatype
    type_to_cols = Dict{Type, Vector{Int}}()
    foreach(i -> begin
        dtype = ds.datatype[i]
        haskey(type_to_cols, dtype) || (type_to_cols[dtype] = Int[])
        push!(type_to_cols[dtype], i)
    end, 1:ncols)
    
    # Find columns with missing values
    cols_with_missing = filter(i -> !isempty(ds.missingidxs[i]) || !isempty(ds.hasmissing[i]), 1:ncols)
    
    # Find columns with NaN values
    cols_with_nans = filter(i -> !isempty(ds.nanidxs[i]) || !isempty(ds.hasnans[i]), 1:ncols)
    
    return (
        ncols = ncols,
        type_to_cols = type_to_cols,
        cols_with_missing = cols_with_missing,
        cols_with_nans = cols_with_nans
    )
end

# one-line
Base.show(io::IO, ds::DatasetStructure) = print(io, "DatasetStructure(", length(ds), " cols)")

# multi-line
function Base.show(io::IO, ::MIME"text/plain", ds::DatasetStructure)
    structure = get_structure(ds)
    
    println(io, "DatasetStructure($(structure.ncols) columns)")
    println(io, "├─ datatypes by columns:")
    
    type_list = collect(structure.type_to_cols)
    for (idx, (dtype, cols)) in enumerate(type_list)
        is_last_type = (idx == length(type_list))
        prefix = is_last_type ? "│  └─" : "│  ├─"
        println(io, "$prefix $dtype: $(cols)")
    end
    
    if !isempty(structure.cols_with_missing)
        println(io, "├─ missing at: columns $(structure.cols_with_missing)")
    end
    
    if !isempty(structure.cols_with_nans)
        println(io, "└─ NaN at: columns $(structure.cols_with_nans)")
    end
end

