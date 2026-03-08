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

- `datatype::Vector{<:Type}`: Data type for each column
- `dims::Vector{Int}`: Dimensionality of elements for each column
- `valididxs::Vector{Vector{Int}}`: Indices of valid values for each column
- `missingidxs::Vector{Vector{Int}}`: Indices of `missing` values for each column
- `nanidxs::Vector{Vector{Int}}`: Indices of `NaN` values for each column
- `hasmissing::Vector{Vector{Int}}`: Indices of elements containing `missing` (for vectors/matrices)
- `hasnans::Vector{Vector{Int}}`: Indices of elements containing `NaN` (for vectors/matrices)
"""
struct DatasetStructure
    datatype::Vector{<:Type}
    dims::Vector{Int}
    valididxs::Vector{Vector{Int}}
    missingidxs::Vector{Vector{Int}}
    nanidxs::Vector{Vector{Int}}
    hasmissing::Vector{Vector{Int}}
    hasnans::Vector{Vector{Int}}

    function DatasetStructure(
        datatype::Vector{<:Type},
        dims::Vector{Int},
        valididxs::Vector{Vector{Int}},
        missingidxs::Vector{Vector{Int}},
        nanidxs::Vector{Vector{Int}},
        hasmissing::Vector{Vector{Int}},
        hasnans::Vector{Vector{Int}}
    )
        validate_vector_lengths(
            datatype,
            dims,
            valididxs,
            missingidxs,
            nanidxs,
            hasmissing,
            hasnans
        )

        new(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end
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
#                               summary method                                 #
# ---------------------------------------------------------------------------- #
"""
    get_structure(ds::DatasetStructure)

Returns a summary of the structure as a named tuple containing:

- `ncols::Int`: Number of columns
- `type_to_cols::Dict{Type, Vector{Int}}`: Mapping between data types and column indices
- `cols_with_missing::Vector{Int}`: Indices of columns with `missing` values
- `cols_with_nans::Vector{Int}`: Indices of columns with `NaN` values

Useful for obtaining an overall view of the dataset structure.

# Example

```julia
ds = get_dataset_structure(df)
structure = get_structure(ds)

# Columns of type Float64
float_cols = structure.type_to_cols[Float64]

# Which columns have missing values?
problematic = union(structure.cols_with_missing, structure.cols_with_nans)
```
"""
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

# ---------------------------------------------------------------------------- #
#                                 show method                                  #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, ds::DatasetStructure)
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
