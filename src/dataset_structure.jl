# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
_isnanval(v) = v isa AbstractFloat && isnan(v)
_isarray(v) = v isa AbstractArray

# ---------------------------------------------------------------------------- #
#                            TargetStructure struct                            #
# ---------------------------------------------------------------------------- #
"""
    TargetStructure

A structure used by `DataTreatments` to store information about the target (dependent variable) of a dataset.
It holds both the vector of target values and, for classification tasks, the labels associated with discrete-encoded classes.

This struct is constructed automatically from a target vector and is used internally by `DataTreatment` objects.

# Fields
- `values::Vector{Union{<:Int, <:AbstractFloat}}`: The encoded target values (integers for classification, floats for regression).
- `labels::Union{Nothing, CategoricalArrays.CategoricalVector}`: The class labels for classification tasks, or `nothing` for regression.
"""
struct TargetStructure
    values::Vector{Union{<:Int,<:AbstractFloat}}
    labels::Union{Nothing,CategoricalArrays.CategoricalVector}

    function TargetStructure(y::AbstractVector)
        eltype(y) <: AbstractFloat ?
            new(y, nothing) :
            new(discrete_encode(y)...)
    end
end

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
get_values(ts::TargetStructure)

Returns the vector of encoded target values.
"""
get_values(ts::TargetStructure) = ts.values

"""
get_labels(ts::TargetStructure)

Returns the class labels for classification tasks, or nothing for regression.
"""
get_labels(ts::TargetStructure) = ts.labels

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

For each column, a single pass over the elements extracts:
- the element type (computed incrementally via `typejoin`)
- the dimensionality (tracked as the maximum `ndims` across array elements)
- the indices of valid, missing, NaN, and internally-corrupt elements

## Arguments
- `dataset::Matrix`: A matrix where each column is a feature. Elements may be scalars,
  arrays, or matrices, and may contain `missing` or `NaN` values.
- `vnames::Union{Vector{String},Nothing}`: Column names associated with each column of `dataset`.
  Defaults to `["V1", "V2", ...]` if not provided.
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
        dataset::Matrix,
        vnames::Union{Vector{String},Nothing}=["V$i" for i in 1:size(dataset, 2)]
    )
        ncols = size(dataset, 2)

        datatype = Vector{Type}(undef, ncols)
        dims = Vector{Int}(undef, ncols)
        valididxs = Vector{Vector{Int}}(undef, ncols)
        missingidxs = Vector{Vector{Int}}(undef, ncols)
        nanidxs = Vector{Vector{Int}}(undef, ncols)
        hasmissing = Vector{Vector{Int}}(undef, ncols)
        hasnans = Vector{Vector{Int}}(undef, ncols)

        Threads.@threads for i in axes(dataset, 2)
            col = @view(dataset[:, i])

            _valid = Int[]
            _missing = Int[]
            _nan = Int[]
            _hasmissing = Int[]
            _hasnans = Int[]

            dt = Any
            maxdims = 0

            @inbounds for j in eachindex(col)
                v = col[j]
                if ismissing(v)
                    push!(_missing, j)
                elseif _isnanval(v)
                    push!(_nan, j)
                else
                    push!(_valid, j)
                    dt = dt === Any ? typeof(v) : typejoin(dt, typeof(v))
                    if _isarray(v)
                        d = ndims(v)
                        d > maxdims && (maxdims = d)
                        any(ismissing, v) && push!(_hasmissing, j)
                        any(_isnanval, v) && push!(_hasnans, j)
                    end
                end
            end

            datatype[i] = dt
            dims[i] = maxdims
            valididxs[i] = _valid
            missingidxs[i] = _missing
            nanidxs[i] = _nan
            hasmissing[i] = _hasmissing
            hasnans[i] = _hasnans
        end

        return new(vnames, datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end

    DatasetStructure(df::DataFrame; kwargs...) = DatasetStructure(Matrix(df), names(df))
end

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
Base.size(ds::DatasetStructure) = (length(ds.datatype),)
Base.length(ds::DatasetStructure) = length(ds.datatype)
Base.eachindex(ds::DatasetStructure) = Base.OneTo(length(ds))

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
#                                 show method                                  #
# ---------------------------------------------------------------------------- #
function get_structure(ds::DatasetStructure)
    ncols = length(ds.datatype)
    
    # group columns by datatype
    type_to_cols = Dict{Type, Vector{Int}}()
    foreach(i -> begin
        dtype = ds.datatype[i]
        haskey(type_to_cols, dtype) || (type_to_cols[dtype] = Int[])
        push!(type_to_cols[dtype], i)
    end, 1:ncols)
    
    # find columns with missing values
    cols_with_missing = filter(i -> !isempty(ds.missingidxs[i]) || !isempty(ds.hasmissing[i]), 1:ncols)
    
    # find columns with NaN values
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

