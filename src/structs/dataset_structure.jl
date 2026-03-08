# ---------------------------------------------------------------------------- #
#                           DatasetStructure struct                            #
# ---------------------------------------------------------------------------- #
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

# Size and length methods for DatasetStructure
Base.size(ds::DatasetStructure) = (length(ds.datatype),)
Base.length(ds::DatasetStructure) = length(ds.datatype)
Base.ndims(ds::DatasetStructure) = 1

# Iteration support
Base.iterate(ds::DatasetStructure, state=1) = state > length(ds) ? nothing : (ds.datatype[state], state + 1)
Base.eachindex(ds::DatasetStructure) = eachindex(ds.datatype)

# Getter methods for DatasetStructure
get_datatype(ds::DatasetStructure) = ds.datatype
get_dims(ds::DatasetStructure) = ds.dims
get_valididxs(ds::DatasetStructure) = ds.valididxs
get_missingidxs(ds::DatasetStructure) = ds.missingidxs
get_nanidxs(ds::DatasetStructure) = ds.nanidxs
get_hasmissing(ds::DatasetStructure) = ds.hasmissing
get_hasnans(ds::DatasetStructure) = ds.hasnans

# Get field by column index
get_datatype(ds::DatasetStructure, i::Int) = ds.datatype[i]
get_dims(ds::DatasetStructure, i::Int) = ds.dims[i]
get_valididxs(ds::DatasetStructure, i::Int) = ds.valididxs[i]
get_missingidxs(ds::DatasetStructure, i::Int) = ds.missingidxs[i]
get_nanidxs(ds::DatasetStructure, i::Int) = ds.nanidxs[i]
get_hasmissing(ds::DatasetStructure, i::Int) = ds.hasmissing[i]
get_hasnans(ds::DatasetStructure, i::Int) = ds.hasnans[i]

# Method to get summary structure from DatasetStructure
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

# Custom show method for DatasetStructure
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