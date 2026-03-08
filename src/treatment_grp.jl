using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    V3 = [3.2, 4.2, 5.2, missing, 2.4],
    V4 = [4.1, NaN, NaN, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
    ts3 = [collect(1.0:1.2:7.0), NaN, NaN, missing, collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

abstract type AbstractDataFeature end

function aggregate end
function reducesize end

include("../src/errors.jl")
include("../src/structs/dataset_structure.jl")

# ---------------------------------------------------------------------------- #
#                                DataFeature                                   #
# ---------------------------------------------------------------------------- #
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    values::Vector{String}
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
end

struct ScalarFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
end

struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    # hasmissing::Vector{Bool}}
    # hasnans::Vector{Bool}}
end

struct ReduceFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    # hasmissing::Vector{Bool}}
    # hasnans::Vector{Bool}}
end

# ---------------------------------------------------------------------------- #
#                            TreatmentGroup struct                             #
# ---------------------------------------------------------------------------- #
struct TreatmentGroup
    dims::Int
    vnames::Vector{Symbol}
    datatype::Type

    function TreatmentGroup(
        vnames::Vector{String},
        dims::Int,
        name_expr::Union{Regex,Base.Callable},
        datatype::Type
    )
        valid_names = name_expr isa Regex ?
        filter(item -> match(name_expr, item) !== nothing, vnames) :
        filter(name_expr, vnames)

        new(dims, Symbol.(valid_names), datatype)
    end

    TreatmentGroup(vnames::Vector{Symbol}, args...) = TreatmentGroup(String.(vnames), args...)

    TreatmentGroup(df::DataFrame, args...) = TreatmentGroup(names(df), args...)
end

function TreatmentGroup(;
    dims::Int,
    name_expr::Union{Regex,Base.Callable}=r".*",
    datatype::Type=Any,
    aggrtype::Base.Callable=aggregate
)
    x -> TreatmentGroup(x, dims, name_expr, datatype)
end

####################################################################

function get_column_structure(col::AbstractVector)
    datatype = Any
    dims = 0
    valididxs = Int[]
    missingidxs = Int[]
    nanidxs = Int[]
    hasmissing = Int[]
    hasnans = Int[]

    for i in eachindex(col)
        val = col[i]
        if ismissing(val)
            push!(missingidxs, i)
        elseif val isa AbstractFloat && isnan(val)
            push!(nanidxs, i)
        elseif val isa AbstractVector{<:AbstractFloat} || val isa AbstractArray{<:AbstractFloat}
            any(ismissing, val) && push!(hasmissing, i)
            any(isnan, val) && push!(hasnans, i)
            datatype = typeof(val)
            dims = ndims(val)
            push!(valididxs, i)
        elseif !(val isa AbstractFloat) || !isnan(val)
            if datatype == Any || !(datatype <: AbstractVector)
                datatype = typeof(val)
                push!(valididxs, i)
            end
        end
    end

    return datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans
end

function get_dataset_structure(dataset::Matrix)
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
            get_column_structure(@view(dataset[:, i]))
    end

    return DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
end

get_dataset_structure(df::DataFrame; kwargs...) = get_dataset_structure(Matrix(df))

######################################################################

function DataTreatment(dataset::Matrix, vnames::Vector{String}; kwargs...)
    ds_struct = get_dataset_structure(dataset)
end

DataTreatment(df::DataFrame; kwargs...) = DataTreatment(Matrix(df), names(df); kwargs...)

########################################################################

test = TreatmentGroup(dims=1, name_expr=r"^V")
test(df)

test = TreatmentGroup(dims=2)
test(df)

DataTreatment(df)

ds = get_dataset_structure(df)
# structure = get_structure(ds)


