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
    img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
)

abstract type AbstractDataFeature end

function aggregate end
function reducesize end


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

function check_column_structure(col::AbstractVector)
    datatype = Any
    valididxs = Int[]
    missingidxs = Int[]
    nanidxs = Int[]
    hasmissing = false
    hasnans = false

    for i in eachindex(col)
        val = col[i]
        if ismissing(val)
            push!(missingidxs, i)
        elseif val isa AbstractFloat && isnan(val)
            push!(nanidxs, i)
        elseif val isa AbstractVector{<:AbstractFloat} || val isa AbstractArray{<:AbstractFloat}
            any(ismissing, val) && (hasmissing = true)
            any(isnan, val) && (hasnans = true)
            datatype = typeof(val)
            push!(valididxs, i)
        elseif !(val isa AbstractFloat) || !isnan(val)
            if datatype == Any || !(datatype <: AbstractVector)
                datatype = typeof(val)
                push!(valididxs, i)
            end
        end
    end

    return datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans
end

function check_dataset_structure(dataset::Matrix, vnames::Vector{String})
    # nrows = size(dataset, 1)
    ncols = size(dataset, 2)

    datafeatures = Vector{AbstractDataFeature}(undef, ncols)

    Threads.@threads for i in axes(dataset, 2)
        # datafeatures[i] = check_column_structure(@view(dataset[:, i]), nrows)
        datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans = check_column_structure(@view(dataset[:, i]))
    end
end

######################################################################

function DataTreatment(dataset::Matrix, vnames::Vector{String}; kwargs...)
    check_dataset_structure(dataset, vnames)
end

DataTreatment(df::DataFrame; kwargs...) = DataTreatment(Matrix(df), names(df); kwargs...)

DataTreatment(df)

########################################################################

test = TreatmentGroup(dims=1, name_expr=r"^V")
test(df)

test = TreatmentGroup(dims=2)
test(df)

