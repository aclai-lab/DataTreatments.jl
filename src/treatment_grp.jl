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

include("../src/errors.jl")
using DataTreatments: reducesize, wholewindow, @evalwindow
include("../src/structs/dataset_structure.jl")
include("../src/structs/treatment_group.jl")

function discrete_encode(X::Matrix)
    to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
    cats = [categorical(to_str.(col)) for col in eachcol(X)]
    return [levelcode.(cat) for cat in cats], levels.(cats)
end

# ---------------------------------------------------------------------------- #
#                                DataFeature                                   #
# ---------------------------------------------------------------------------- #
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    levels::CategoricalArrays.CategoricalVector
    valididxs::Vector{Int}
    missingidxs::Vector{Int}

    function DiscreteFeat{T}(
        id::Int,
        vname::Union{String,Symbol},
        levels::CategoricalArrays.CategoricalVector,
        valididxs::Vector{Int},
        missingidxs::Vector{Int}
    ) where T
        new{T}(id, Symbol(vname), levels, valididxs, missingidxs)
    end
end

struct ContinuousFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}

    function ContinuousFeat{T}(
        id::Int,
        vname::Union{String,Symbol},
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int}
    ) where T
        new{T}(id, Symbol(vname), valididxs, missingidxs, nanidxs)
    end
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

######################################################################

function aggregate(
    X::AbstractArray,
    idx::AbstractVector{Vector{Int}},
    float_type::Type;
    win::Tuple{Vararg{Base.Callable}},
    features::Tuple{Vararg{Base.Callable}},
)
    colwin = [[n > length(win) ? last(win) : win[n] for n in 1:ndims(X[first(idx[i]), i])] for i in axes(X, 2)]
    nwindows = [prod(hasfield(typeof(w), :nwindows) ? w.nwindows : 1 for w in c) for c in colwin]
    nfeats = length(features)
# vettore di indici
    Xa = Matrix{Union{Missing,float_type}}(undef, size(X, 1), sum(nwindows) * nfeats)
    outtmp = 1

    @inbounds for colidx in axes(X, 2)
        outidx = outtmp

        for rowidx in axes(X, 1)
            x = X[rowidx, colidx]
            outidx = outtmp

            if rowidx in idx[colidx]
                intervals = @evalwindow X[rowidx, colidx] colwin[colidx]...
                for feat in features
                    for cartidx in CartesianIndices(length.(intervals))
                        ranges = get_window_ranges(intervals, cartidx)
                        window_view = @views x[ranges...]
                        Xa[rowidx, outidx] = safe_feat(reshape(window_view, :), feat)
                        outidx += 1
                    end
                end
            else
                intervals = nwindows[colidx] * nfeats
                Xa[rowidx, outidx:outidx+intervals-1] .= ismissing(x) ? x : float_type(x)
                outidx += intervals
            end
        end

        outtmp = outidx
    end

    return Xa, nwindows
end

aggregate(; kwargs...) = (x, i, ft) -> aggregate(x, i, ft; kwargs...)

######################################################################

struct DataTreatment
    dataset::Matrix
    ds_struct::DatasetStructure
    t_groups::Vector{TreatmentGroup}

    function DataTreatment(
        dataset::Matrix,
        vnames::Vector{String},
        treatments::Base.Callable...=TreatmentGroup(
            aggrfunc=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
        );
        float_type::Type=Float64
    )
        ds_struct = get_dataset_structure(dataset, vnames)
        t_groups = [treat(ds_struct) for treat in treatments]
        # build_datasets(dataset, ds_struct, vnames, float_type)
        new(dataset, ds_struct, t_groups)
    end

    DataTreatment(df::DataFrame, args...; kwargs...) =
        DataTreatment(Matrix(df), names(df), args...; kwargs...)
end

########################################################################

# test = DataTreatment(df, TreatmentGroup(dims=1, name_expr=r"^V"))

test = DataTreatment(
    df,
    TreatmentGroup(dims=0),
    TreatmentGroup(name_expr=r"^V"),
    TreatmentGroup(dims=2)
)

# test = DataTreatment(df)

# TreatmentGroup(win=wholewindow(), features=(maximum, minimum, mean))

