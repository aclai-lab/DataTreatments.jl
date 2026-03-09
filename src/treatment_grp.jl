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
using DataTreatments: aggregate, reducesize, wholewindow
include("../src/structs/dataset_structure.jl")
include("../src/structs/treatment_group.jl")

function discrete_encode(X::Matrix)
    to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
    cats = [categorical(to_str.(col)) for col in eachcol(X)]
    return [levelcode.(cat) for cat in cats], levels.(cats)
end

# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
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
        new{T}(id, vname, levels, valididxs, missingidxs)
    end
end

struct ContinuousFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
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
        new{T}(id, vname, valididxs, missingidxs, nanidxs)
    end
end

struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Bool}
    hasnans::Vector{Bool}

    function AggregateFeat{T}(
        id::Int,
        vname::Union{String,Symbol},
        feat::Base.Callable,
        nwin::Int,
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int},
        hasmissing::Vector{Bool},
        hasnans::Vector{Bool}
    ) where T
        new{T}(id, vname, feat, nwin, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end
end

struct ReduceFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Bool}
    hasnans::Vector{Bool}

    function ReduceFeat{T}(
        id::Int,
        vname::Union{String,Symbol},
        reducefunc::Base.Callable,
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int},
        hasmissing::Vector{Bool},
        hasnans::Vector{Bool}
    ) where T
        new{T}(id, vname, reducefunc, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
    end
end

# ---------------------------------------------------------------------------- #
#                               dataset structs                                #
# ---------------------------------------------------------------------------- #
struct DiscreteDataset

end

struct ContinuousDataset

end

struct AggregatedMultidimDataset

end

struct ReducedMultidimDataset

end

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_id(f::AbstractDataFeature) -> Int

Returns the column id of the feature.
"""
get_id(f::AbstractDataFeature) = f.id

"""
    get_vname(f::AbstractDataFeature) -> String

Returns the variable name of the feature.
"""
get_vname(f::AbstractDataFeature) = f.vname

"""
    get_valididxs(f::AbstractDataFeature) -> Vector{Int}

Returns the indices of valid values for the feature.
"""
get_valididxs(f::AbstractDataFeature) = f.valididxs

"""
    get_missingidxs(f::AbstractDataFeature) -> Vector{Int}

Returns the indices of `missing` values for the feature.
"""
get_missingidxs(f::AbstractDataFeature) = f.missingidxs

"""
    get_nanidxs(f::Union{ContinuousFeat,AggregateFeat,ReduceFeat}) -> Vector{Int}

Returns the indices of `NaN` values for the feature.
"""
get_nanidxs(f::Union{ContinuousFeat,AggregateFeat,ReduceFeat}) = f.nanidxs

"""
    get_hasmissing(f::Union{AggregateFeat,ReduceFeat}) -> Vector{Bool}

Returns the per-element flags indicating presence of `missing` values internally.
"""
get_hasmissing(f::Union{AggregateFeat,ReduceFeat}) = f.hasmissing

"""
    get_hasnans(f::Union{AggregateFeat,ReduceFeat}) -> Vector{Bool}

Returns the per-element flags indicating presence of `NaN` values internally.
"""
get_hasnans(f::Union{AggregateFeat,ReduceFeat}) = f.hasnans

"""
    get_levels(f::DiscreteFeat) -> CategoricalArrays.CategoricalVector

Returns the categorical levels of the discrete feature.
"""
get_levels(f::DiscreteFeat) = f.levels

"""
    get_feat(f::AggregateFeat) -> Base.Callable

Returns the aggregation function of the feature.
"""
get_feat(f::AggregateFeat) = f.feat

"""
    get_nwin(f::AggregateFeat) -> Int

Returns the number of windows of the aggregate feature.
"""
get_nwin(f::AggregateFeat) = f.nwin

"""
    get_reducefunc(f::ReduceFeat) -> Base.Callable

Returns the reduce function of the feature.
"""
get_reducefunc(f::ReduceFeat) = f.reducefunc


# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
struct DataTreatment
    dataset::Matrix
    ds_struct::DatasetStructure
    t_groups::Vector{TreatmentGroup}
    float_type::Type

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
        new(dataset, ds_struct, t_groups, float_type)
    end

    DataTreatment(df::DataFrame, args...; kwargs...) =
        DataTreatment(Matrix(df), names(df), args...; kwargs...)
end

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
"""
    Base.size(dt::DataTreatment)

Returns the size of the dataset as a tuple `(nrows, ncols)`.
"""
Base.size(dt::DataTreatment) = size(dt.dataset)

"""
    Base.length(dt::DataTreatment)

Returns the number of treatment groups.
"""
Base.length(dt::DataTreatment) = length(dt.t_groups)

"""
    Base.ndims(dt::DataTreatment)

Returns the number of dimensions in the dataset (always 2 for a matrix).
"""
Base.ndims(dt::DataTreatment) = 2

"""
    Base.iterate(dt::DataTreatment, state=1)

Iterates over the treatment groups.
"""
Base.iterate(dt::DataTreatment, state=1) = state > length(dt) ? nothing : (dt.t_groups[state], state + 1)

"""
    Base.eachindex(dt::DataTreatment)

Returns the indices of the treatment groups.
"""
Base.eachindex(dt::DataTreatment) = eachindex(dt.t_groups)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_dataset(dt::DataTreatment)

Returns the raw dataset matrix.
"""
get_dataset(dt::DataTreatment) = dt.dataset

"""
    get_dataset_structure(dt::DataTreatment)

Returns the dataset structure containing metadata about the dataset.
"""
get_dataset_structure(dt::DataTreatment) = dt.ds_struct

"""
    get_treatment_groups(dt::DataTreatment)
    get_treatment_groups(dt::DataTreatment, i::Int)

Returns the treatment groups. If `i` is provided, returns the `i`-th treatment group.
"""
get_treatment_groups(dt::DataTreatment) = dt.t_groups
get_treatment_groups(dt::DataTreatment, i::Int) = dt.t_groups[i]

"""
    get_float_type(dt::DataTreatment)

Returns the floating-point type used for processing.
"""
get_float_type(dt::DataTreatment) = dt.float_type

"""
    get_nrows(dt::DataTreatment)

Returns the number of rows in the dataset.
"""
get_nrows(dt::DataTreatment) = size(dt.dataset, 1)

"""
    get_ncols(dt::DataTreatment)

Returns the number of columns in the dataset.
"""
get_ncols(dt::DataTreatment) = size(dt.dataset, 2)

# ---------------------------------------------------------------------------- #
#                             custom lazy methods                              #
# ---------------------------------------------------------------------------- #
function get_dataset(dt::DataTreatment; split=true, dataframe=false)
    
    # first step: defines datasets based on treatment groups

    # second step: defines datasets on leftover indicies
end

function get_dataset(dt::DataTreatment, grp::TreatmentGroup; dataframe=false)

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

