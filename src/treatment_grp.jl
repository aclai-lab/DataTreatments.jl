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
include("../src/structs/metadata.jl")

function discrete_encode(X::Matrix)
    to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
    cats = [categorical(to_str.(col)) for col in eachcol(X)]
    return [levelcode.(cat) for cat in cats], levels.(cats)
end

# ---------------------------------------------------------------------------- #
#                               dataset structs                                #
# ---------------------------------------------------------------------------- #
struct DiscreteDataset
    dataset::Matrix
    info::Vector{DiscreteFeat}

    DiscreteDataset(dataset::Matrix, info::Vector{DiscreteFeat}) = new(dataset, info)
    
    function DiscreteDataset(
        id::Vector,
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int}
    )
        T = get_datatype(ds_struct, cols)
        vnames = get_vnames(ds_struct, cols)
        idx = get_valididxs(ds_struct, cols)
        miss = get_missingidxs(ds_struct, cols)
        codes, levels = discrete_encode(dataset[:, cols])

        return DiscreteDataset(
            stack(codes),
            [DiscreteFeat{T[i]}(push!(id, i), vnames[i], levels[i], idx[i], miss[i])
                for i in eachindex(vnames)]
        )
    end
end

struct ContinuousDataset{T}
    dataset::Matrix
    info::Vector{ContinuousFeat}

    ContinuousDataset(dataset::Matrix, info::Vector{ContinuousFeat{T}}) where T =
        new{T}(dataset, info)

    function ContinuousDataset(
        id::Vector,
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int},
        float_type::Type
    )
        vnames = get_vnames(ds_struct, cols)
        idx = get_valididxs(ds_struct, cols)
        miss = get_missingidxs(ds_struct, cols)
        nan = get_nanidxs(ds_struct, cols)

        return ContinuousDataset(
            reduce(hcat, [map(x -> ismissing(x) ? missing : float_type(x), @view dataset[:, col])
                for col in cols]),
            [ContinuousFeat{float_type}(push!(id, i), vnames[i], idx[i], miss[i], nan[i])
                for i in eachindex(vnames)]
        )
    end
end

struct AggregatedMultidimDataset{T}
    dataset::Matrix
    info::Vector{AggregateFeat}

    AggregatedMultidimDataset(dataset::Matrix, info::Vector{ContinuousFeat{T}}) where T =
        new{T}(dataset, info)

    function AggregatedMultidimDataset(
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int}
    )

    end
end

struct ReducedMultidimDataset{T}
    dataset::Matrix
    info::Vector{ReduceFeat}

    ReducedMultidimDataset(dataset::Matrix, info::Vector{ContinuousFeat{T}}) where T =
        new{T}(dataset, info)

    function ReducedMultidimDataset(
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int}
    )

    end
end


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
        ds_struct = DatasetStructure(dataset, vnames)
        t_groups = [treat(ds_struct) for treat in treatments]

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
    get_ds_struct(dt::DataTreatment)

Returns the dataset structure containing metadata about the dataset.
"""
get_ds_struct(dt::DataTreatment) = dt.ds_struct

"""
    get_t_groups(dt::DataTreatment)
    get_t_groups(dt::DataTreatment, i::Int)

Returns the treatment groups. If `i` is provided, returns the `i`-th treatment group.
"""
get_t_groups(dt::DataTreatment) = dt.t_groups
get_t_groups(dt::DataTreatment, i::Int) = dt.t_groups[i]

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
#                               dataset builder                                #
# ---------------------------------------------------------------------------- #
get_features(a::Base.Callable) = a.features
get_reducefunc(r::Base.Callable) = r.reducefunc

function build_datasets(
    id::Vector,
    dataset::Matrix,
    ds_struct::DatasetStructure,
    idxs::Vector{Int},
    aggrfunc::Base.Callable,
    float_type::Type=Float64
)
    # dataset = @views dataset[:, idxs]

    # ds_td = ds_tc = ds_md = nothing
    # vnames = get_vnames(ds_struct)
    valtype = get_datatype(ds_struct)

    # idx = get_valididxs(ds_struct)
    # missingidx = get_missingidxs(ds_struct)
    # nanidx = get_nanidxs(ds_struct)

    td_cols = idxs ∩ findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    ds_td = isempty(td_cols) ? [] : DiscreteDataset(id, dataset, ds_struct, td_cols)
    ds_tc = isempty(tc_cols) ? [] : ContinuousDataset(id, dataset, ds_struct, tc_cols, float_type)

    # discrete
    # if !isempty(td_cols)
        # T = get_datatype(ds_struct, td_cols)
        # vnames_td = get_vnames(ds_struct, td_cols)
        # idx = get_valididxs(ds_struct, td_cols)
        # miss_td = get_missingidxs(ds_struct, td_cols)
        # codes, levels = discrete_encode(dataset[:, td_cols])

        # ds_td = DiscreteDataset(
        #     stack(codes),
        #     [DiscreteFeat{T[i]}(push!(id, i), vnames_td[i], levels[i], idx[i], miss_td[i])
        #         for i in eachindex(vnames_td)]
        # )
        
    # end

    # continue
    # if !isempty(tc_cols)
    #     vnames_tc = get_vnames(ds_struct, tc_cols)
    #     idx = get_valididxs(ds_struct, tc_cols)
    #     miss_tc = get_missingidxs(ds_struct, tc_cols)
    #     nan_tc = get_nanidxs(ds_struct, tc_cols)

    #     ds_tc = ContinuousDataset(
    #         reduce(hcat, [map(x -> ismissing(x) ? missing : float_type(x), @view dataset[:, col])
    #             for col in tc_cols]),
    #         [ContinuousFeat{float_type}(push!(id, i), vnames_tc[i], idx[i], miss_tc[i], nan_tc[i]) for i in eachindex(vnames_tc)]
    #     )
    # end

    # multidimensional
    if !isempty(md_cols)
        dataset = @view dataset[:, md_cols]
        vnames_md = get_vnames(ds_struct, md_cols)
        idx = get_valididxs(ds_struct, md_cols)
        miss_md = get_missingidxs(ds_struct, md_cols)
        nan_md = get_nanidxs(ds_struct, md_cols)
        hasmiss_md = get_hasmissing(ds_struct, md_cols)
        hasnan_md = get_hasnans(ds_struct, md_cols)

        md, nwindows = aggrfunc(dataset, idx, float_type)

        if hasfield(typeof(aggrfunc), :features)
            md_feats = vec([AggregateFeat{float_type}(push!(id, i), vnames_md[c], f, nwindows[c], idx[c], miss_md[c], nan_md[c], hasmiss_md[c], hasnan_md[c])
                    for (i, (f, c)) in enumerate(Iterators.product(get_features(aggrfunc), axes(dataset,2)))])
            ds_md = AggregatedMultidimDataset(md, md_feats)
        elseif hasfield(typeof(aggrfunc), :reducefunc)
            md_feats = [ReduceFeat{AbstractArray{float_type}}(push!(id, i), vnames_md[c], get_reducefunc(aggrfunc), idx[c], miss_md[c], nan_md[c], hasmiss_md[c], hasnan_md[c])
                for (i, c) in enumerate(axes(dataset,2))]
            ds_md = ReducedMultidimDataset(md, md_feats)
        else
            error("aggrfunc must have either a `features` field (aggregate) or a `reducefunc` field (reducesize), got: $(typeof(aggrfunc))")
        end
    end

    # @show ds_md

    # return ds_td, ds_tc, ds_md
end

# ---------------------------------------------------------------------------- #
#                             custom lazy methods                              #
# ---------------------------------------------------------------------------- #
function get_datasets(dt::DataTreatment; split=true, dataframe=false)
    dataset = get_dataset(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    # first step: defines datasets based on treatment groups
    treats = get_t_groups(dt)
    idxs = get_idxs(treats)

    for i in eachindex(treats)
        build_datasets(
            [:treatment_group, i],
            dataset,
            ds_struct,
            idxs[i],
            get_aggrfunc(treats[i]),
            float_type
        )
    end

    # second step: defines datasets on leftover indicies
end

function get_datasets(dt::DataTreatment, grp::TreatmentGroup; dataframe=false)

end

########################################################################

# test = DataTreatment(df, TreatmentGroup(dims=1, name_expr=r"^V"))

test = DataTreatment(
    df,
    TreatmentGroup(dims=0),
    TreatmentGroup(name_expr=r"^V"),
    TreatmentGroup(dims=1),
    TreatmentGroup(dims=2, aggrfunc=reducesize())
)

a=get_datasets(test)
# test = DataTreatment(df)

# TreatmentGroup(win=wholewindow(), features=(maximum, minimum, mean))

