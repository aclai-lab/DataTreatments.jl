using Test

using MLJ
using DataFrames, Random
using SoleData: Artifacts

using DataTreatments: aggregate, wholewindow, adaptivewindow
using CategoricalArrays

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                               Dataset struct                                 #
# ---------------------------------------------------------------------------- #
# mutable struct DataTreatment
#     data::Vector{DT.AbstractDataset}
#     treats::Vector{DT.TreatmentGroup}
# end

include("NEW_treatment_group.jl")
include("NEW_output_datasets.jl")
include("NEW_load_dataset.jl")


function load_dataset(
    data::Matrix,
    vnames::Vector{String}=["V$i" for i in 1:size(data, 2)],
    target::Union{Nothing,AbstractVector}=nothing,
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    treatment_ds::Bool=true,
    leftover_ds::Bool=true,
    # data_type::Base.Callable=tabular,
    float_type::Type=Float64
)
    datastruct = _collect_dataset_info(data)

    ctarget, clevels = if !isnothing(target) && !isa(eltype(target), AbstractFloat)
        _discrete_encode(target), levels(target)
    else
        target, nothing
    end

    treats = [treat(datastruct, vnames) for treat in treatments]

    ds = AbstractDataset[]

    if treatment_ds
        ds_td, ds_tc, ds_md = _treatments_ds(data, vnames, datastruct, treats, float_type)
        !isnothing(ds_td) && append!(ds, ds_td)
        !isnothing(ds_tc) && append!(ds, ds_tc)
        !isnothing(ds_md) && append!(ds, ds_md)
    end

    if leftover_ds
        ds_td, ds_tc, ds_md = _leftovers_ds(data, vnames, datastruct, treats, float_type)
        !isnothing(ds_td) && append!(ds, ds_td)
        !isnothing(ds_tc) && append!(ds, ds_tc)
        !isnothing(ds_md) && append!(ds, ds_md)
    end

    # data, treats = data_type(dt, args...; kwargs...)

    # data, treats = get_dataset(dt::DataTreatment, args...; kwargs...)
    
    # return vcat(get_discrete(data), get_continuous(data), get_aggregated(data)), treats

    # Dataset(data, treats)
    ds
end

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), args...; kwargs...)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

ds = load_dataset(Xc, yc)

ds =load_dataset(
    Xc, yc,
    TreatmentGroup(name_expr=["petal_length", "petal_width"], grouped=true)
)

ds =load_dataset(Xr, yr)

ds =load_dataset(
    Xts, yts,
    TreatmentGroup(
        dims=1,
        aggrfunc=aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )
    )
)

# ds =load_dataset(
#     Xts, yts,
#     TreatmentGroup(
#         dims=1,
#         aggrfunc=SF.reducesize(
#             reducefunc=mean,
#             win=(splitwindow(nwindows=5),)
#         )
#     );
#     data_type=multidim
# )

# ds =load_dataset(
#     Xts, yts,
#     TreatmentGroup(
#         dims=1,
#         aggrfunc=SF.reducesize()
#     );
#     data_type=tabular
# )
# @test isempty(SF.get_data(ds))