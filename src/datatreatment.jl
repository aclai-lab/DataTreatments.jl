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
    valtype = get_datatype(ds_struct)

    td_cols = idxs ∩ findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    ds_td = isempty(td_cols) ? [] : DiscreteDataset(id, dataset, ds_struct, td_cols)
    ds_tc = isempty(tc_cols) ? [] : ContinuousDataset(id, dataset, ds_struct, tc_cols, float_type)
    ds_md = isempty(md_cols) ? [] : MultidimDataset(id, dataset, ds_struct, md_cols, aggrfunc, float_type)

    return ds_td, ds_tc, ds_md
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