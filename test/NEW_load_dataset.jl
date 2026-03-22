_isnanval(v) = v isa AbstractFloat && isnan(v)
_isarray(v) = v isa AbstractArray

_to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
_discrete_encode(X::Matrix) = [_discrete_encode(x) for x in eachcol(X)]
_discrete_encode(x::AbstractVector) = [levelcode(cat) for cat in categorical(_to_str.(x))]

function _collect_dataset_info(data::Matrix)
        ncols = size(data, 2)

        id = Vector{Int}(undef, ncols)
        datatype = Vector{Type}(undef, ncols)
        dims = Vector{Int}(undef, ncols)
        valididxs = Vector{Vector{Int}}(undef, ncols)
        missingidxs = Vector{Vector{Int}}(undef, ncols)
        nanidxs = Vector{Vector{Int}}(undef, ncols)
        hasmissing = Vector{Vector{Int}}(undef, ncols)
        hasnans = Vector{Vector{Int}}(undef, ncols)

        Threads.@threads for i in axes(data, 2)
            id[i] = i
            col = @view(data[:, i])

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

        return (
            id=id,
            datatype=datatype,
            dims=dims,
            valididxs=valididxs,
            missingidxs=missingidxs,
            nanidxs=nanidxs,
            hasmissing=hasmissing,
            hasnans=hasnans
        )
end

function _split_md_by_dims(ds_md::MultidimDataset)
    dims = get_dims(ds_md)
    unique_dims = unique(get_dims(ds_md))

    idxs = [filter(i -> dims[i] == ud, collect(eachindex(dims))) for ud in unique_dims]

    return [ds_md[idx] for idx in idxs]
end

function _build_ds(
    ids::Vector{Int},
    treat::TreatmentGroup,
    data::Matrix,
    vnames::Vector{String},
    datastruct::NamedTuple,
    float_type::Type
)
    aggrfunc = get_aggrfunc(treat)
    valtype = datastruct.datatype
    groups = has_groupby(treat) ? get_groupby(treat) : nothing

    td_ids = ids ∩ findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_ids = ids ∩ findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_ids = ids ∩ findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    ds_td = isempty(td_ids) ?
        nothing :
        DiscreteDataset(td_ids, data, vnames, datastruct)
    ds_tc = isempty(tc_ids) ?
        nothing :
        ContinuousDataset(tc_ids, data, vnames, datastruct, float_type)
    ds_md = isempty(md_ids) ?
        nothing :
        MultidimDataset(md_ids, data, vnames, datastruct, aggrfunc, float_type, groups)

    return ds_td, ds_tc, ds_md
end

function _treatments_ds(
    data::Matrix,
    vnames::Vector{String},
    datastruct::NamedTuple,
    treats::Vector{<:TreatmentGroup},
    float_type::Type
)
    ntreats = length(treats)
    ds_td = Vector{Union{Nothing,DiscreteDataset}}(undef, ntreats)
    ds_tc = Vector{Union{Nothing,ContinuousDataset}}(undef, ntreats)
    ds_md = Vector{Union{Nothing,MultidimDataset}}(undef, ntreats)

    Threads.@threads for i in eachindex(treats)
        ds_td[i], ds_tc[i], ds_md[i] = _build_ds(
            get_ids(treats[i]),
            treats[i],
            data,
            vnames,
            datastruct,
            float_type
        )
    end

    td_filtered = filter(!isnothing, ds_td)
    tc_filtered = filter(!isnothing, ds_tc)
    md_filtered = filter(!isnothing, ds_md)

    md_split = isempty(md_filtered) ? MultidimDataset[] : reduce(vcat, _split_md_by_dims.(md_filtered))

    return td_filtered, tc_filtered, md_split
end

function _leftovers_ds(
    data::Matrix,
    vnames::Vector{String},
    datastruct::NamedTuple,
    treats::Vector{<:TreatmentGroup},
    float_type::Type
)
    ids = setdiff(datastruct.id, reduce(vcat, get_ids(treats)))

    ds_td, ds_tc, ds_md = _build_ds(
        ids,
        TreatmentGroup(datastruct, vnames; aggrfunc=DefaultAggrFunc),
        data,
        vnames,
        datastruct,
        float_type
    )

    td_filtered = isnothing(ds_td) ? AbstractDataset[] : AbstractDataset[ds_td]
    tc_filtered = isnothing(ds_tc) ? AbstractDataset[] : AbstractDataset[ds_tc]
    md_split = isnothing(ds_md) ? AbstractDataset[] : _split_md_by_dims(ds_md)

    return td_filtered, tc_filtered, md_split
end
