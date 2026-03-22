struct TreatmentGroup{T}
    ids::Vector{Int}
    dims::Int
    vnames::Vector{String}
    aggrfunc::Base.Callable
    grouped::Bool
    groupby::Union{Nothing,Tuple{Vararg{Symbol}}}

    function TreatmentGroup(
        datastruct::NamedTuple,
        vnames::Vector{String};
        dims::Int=-1,
        name_expr::Union{Regex,Base.Callable,Vector{String}}=r".*",
        datatype::Type=Any,
        aggrfunc::Base.Callable=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
        grouped::Bool=false,
        groupby::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}=nothing
    )
        all_dims = datastruct.dims
        all_types = datastruct.datatype
        groupby isa Symbol && (groupby = (groupby,))

        name_match = if name_expr isa Regex
            n -> match(name_expr, n) !== nothing
        elseif name_expr isa Vector{String}
            name_set = Set(name_expr)
            n -> n in name_set
        else
            name_expr
        end

        ids = Int[]

        for i in datastruct.id
            (dims != -1 && all_dims[i] != dims) && continue
            (datatype != Any && all_types[i] != datatype) && continue
            name_match(vnames[i]) || continue
            push!(ids, i)
        end

        col_types = all_types[ids]
        T = isempty(col_types) ? Any : mapreduce(identity, typejoin, col_types)

        new{T}(ids, dims, vnames[ids], aggrfunc, grouped, groupby)
    end
end

TreatmentGroup(; kwargs...) = (d, v) -> TreatmentGroup(d, v; kwargs...)

get_ids(t::Vector{<:TreatmentGroup}) = get_ids.(t)
get_ids(t::TreatmentGroup) = t.ids

get_aggrfunc(t::TreatmentGroup) = t.aggrfunc

has_groupby(t::TreatmentGroup) = !isnothing(t.groupby)
get_groupby(t::TreatmentGroup) = t.groupby

const DefaultAggrFunc = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))
const DefaultGrouped = false
const DefaultTreatmentGroup = TreatmentGroup(aggrfunc=DefaultAggrFunc, grouped=DefaultGrouped)
