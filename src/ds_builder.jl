# ---------------------------------------------------------------------------- #
#                           nan/missing handle utils                           #
# ---------------------------------------------------------------------------- #
function base_eltype(col::AbstractVector)
    valtype, hasmissing, hasnan = nothing, false, false
    for val in col
        if ismissing(val)
            hasmissing = true
        elseif val isa AbstractFloat
            isnan(val) && (hasnan = true)
            isnothing(valtype) && (valtype = Float64)
        elseif val isa AbstractVector{<:AbstractFloat}
            if any(isnan, val)
                hasnan = true
            end
            isnothing(valtype) && (valtype = typeof(val))
        else
            isnothing(valtype) && (valtype = typeof(val))
        end
        !isnothing(valtype) && hasmissing && hasnan && break
    end
    return valtype, hasmissing, hasnan
end

function check_integrity(X::Matrix{T}) where T
    valtype = Vector{Type}(undef, size(X, 2))
    hasmissing = Vector{Bool}(undef, size(X, 2))
    hasnan = Vector{Bool}(undef, size(X, 2))

    Threads.@threads for i in axes(X, 2)
        valtype[i], hasmissing[i], hasnan[i] = base_eltype(@view(X[:, i]))
    end
    return valtype, hasmissing, hasnan
end

# ---------------------------------------------------------------------------- #
#                               dataset builder                                #
# ---------------------------------------------------------------------------- #
function build_datasets(
    X::Matrix;
    aggrtype::Symbol=:aggregate,
    vnames::Union{Vector{Symbol},Nothing}=[Symbol("V$i") for i in 1:size(X, 2)],
    win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}}=wholewindow(),
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    float_type::DataType=Float64,
    kwargs...
)
    Xtd = Xtc = Xmd = td_feats = tc_feats = md_feats = nothing
    valtype, hasmissing, hasnan = check_integrity(X)

    td_cols = findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    if !isempty(td_cols)
        n_td_cols = length(td_cols)
        Xtd = Matrix{<:Discrete}(undef, (size(X,1), n_td_cols))
    end

    if !isempty(tc_cols)
        vnames_tc = @views vnames[tc_cols]
        miss_tc, nan_tc = hasmissing[tc_cols], hasnan[tc_cols]

        Xtc = @views float_type.(X[:, tc_cols])
        tc_feats = [ScalarFeat{float_type}(i, float_type, vnames[i], miss_tc[i], nan_tc[i]) for i in eachindex(vnames_tc)]
    end

    if !isempty(md_cols)
        vnames_md = @views vnames[md_cols]
        miss_md, nan_md = hasmissing[md_cols], hasnan[md_cols]

        _X = @view X[:, md_cols]
        uniform = has_uniform_element_size(_X)
        win isa Base.Callable && (win = (win,))
        intervals = @evalwindow first(_X) win...
        nwindows = prod(length.(intervals))

        if aggrtype == :aggregate
            Xmd = DataTreatments.aggregate(_X, intervals; features, win, uniform, float_type)

            md_feats = if nwindows == 1
                # single window: apply to whole time series
                vec([AggregateFeat{float_type}(i, float_type, vnames_md[c], f, 1, miss_md[c], nan_md[c])
                    for (i, (f, c)) in enumerate(Iterators.product(features, axes(_X,2)))])
            else
                # multiple windows: apply to each interval
                vec([AggregateFeat{float_type}(i, float_type, vnames_md[c], f, n, miss_md[c], nan_md[c])
                    for (i, (n, f, c)) in enumerate(Iterators.product(1:nwindows, features, axes(_X,2)))])
            end

        elseif aggrtype == :reducesize
            Xmd = DataTreatments.reducesize(X, intervals; reducefunc, win, uniform)
            md_feats = [ReduceFeat{AbstractArray{float_type}}(i, T, vnames_md[c], reducefunc, miss_md[c], nan_md[c])
                for (i, c) in enumerate(axes(X,2))]

        else
            error("Unknown treatment type: $treat")
        end
    end

    return Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats
end

function _build_dataset(
    T::Type,
    x::AbstractVector;
    vname::Symbol,
    hasmissing::Bool,
    hasnan::Bool,
    kwargs...
)
    return x, ScalarFeat{T}(0, T, vname, hasmissing, hasnan)
end

function _build_dataset(
    T::AbstractArray,
    x::AbstractVector;
    vnames::Symbol,
    aggrtype::Symbol=:aggregate,
    win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}}=wholewindow(),
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    reducefunc::Base.Callable=mean,
    hasmissing::Bool,
    hasnan::Bool,
    kwargs...
)
    # uniform size check
    # if elements differ in size, only adaptivewindow or wholewindow are allowed
    # (computing the window per-element is a significant overhead otherwise)
    uniform = has_uniform_element_size(x)

    # convert to float
    T isa AbstractFloat || (X = DataTreatments.convert(X))

    win isa Base.Callable && (win = (win,))
    intervals = @evalwindow first(X) win...
    nwindows = prod(length.(intervals))

    return if aggrtype == :aggregate begin
        (DataTreatments.aggregate(X, intervals; features, win, uniform),
        if nwindows == 1
            # single window: apply to whole time series
            vec([AggregateFeat{T}(i, T, vnames[c], f, 1)
                for (i, (f, c)) in enumerate(Iterators.product(features, axes(X,2)))])
        else
            # multiple windows: apply to each interval
            vec([AggregateFeat{T}(i, T, vnames[c], f, n)
                for (i, (n, f, c)) in enumerate(Iterators.product(1:nwindows, features, axes(X,2)))])
        end
        )
    end

    elseif aggrtype == :reducesize begin
        (DataTreatments.reducesize(X, intervals; reducefunc, win, uniform),
        [ReduceFeat{T}(i, T, vnames[c], reducefunc) for (i, c) in enumerate(axes(X,2))]
        )
    end

    else
        error("Unknown treatment type: $treat")
    end

    # grp_result = if !isnothing(groups)
    #     fields = collect(groups)
    #     groupidxs, _ = _groupby(Xresult, Xinfo, fields)
    #     [GroupResult(groupidx, fields) for groupidx in groupidxs]
    # else
    #     nothing
    # end

    # if !isnothing(norm)
    #     norm isa Type{<:AbstractNormalization} && (norm = norm())
    #     if isnothing(grp_result)
    #         Xresult = normalize(Xresult, norm)
    #     else
    #         Threads.@threads for g in grp_result
    #             Xresult[:, get_group(g)] =
    #                 normalize(Xresult[:, get_group(g)], norm)
    #         end
    #     end
    # end
end