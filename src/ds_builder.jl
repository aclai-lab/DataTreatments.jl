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
function build_dataset(
    X::Matrix;
    aggrtype::Symbol=:aggregate,
    vnames::Union{Vector{Symbol},Nothing}=[Symbol("V$i") for i in 1:size(X, 2)],
    win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}}=wholewindow(),
    # features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    kwargs...
)
    valtype, hasmissing, hasnan = check_integrity(X)

    td_cols = findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    if !isempty(md_cols)
        Xmd = @view X[:, md_cols]
        uniform = has_uniform_element_size(Xmd)
        win isa Base.Callable && (win = (win,))
        intervals = @evalwindow first(Xmd) win...
        nwindows = prod(length.(intervals))
    end

    # X, features = _build_dataset(X; aggrtype, vnames, kwargs...)
    results = map(axes(X, 2)) do i
        _build_dataset(pinfo[i]..., @view(X[:, i]); aggrtype, vname=vnames[i], kwargs...)
    end

    return reduce(hcat, first.(results)), last.(results)
end

function _build_dataset(
    T::Type,
    hasmissing::Bool,
    hasnan::Bool,
    x::AbstractVector;
    vname::Symbol,
    kwargs...
)
    return x, TabularFeat{T}(0, T, vname, hasmissing, hasnan)
end

function _build_dataset(
    T::AbstractArray,
    hasmissing::Bool,
    hasnan::Bool,
    x::AbstractVector;
    vnames::Symbol,
    aggrtype::Symbol=:aggregate,
    win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}}=wholewindow(),
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    reducefunc::Base.Callable=mean,
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