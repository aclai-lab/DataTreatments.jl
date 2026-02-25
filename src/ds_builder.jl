# ---------------------------------------------------------------------------- #
#                               dataset builder                                #
# ---------------------------------------------------------------------------- #
function build_dataset(
    X::Dataset;
    aggrtype::Symbol=:aggregate,
    kwargs...
) where T
    if T == Any
        # caso speciale: non è uniforme.
        # costruire un vettore dei tipi colonna?
        # costruire i groupby?
        # aggregare o ridurre solo le colonne multidim,
        # se presenti
    end

    # caso normale: è uniforme
    # ma dobbiamo splittare se multidimensionale
    X, features = _build_dataset(X; aggrtype, kwargs...)
end

function _build_dataset(
    X::Matrix{T};
    vnames::Vector{Symbol},
    kwargs...
) where T
    # l'output sarà X e vnames
    # vnames non viene alterato
    # X necessita di normalizzazione?
    # X ha missing?

    # if !isnothing(norm)
    #     norm isa Type{<:AbstractNormalization} && (norm = norm())
    #     X = normalize(X, norm)
    # end

    # features = [TabularFeat{T}(v) for v in vnames]

    # per ora ritorno X pulito
    return X, [TabularFeat{T}(T, v) for v in vnames]
end

function _build_dataset(
    X::Matrix{T};
    vnames::Vector{Symbol},
    aggrtype::Symbol=:aggregate,
    win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}}=wholewindow(),
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    reducefunc::Base.Callable=mean,
    kwargs...
) where {T<:AbstractVector}
    # uniform size check
    # if elements differ in size, only adaptivewindow or wholewindow are allowed
    # (computing the window per-element is a significant overhead otherwise)
    uniform = has_uniform_element_size(X)

    # convert to float
    T isa AbstractFloat || (X = DataTreatments.convert(X))

    win isa Base.Callable && (win = (win,))
    intervals = @evalwindow first(X) win...
    nwindows = prod(length.(intervals))

    return if aggrtype == :aggregate begin
        (DataTreatments.aggregate(X, intervals; features, win, uniform),
        if nwindows == 1
            # single window: apply to whole time series
            [AggregateFeat{T}(T, vnames[c], f, 1) for f in features, c in axes(X,2)] |> vec
        else
            # multiple windows: apply to each interval
            [AggregateFeat{T}(T, vnames[c], f, i) for i in 1:nwindows, f in features, c in axes(X,2)] |> vec
        end
        )
    end

    elseif aggrtype == :reducesize begin
        (DataTreatments.reducesize(X, intervals; reducefunc, win, uniform),
        [ReduceFeat{T}(T, vnames[c], reducefunc) for c in axes(X,2)]
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