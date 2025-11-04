function _aggregate(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)
    aggregated = similar(X, length.(intervals)...)

    @inbounds map!(aggregated, CartesianIndices(aggregated)) do cart_idx
        ranges = ntuple(i -> intervals[i][cart_idx[i]], length(intervals))
        reducefunc(@view X[ranges...])
    end
end

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
# apply a feature reduce function to all time-series in a column
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{<:Vector{<:Real}},
    feature_func::Function,
    col_name::Symbol
)::Vector{<:Real}
    @views @inbounds X[!, col_name] = collect(feature_func(col) for col in X_col)
end

# apply a feature function to a specific time interval within each time-series
# - interval: time range to extract features from (e.g., 1:50 for first 50 points)
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{<:Vector{<:Real}},
    feature_func::Function,
    col_name::Symbol,
    interval::UnitRange{Int64}
)::Vector{<:Real}
    @views @inbounds X[!, col_name] = collect(feature_func(col[interval]) for col in X_col)
end

# apply a reduction function across multiple intervals for modal algorithm preparation
function apply_vectorized!(
    X::DataFrame,
    X_col::Vector{<:Vector{<:Real}},
    reducefunc_func::Function,
    col_name::Symbol,
    intervals::Vector{UnitRange{Int64}}
)::Vector{<:Vector{<:Real}}
    X[!, col_name] = [
        [reducefunc_func(@view(ts[interval])) for interval in intervals]
        for ts in X_col
    ]
end

# check dataframe
is_multidim_dataframe(X::DataFrame)::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
using DataFrames

function treatment(
    # X           :: AbstractDataFrame,
    X           :: AbstractArray,
    treat       :: Symbol,
    intervals   :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    features    :: Tuple{Vararg{Base.Callable}}=(mean,),
    reducefunc  :: Base.Callable=mean
)
    # is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
    #     "does not contain multidimensional data."))

    # vnames = propertynames(X)
    # _X = DataFrame()
    # intervals = win(length(X[1,1]))

    # # propositional models
    # isempty(features) && (treat = :none)

    # if treat == :aggregate
    #     for f in features, v in vnames
    #         if length(intervals) == 1
    #             # single window: apply to whole time series
    #             col_name = Symbol("$(f)($(v))")
    #             apply_vectorized!(_X, X[!, v], f, col_name)
    #         else
    #             # multiple windows: apply to each interval
    #             for (i, interval) in enumerate(intervals)
    #                 col_name = Symbol("$(f)($(v))w$(i)")
    #                 apply_vectorized!(_X, X[!, v], f, col_name, interval)
    #             end
    #         end
    #     end

    # # modal models
    # elseif treat == :reducesize
    #     for v in vnames
    #         apply_vectorized!(_X, X[!, v], reducefunc, v, intervals)
    #     end
        
    # elseif treat == :none
    #     _X = X

    # else
    #     error("Unknown treatment type: $treat")
    # end

    # return _X, TreatmentInfo(features, win, treat, reducefunc)
end

using DataFrames
using DataTreatments
reducefunc = mean

X = rand(1000)
wfunc = splitwindow(nwindows=10)
intervals = @evalwindow X wfunc

X = rand(200, 120)
intervals = @evalwindow X splitwindow(nwindows=5) splitwindow(nwindows=3)

Xm = fill(X, 10, 10)
Xd = DataFrame(Xm, :auto)

#####################################################################



    # # propositional models
    # isempty(features) && (treat = :none)

function _aggregate(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)
    aggregated = similar(X, length.(intervals)...)

    @inbounds map!(aggregated, CartesianIndices(aggregated)) do cart_idx
        ranges = ntuple(i -> intervals[i][cart_idx[i]], length(intervals))
        reducefunc(@view X[ranges...])
    end
end
# 28.552 μs (172 allocations: 5.95 KiB)


    #     for f in features, v in vnames
    #         if length(intervals) == 1
    #             # single window: apply to whole time series
    #             col_name = Symbol("$(f)($(v))")
    #             apply_vectorized!(_X, X[!, v], f, col_name)
    #         else
    #             # multiple windows: apply to each interval
    #             for (i, interval) in enumerate(intervals)
    #                 col_name = Symbol("$(f)($(v))w$(i)")
    #                 apply_vectorized!(_X, X[!, v], f, col_name, interval)
    #             end
    #         end
    #     end

    # # modal models
    # elseif treat == :reducesize
    #     for v in vnames
    #         apply_vectorized!(_X, X[!, v], reducefunc, v, intervals)
    #     end
        
    # elseif treat == :none
    #     _X = X

    # else
    #     error("Unknown treatment type: $treat")
    # end