# this module transforms multidimensional datasets
# into formats suitable for different model algorithm families:

# 1. propositional algorithms (DecisionTree, XGBoost):
#    - applies windowing to divide time series into segments
#    - extracts scalar features (max, min, mean, etc.) from each window
#    - returns a standard tabular DataFrame

# 2. modal algorithms (ModalDecisionTree):
#    - creates windowed time series preserving temporal structure
#    - applies reduction functions to manage dimensionality

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# base type for metadata containers
abstract type AbstractTreatmentInfo end

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
# check dataframe
is_multidim_dataframe(X::DataFrame)::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

# ---------------------------------------------------------------------------- #
#                          multidimensional dataset                            #
# ---------------------------------------------------------------------------- #
# metadata container for dataset preprocessing operations.
struct TreatmentInfo <: AbstractTreatmentInfo
    features    :: Tuple{Vararg{Base.Callable}}
    winparams   :: WinFunction
    treatment   :: Symbol
    modalreduce :: Base.Callable
end

# simplified metadata for aggregation-only preprocessing.
struct AggregationInfo <: AbstractTreatmentInfo
    features    :: Tuple{Vararg{Base.Callable}}
    winparams   :: WinFunction
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
get_treatment(t::TreatmentInfo) = t.treatment

# ---------------------------------------------------------------------------- #
#                                   base show                                  #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, info::TreatmentInfo)
    println(io, "TreatmentInfo:")
    for field in fieldnames(TreatmentInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

function Base.show(io::IO, info::AggregationInfo)
    println(io, "AggregationInfo:")
    for field in fieldnames(AggregationInfo)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 15), value)
    end
end

# ---------------------------------------------------------------------------- #
#                        modal -> propositional adapter                        #
# ---------------------------------------------------------------------------- #
# convert treatment information (features and winparams) to aggregation information.
treat2aggr(t::TreatmentInfo)::AggregationInfo = 
    AggregationInfo(t.features, t.winparams)

# ---------------------------------------------------------------------------- #
#                                 constructors                                 #
# ---------------------------------------------------------------------------- #
function treatment end

"""
    treatment(X::AbstractDataFrame; treat::Symbol, win::WinFunction, 
             features::Tuple, modalreduce::Base.Callable) -> (DataFrame, TreatmentInfo)

Transform multidimensional dataset based on specified treatment strategy.

# Arguments
- `X::AbstractDataFrame`: Input dataset with time series in each cell
- `treat::Symbol`: Treatment type - :aggregate, :reducesize or :none
- `win::WinFunction`: Windowing strategy (default: `AdaptiveWindow(nwindows=3, relative_overlap=0.1)`)
- `features::Tuple`: Feature extraction functions (default: `(maximum, minimum)`)
- `modalreduce::Base.Callable`: Reduction function for modal treatments (default: `mean`)

# Treatment Types

## `:aggregate` (for Propositional Algorithms)
Extracts scalar features from time series windows:
- Single window: Applies features to entire time series
- Multiple windows: Creates feature columns per window (e.g., "max(col1)w1")

## `:reducesize` (for Modal Algorithms)
Preserves temporal structure while reducing dimensionality:
- Applies reduction function to each window
- Maintains Vector{Float64} format for modal logic compatibility

## `:none` (for particular cases)
Returns the dataset
"""
function treatment(
    X           :: AbstractDataFrame,
    treat       :: Symbol;
    win         :: WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features    :: Tuple{Vararg{Base.Callable}}=(maximum, minimum),
    modalreduce :: Base.Callable=mean
)
    is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
        "does not contain multidimensional data."))

    vnames = propertynames(X)
    _X = DataFrame()
    intervals = win(length(X[1,1]))

    # propositional models
    isempty(features) && (treat = :none)

    if treat == :aggregate
        for f in features, v in vnames
            if length(intervals) == 1
                # single window: apply to whole time series
                col_name = Symbol("$(f)($(v))")
                apply_vectorized!(_X, X[!, v], f, col_name)
            else
                # multiple windows: apply to each interval
                for (i, interval) in enumerate(intervals)
                    col_name = Symbol("$(f)($(v))w$(i)")
                    apply_vectorized!(_X, X[!, v], f, col_name, interval)
                end
            end
        end

    # modal models
    elseif treat == :reducesize
        for v in vnames
            apply_vectorized!(_X, X[!, v], modalreduce, v, intervals)
        end
        
    elseif treat == :none
        _X = X

    else
        error("Unknown treatment type: $treat")
    end

    return _X, TreatmentInfo(features, win, treat, modalreduce)
end