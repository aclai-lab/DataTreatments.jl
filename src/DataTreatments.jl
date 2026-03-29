module DataTreatments

using Reexport

using CategoricalArrays
using DataFrames
using Catch22

using Statistics: mean, median, std, cov

using Impute
@reexport using Impute: Interpolate, Impute.LOCF, Impute.NOCB
@reexport using Impute: Impute.Substitute, Impute.SVD

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataset end
abstract type AbstractDataFeature end

const Float = Union{Float32,Float64}

# ---------------------------------------------------------------------------- #
#                                 includes                                     #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
       trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
       forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
       stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq,
       transition_variance, periodicity, base_set, catch9, catch22_set, complete_set
include("featureset.jl")

export movingwindow, wholewindow, splitwindow, adaptivewindow
export @evalwindow
include("windowing.jl")

include("inspecting.jl")

export aggregate, reducesize
include("multidim_treatment.jl")

export TreatmentGroup
include("treatment_group.jl")

const DefaultAggrFunc = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))
const DefaultGrouped = false
const DefaultTreatmentGroup = TreatmentGroup(aggrfunc=DefaultAggrFunc, grouped=DefaultGrouped)

include("output_datasets.jl")
include("treatment.jl")

include("groupby.jl")

export DataTreatment
export load_dataset
export nrows
export get_target, get_levels
export get_discrete, get_continuous
export get_aggregated, get_reduced
export get_tabular, get_multidim
export is_tabular, is_multidim
include("datatreatment.jl")

end
