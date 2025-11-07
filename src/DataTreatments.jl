module DataTreatments

using Statistics, StatsBase
using DataFrames
using Catch22

# feature extraction via Catch22
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
       trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
       forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
       stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, centroid_freq,
       transition_variance, periodicity, base_set, catch9, catch22_set, complete_set
include("featureset.jl")

export normalize_dataset
include("normalize.jl")

export movingwindow, wholewindow, splitwindow, adaptivewindow
export @evalwindow
include("slidingwindow.jl")

export applyfeat, aggregate, reducesize
include("treatment.jl")

export DataTreatment
include("constructor.jl")

end
