module DataTreatments

using Reexport

using CategoricalArrays
using DataFrames
using Random
using Catch22

using StatsBase: mad
using Statistics: mean, median, std, cov
using LinearAlgebra: norm

using Impute
@reexport using Impute: Interpolate, Impute.LOCF, Impute.NOCB
@reexport using Impute: Impute.Substitute, Impute.SVD

using Imbalance:
    random_oversample,
    random_walk_oversample,
    rose,
    smote,
    borderline_smote1,
    smoten,
    smotenc,
    random_undersample,
    cluster_undersample,
    enn_undersample,
    tomek_undersample

using Normalization

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataset end
abstract type AbstractDataFeature end
abstract type AbstractBalance end

const Float = Union{Float32,Float64}

# ---------------------------------------------------------------------------- #
#                                 includes                                     #
# ---------------------------------------------------------------------------- #
# feature extraction via Catch22
# export user friendly Catch22 nicknames
export mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min, ami2,
    trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale,
    forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
    stretch_high, entropy_pairs, rs_range, dfa, low_freq_power,
    centroid_freq, transition_variance, periodicity,
    base_set, catch9, catch22_set, complete_set
include("featureset.jl")

export movingwindow, wholewindow, splitwindow, adaptivewindow, @evalwindow
include("windowing.jl")

include("impute.jl")

export RandomOversampler, RandomWalkOversampler, ROSE, SMOTE,
    BorderlineSMOTE1, SMOTEN, SMOTENC, RandomUndersampler,
    ClusterUndersampler, ENNUndersampler, TomekUndersampler
include("imbalance.jl")

export ZScore, MinMax, Center, Sigmoid, UnitEnergy, UnitPower,
    Scale, ScaleMad, ScaleFirst, PNorm1, PNorm, PNormInf,
    MissingSafe, Robust
include("normalization.jl")

include("inspecting.jl")

export aggregate, reducesize
include("multidim_treatment.jl")

export TreatmentGroup
include("treatment_group.jl")

const DefaultAggrFunc =
    aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))
const DefaultGrouped = false
const DefaultTreatmentGroup =
    TreatmentGroup(aggrfunc=DefaultAggrFunc, grouped=DefaultGrouped)

include("output_datasets.jl")
include("treatment.jl")

include("groupby.jl")

export DataTreatment,
    nrows, ncols, get_target, get_treats, get_balance, load_dataset
include("load_dataset.jl")

export get_discrete, get_continuous, get_aggregated, get_reduced,
    get_tabular, get_multidim, is_tabular, is_multidim,
    has_tabular, has_multidim,
    filter_missing
include("datatreatment.jl")

end
