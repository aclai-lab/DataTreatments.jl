module DataTreatments

using CategoricalArrays
using DataFrames
using Catch22

using Statistics: mean, median, std, cov

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractDataset end
abstract type AbstractDataFeature end

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

export aggregate, reducesize
include("treatment.jl")

export DatasetStructure
export get_vnames, get_datatype, get_dims
export get_valididxs, get_missingidxs, get_nanidxs
export get_hasmissing, get_hasnans
export get_dataset_structure
include("dataset_structure.jl")

export TreatmentGroup
export get_idxs, get_dims, get_vnames, get_aggrfunc, get_grouped, get_groupby, has_groupby
include("treatment_group.jl")

export DiscreteFeat, ContinuousFeat, AggregateFeat, ReduceFeat
export get_id, get_idx, get_vname, get_dims, get_valididxs
export get_missingidxs, get_nanidxs, get_hasmissing, get_hasnans
export get_levels, get_feat, get_nwin, get_reducefunc
include("metadata.jl")

export DiscreteDataset, ContinuousDataset, MultidimDataset
export discrete_encode
export get_data, get_info, get_nrows, get_ncols, get_vnames, get_idxs
export has_groups, get_groups
include("output_dataset.jl")
include("groupby.jl")

export DataTreatment, TreatmentOutput
export groupby
export get_data, get_target, get_ds_struct, get_t_groups, get_float_type
export get_nrows, get_ncols
export get_dataset
export get_treatments_datasets, get_leftover_datasets
export standard, matrix, dataframe
include("datatreatment.jl")

export get_discrete, get_continuous, get_multidim, get_aggregated, get_reduced
export get_tabular, get_multidim
include("methods.jl")

end
