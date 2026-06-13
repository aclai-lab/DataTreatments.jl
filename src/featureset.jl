# References
# The Catch22 features 
# are based on the CAnonical Time-series CHaracteristics from:
# - Repository: https://github.com/DynamicsAndNeuralSystems/catch22
# - Article:    https://doi.org/10.1007/s10618-019-00647-x
# - Author: Carl H. Lubba et al

# ---------------------------------------------------------------------------- #
#                        catch22 pretty named functions                        #
# ---------------------------------------------------------------------------- #
"""
    mode_5(x)

Mode of a z-scored time series binned into 5 equal-width bins.
"""
mode_5(x) = Catch22.DN_HistogramMode_5(x)

"""
    mode_10(x)

Mode of a z-scored time series binned into 10 equal-width bins.
"""
mode_10(x) = Catch22.DN_HistogramMode_10(x)

"""
    embedding_dist(x)

Mean distance from a 2D time-delay embedding to an exponential fit.
"""
embedding_dist(x) = Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(x)

"""
    acf_timescale(x)

First 1/e crossing of the autocorrelation function.
"""
acf_timescale(x) = Catch22.CO_f1ecac(x)

"""
    acf_first_min(x)

First minimum of the autocorrelation function.
"""
acf_first_min(x) = Catch22.CO_FirstMin_ac(x)

"""
    ami2(x)

Automutual information (lag 2, 5 bins, uniform binning).
"""
ami2(x) = Catch22.CO_HistogramAMI_even_2_5(x)

"""
    trev(x)

Time-reversibility statistic from tercile differences at lag 1.
"""
trev(x) = Catch22.CO_trev_1_num(x)

"""
    outlier_timing_pos(x)

Median timing of positive outliers above 0.01 increments.
"""
outlier_timing_pos(x) = Catch22.DN_OutlierInclude_p_001_mdrmd(x)

"""
    outlier_timing_neg(x)

Median timing of negative outliers below 0.01 increments.
"""
outlier_timing_neg(x) = Catch22.DN_OutlierInclude_n_001_mdrmd(x)

"""
    whiten_timescale(x)

Timescale of residuals after mean-1 local simple forecasting.
"""
whiten_timescale(x) = Catch22.FC_LocalSimple_mean1_tauresrat(x)

"""
    forecast_error(x)

Standard error of mean-3 local simple forecasting residuals.
"""
forecast_error(x) = Catch22.FC_LocalSimple_mean3_stderr(x)

"""
    ami_timescale(x)

First minimum of automutual information (Gaussian kernel, 40 lags).
"""
ami_timescale(x) = Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(x)

"""
    high_fluctuation(x)

Proportion of successive differences exceeding 40% of the std
(HRV pNN40 measure).
"""
high_fluctuation(x) = Catch22.MD_hrv_classic_pnn40(x)

"""
    stretch_decreasing(x)

Longest stretch of decreasing values in the binarised time series.
"""
stretch_decreasing(x) = Catch22.SB_BinaryStats_diff_longstretch0(x)

"""
    stretch_high(x)

Mean length of stretches above the mean in the binarised series.
"""
stretch_high(x) = Catch22.SB_BinaryStats_mean_longstretch1(x)

"""
    entropy_pairs(x)

Entropy of two-symbol transition motifs (quantile-based encoding).
"""
entropy_pairs(x) = Catch22.SB_MotifThree_quantile_hh(x)

"""
    rs_range(x)

Proportion of longer time scales in an RS range fluctuation analysis.
"""
rs_range(x) = Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(x)

"""
    dfa(x)

Proportion of longer time scales in a DFA fluctuation analysis.
"""
dfa(x) = Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x)

"""
    low_freq_power(x)

Proportion of power in the lowest 20% of the Welch power spectrum.
"""
low_freq_power(x) = Catch22.SP_Summaries_welch_rect_area_5_1(x)

"""
    centroid_freq(x)

Centroid frequency of the Welch power spectrum.
"""
centroid_freq(x) = Catch22.SP_Summaries_welch_rect_centroid(x)

"""
    transition_variance(x)

Variance of the diagonal of the 3-state transition matrix (lag = ACF).
"""
transition_variance(x) = Catch22.SB_TransitionMatrix_3ac_sumdiagcov(x)

"""
    periodicity(x)

Periodicity measure based on the Wang threshold method (th = 0.01).
"""
periodicity(x) = Catch22.PD_PeriodicityWang_th0_01(x)

# ---------------------------------------------------------------------------- #
#                                  featuresets                                 #
# ---------------------------------------------------------------------------- #
"""
    base_set

A minimal feature set containing only basic statistical measures
for time series analysis.

# Features
- `maximum`: Maximum value in the time series
- `minimum`: Minimum value in the time series
- `mean`   : Arithmetic mean of the time series
- `std`    : Standard deviation of the time series
"""
base_set = (maximum, minimum, mean, std)

"""
    catch9

A curated subset of 9 features combining basic statistics with
Symbolic Catch22 measures.

# Features
- Basic statistics: `maximum`, `minimum`, `mean`, `median`, `std`
- Symbolic Catch22 features:
  - `stretch_high`       : Measures persistence of high values
  - `stretch_decreasing` : Captures decreasing trend patterns
  - `entropy_pairs`      : Quantifies local pattern complexity
  - `transition_variance`: Measures state transition variability

# References
The Catch22 features are based on the CAnonical Time-series
CHaracteristics from:
- Repository: https://github.com/DynamicsAndNeuralSystems/catch22
- Article:    https://doi.org/10.1007/s10618-019-00647-x
- Author: Carl H. Lubba et al
"""
catch9 = (maximum, minimum, mean, median, std,
    stretch_high, stretch_decreasing, entropy_pairs, transition_variance)

"""
    catch22_set

The complete Catch22 feature set. Each feature captures different
aspects of time series dynamics including correlation structure,
distribution properties and temporal patterns.

# Feature Categories
- **Distribution shape**:
  `mode_5`, `mode_10`
- **Extreme event timing**:
  `outlier_timing_pos`, `outlier_timing_neg`
- **Linear autocorrelation**:
  `acf_timescale`, `acf_first_min`, `low_freq_power`, `centroid_freq`
- **Simple forecasting**:
  `forecast_error`
- **Incremental differences**:
  `whiten_timescale`, `high_fluctuation`
- **Symbolic**:
  `stretch_high`, `stretch_decreasing`, `entropy_pairs`,
  `transition_variance`
- **Nonlinear autocorrelation**:
  `ami2`, `trev`
- **Linear autocorrelation structure**:
  `ami_timescale`, `periodicity`
- **Self-affine scaling**:
  `rs_range`, `dfa`
- **Other**:
  `embedding_dist`

# References
The Catch22 features are based on the CAnonical Time-series
CHaracteristics from:
- Repository: https://github.com/DynamicsAndNeuralSystems/catch22
- Article:    https://doi.org/10.1007/s10618-019-00647-x
- Author: Carl H. Lubba et al
"""
catch22_set = (mode_5, mode_10, embedding_dist, acf_timescale, acf_first_min,
    ami2, trev, outlier_timing_pos, outlier_timing_neg, whiten_timescale, 
    forecast_error, ami_timescale, high_fluctuation, stretch_decreasing,
    stretch_high, entropy_pairs, rs_range, dfa, low_freq_power, 
    centroid_freq, transition_variance, periodicity)

"""
    complete_set

The most comprehensive feature set combining basic statistical
measures, covariance analysis, and the full Catch22 suite.

# Features
- **Basic statistics**:
  `maximum`, `minimum`, `mean`, `median`, `std`, `cov`
- **Distribution shape**:
  `mode_5`, `mode_10`
- **Extreme event timing**:
  `outlier_timing_pos`, `outlier_timing_neg`
- **Linear autocorrelation**:
  `acf_timescale`, `acf_first_min`, `low_freq_power`, `centroid_freq`
- **Simple forecasting**:
  `forecast_error`
- **Incremental differences**:
  `whiten_timescale`, `high_fluctuation`
- **Symbolic**:
  `stretch_high`, `stretch_decreasing`, `entropy_pairs`,
  `transition_variance`
- **Nonlinear autocorrelation**:
  `ami2`, `trev`
- **Linear autocorrelation structure**:
  `ami_timescale`, `periodicity`
- **Self-affine scaling**:
  `rs_range`, `dfa`
- **Other**:
  `embedding_dist`
"""
complete_set = (maximum, minimum, mean, median, std,
    cov, mode_5, mode_10, embedding_dist, acf_timescale,
    acf_first_min, ami2, trev, outlier_timing_pos, outlier_timing_neg,
    whiten_timescale, forecast_error, ami_timescale, high_fluctuation,
    stretch_decreasing, stretch_high, entropy_pairs, rs_range, dfa,
    low_freq_power, centroid_freq, transition_variance, periodicity)