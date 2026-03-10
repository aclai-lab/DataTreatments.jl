using Test
using DataTreatments
const DT = DataTreatments

using Statistics
using Catch22

@testset "FeatureSet" begin

    # A simple deterministic time series for reproducible results
    ts = collect(1.0:100.0)
    # A more varied signal for catch22 features
    ts_varied = Float64[sin(2π * i / 20) + 0.5 * cos(2π * i / 7) for i in 1:200]

    # ------------------------------------------------------------------ #
    #                   Catch22 pretty-named wrappers                    #
    # ------------------------------------------------------------------ #
    @testset "Catch22 wrappers return scalars" begin
        fns = [
            DT.mode_5,
            DT.mode_10,
            DT.embedding_dist,
            DT.acf_timescale,
            DT.acf_first_min,
            DT.ami2,
            DT.trev,
            DT.outlier_timing_pos,
            DT.outlier_timing_neg,
            DT.whiten_timescale,
            DT.forecast_error,
            DT.ami_timescale,
            DT.high_fluctuation,
            DT.stretch_decreasing,
            DT.stretch_high,
            DT.entropy_pairs,
            DT.rs_range,
            DT.dfa,
            DT.low_freq_power,
            DT.centroid_freq,
            DT.transition_variance,
            DT.periodicity,
        ]

        for f in fns
            result = f(ts_varied)
            @test result isa Real
            @test isfinite(result) || result isa Integer
        end
    end

    # ------------------------------------------------------------------ #
    #                     Wrapper consistency                             #
    # ------------------------------------------------------------------ #
    @testset "Wrappers match Catch22 originals" begin
        @test DT.mode_5(ts_varied) == Catch22.DN_HistogramMode_5(ts_varied)
        @test DT.mode_10(ts_varied) == Catch22.DN_HistogramMode_10(ts_varied)
        @test DT.acf_timescale(ts_varied) == Catch22.CO_f1ecac(ts_varied)
        @test DT.acf_first_min(ts_varied) == Catch22.CO_FirstMin_ac(ts_varied)
        @test DT.ami2(ts_varied) == Catch22.CO_HistogramAMI_even_2_5(ts_varied)
        @test DT.trev(ts_varied) == Catch22.CO_trev_1_num(ts_varied)
        @test DT.embedding_dist(ts_varied) == Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff(ts_varied)
        @test DT.outlier_timing_pos(ts_varied) == Catch22.DN_OutlierInclude_p_001_mdrmd(ts_varied)
        @test DT.outlier_timing_neg(ts_varied) == Catch22.DN_OutlierInclude_n_001_mdrmd(ts_varied)
        @test DT.whiten_timescale(ts_varied) == Catch22.FC_LocalSimple_mean1_tauresrat(ts_varied)
        @test DT.forecast_error(ts_varied) == Catch22.FC_LocalSimple_mean3_stderr(ts_varied)
        @test DT.ami_timescale(ts_varied) == Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(ts_varied)
        @test DT.high_fluctuation(ts_varied) == Catch22.MD_hrv_classic_pnn40(ts_varied)
        @test DT.stretch_decreasing(ts_varied) == Catch22.SB_BinaryStats_diff_longstretch0(ts_varied)
        @test DT.stretch_high(ts_varied) == Catch22.SB_BinaryStats_mean_longstretch1(ts_varied)
        @test DT.entropy_pairs(ts_varied) == Catch22.SB_MotifThree_quantile_hh(ts_varied)
        @test DT.rs_range(ts_varied) == Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(ts_varied)
        @test DT.dfa(ts_varied) == Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(ts_varied)
        @test DT.low_freq_power(ts_varied) == Catch22.SP_Summaries_welch_rect_area_5_1(ts_varied)
        @test DT.centroid_freq(ts_varied) == Catch22.SP_Summaries_welch_rect_centroid(ts_varied)
        @test DT.transition_variance(ts_varied) == Catch22.SB_TransitionMatrix_3ac_sumdiagcov(ts_varied)
        @test DT.periodicity(ts_varied) == Catch22.PD_PeriodicityWang_th0_01(ts_varied)
    end

    # ------------------------------------------------------------------ #
    #                        Feature set tuples                          #
    # ------------------------------------------------------------------ #
    @testset "base_set" begin
        bs = DT.base_set

        @test bs isa Tuple
        @test length(bs) == 4
        @test maximum in bs
        @test minimum in bs
        @test mean in bs
        @test std in bs

        # All functions should work on a simple time series
        for f in bs
            @test f(ts) isa Real
        end
    end

    @testset "catch9" begin
        c9 = DT.catch9

        @test c9 isa Tuple
        @test length(c9) == 9

        # Contains basic stats
        @test maximum in c9
        @test minimum in c9
        @test mean in c9
        @test median in c9
        @test std in c9

        # Contains catch22 subset
        @test DT.stretch_high in c9
        @test DT.stretch_decreasing in c9
        @test DT.entropy_pairs in c9
        @test DT.transition_variance in c9

        for f in c9
            @test f(ts_varied) isa Real
        end
    end

    @testset "catch22_set" begin
        c22 = DT.catch22_set

        @test c22 isa Tuple
        @test length(c22) == 22

        # All 22 catch22 wrapper functions should be present
        expected = [
            DT.mode_5, DT.mode_10,
            DT.embedding_dist, DT.acf_timescale,
            DT.acf_first_min, DT.ami2,
            DT.trev, DT.outlier_timing_pos,
            DT.outlier_timing_neg, DT.whiten_timescale,
            DT.forecast_error, DT.ami_timescale,
            DT.high_fluctuation, DT.stretch_decreasing,
            DT.stretch_high, DT.entropy_pairs,
            DT.rs_range, DT.dfa,
            DT.low_freq_power, DT.centroid_freq,
            DT.transition_variance, DT.periodicity,
        ]
        for f in expected
            @test f in c22
        end

        # No duplicates
        @test length(unique(c22)) == 22

        for f in c22
            @test f(ts_varied) isa Real
        end
    end

    @testset "complete_set" begin
        cs = DT.complete_set

        @test cs isa Tuple
        @test length(cs) == 28

        # Contains all of base_set
        for f in DT.base_set
            @test f in cs
        end

        # Contains all of catch22_set
        for f in DT.catch22_set
            @test f in cs
        end

        # Contains extras not in base or catch22
        @test median in cs
        @test cov in cs

        # No duplicates
        @test length(unique(cs)) == 28

        for f in cs
            @test f(ts_varied) isa Real
        end
    end

    # ------------------------------------------------------------------ #
    #                        Set relationships                           #
    # ------------------------------------------------------------------ #
    @testset "Set containment relationships" begin
        # base_set ⊂ complete_set
        for f in DT.base_set
            @test f in DT.complete_set
        end

        # catch9 ⊂ complete_set
        for f in DT.catch9
            @test f in DT.complete_set
        end

        # catch22_set ⊂ complete_set
        for f in DT.catch22_set
            @test f in DT.complete_set
        end

        # catch9 catch22 subset ⊂ catch22_set
        catch22_fns_in_catch9 = [
            DT.stretch_high,
            DT.stretch_decreasing,
            DT.entropy_pairs,
            DT.transition_variance,
        ]
        for f in catch22_fns_in_catch9
            @test f in DT.catch22_set
        end
    end

    @testset "Wrapper docstrings exist" begin
        fns = [
            DT.mode_5, DT.mode_10,
            DT.embedding_dist, DT.acf_timescale,
            DT.acf_first_min, DT.ami2,
            DT.trev, DT.outlier_timing_pos,
            DT.outlier_timing_neg, DT.whiten_timescale,
            DT.forecast_error, DT.ami_timescale,
            DT.high_fluctuation, DT.stretch_decreasing,
            DT.stretch_high, DT.entropy_pairs,
            DT.rs_range, DT.dfa,
            DT.low_freq_power, DT.centroid_freq,
            DT.transition_variance, DT.periodicity,
        ]

        for f in fns
            docstr = string(@doc f)
            @test length(docstr) > 10
        end
    end
end