using Test
using DataTreatments
const DT = DataTreatments

using CategoricalArrays
using Statistics

@testset "Metadata" begin

    @testset "DiscreteFeat" begin
        levels = categorical(["red", "blue", "green"])
        valid = [1, 2, 3]
        miss = [4, 5]

        df = DiscreteFeat{String}([1], "color", levels, valid, miss)

        @testset "Construction" begin
            @test df isa DiscreteFeat{String}
            @test df isa DT.AbstractDataFeature
        end

        @testset "get_id" begin
            @test get_id(df) == [1]
        end

        @testset "get_vname" begin
            @test get_vname(df) == "color"
        end

        @testset "get_valididxs" begin
            @test get_valididxs(df) == [1, 2, 3]
        end

        @testset "get_missingidxs" begin
            @test get_missingidxs(df) == [4, 5]
        end

        @testset "get_levels" begin
            @test get_levels(df) === levels
            @test Set(levels) == Set(["red", "blue", "green"])
        end

        @testset "type parameter" begin
            df_int = DiscreteFeat{Int}([2], "grade", categorical([1, 2, 3]), [1, 2, 3], Int[])
            @test df_int isa DiscreteFeat{Int}
        end

        @testset "empty missing" begin
            df_clean = DiscreteFeat{String}([1], "x", categorical(["a"]), [1], Int[])
            @test isempty(get_missingidxs(df_clean))
        end
    end

    @testset "ContinuousFeat" begin
        valid = [1, 2, 4, 5]
        miss = [3]
        nans = [6]

        cf = ContinuousFeat{Float64}([2], "temperature", valid, miss, nans)

        @testset "Construction" begin
            @test cf isa ContinuousFeat{Float64}
            @test cf isa DT.AbstractDataFeature
        end

        @testset "get_id" begin
            @test get_id(cf) == [2]
        end

        @testset "get_vname" begin
            @test get_vname(cf) == "temperature"
        end

        @testset "get_valididxs" begin
            @test get_valididxs(cf) == [1, 2, 4, 5]
        end

        @testset "get_missingidxs" begin
            @test get_missingidxs(cf) == [3]
        end

        @testset "get_nanidxs" begin
            @test get_nanidxs(cf) == [6]
        end

        @testset "type parameter Int" begin
            cf_int = ContinuousFeat{Int}([3], "count", [1, 2], Int[], Int[])
            @test cf_int isa ContinuousFeat{Int}
        end

        @testset "all clean" begin
            cf_clean = ContinuousFeat{Float64}([1], "x", [1, 2, 3], Int[], Int[])
            @test isempty(get_missingidxs(cf_clean))
            @test isempty(get_nanidxs(cf_clean))
        end
    end

    @testset "AggregateFeat" begin
        valid = [1, 2]
        miss = [3]
        nans = [4]
        hmiss = [5]
        hnans = [2]

        af = AggregateFeat{Float64}(
            [3], "audio_signal", 1, maximum, 4,
            valid, miss, nans, hmiss, hnans
        )

        @testset "Construction" begin
            @test af isa AggregateFeat{Float64}
            @test af isa DT.AbstractDataFeature
        end

        @testset "get_id" begin
            @test get_id(af) == [3]
        end

        @testset "get_vname" begin
            @test get_vname(af) == "audio_signal"
        end

        @testset "get_dims" begin
            @test get_dims(af) == 1
        end

        @testset "get_feat" begin
            @test get_feat(af) === maximum
        end

        @testset "get_nwin" begin
            @test get_nwin(af) == 4
        end

        @testset "get_valididxs" begin
            @test get_valididxs(af) == [1, 2]
        end

        @testset "get_missingidxs" begin
            @test get_missingidxs(af) == [3]
        end

        @testset "get_nanidxs" begin
            @test get_nanidxs(af) == [4]
        end

        @testset "get_hasmissing" begin
            @test get_hasmissing(af) == [5]
        end

        @testset "get_hasnans" begin
            @test get_hasnans(af) == [2]
        end

        @testset "different feature functions" begin
            af_mean = AggregateFeat{Float64}(
                [1], "ts", 1, mean, 2,
                [1, 2], Int[], Int[], Int[], Int[]
            )
            @test get_feat(af_mean) === mean
            @test get_nwin(af_mean) == 2
            @test get_dims(af_mean) == 1
        end

        @testset "2D source" begin
            af_2d = AggregateFeat{Float64}(
                [5], "spectrogram", 2, mean, 1,
                [1, 2, 3], Int[], Int[], Int[], Int[]
            )
            @test get_dims(af_2d) == 2
            @test get_vname(af_2d) == "spectrogram"
        end

        @testset "all clean" begin
            af_clean = AggregateFeat{Float64}(
                [1], "x", 1, maximum, 1,
                [1, 2, 3], Int[], Int[], Int[], Int[]
            )
            @test isempty(get_missingidxs(af_clean))
            @test isempty(get_nanidxs(af_clean))
            @test isempty(get_hasmissing(af_clean))
            @test isempty(get_hasnans(af_clean))
        end
    end

    @testset "ReduceFeat" begin
        my_downsample = x -> x[1:2:end]
        valid = [1, 2, 3]
        miss = [4]
        nans = Int[]
        hmiss = Int[]
        hnans = [3]

        rf = ReduceFeat{Float64}(
            [4], "spectrogram", 2, my_downsample,
            valid, miss, nans, hmiss, hnans
        )

        @testset "Construction" begin
            @test rf isa ReduceFeat{Float64}
            @test rf isa DT.AbstractDataFeature
        end

        @testset "get_id" begin
            @test get_id(rf) == [4]
        end

        @testset "get_vname" begin
            @test get_vname(rf) == "spectrogram"
        end

        @testset "get_dims" begin
            @test get_dims(rf) == 2
        end

        @testset "get_reducefunc" begin
            @test get_reducefunc(rf) === my_downsample
        end

        @testset "get_valididxs" begin
            @test get_valididxs(rf) == [1, 2, 3]
        end

        @testset "get_missingidxs" begin
            @test get_missingidxs(rf) == [4]
        end

        @testset "get_nanidxs" begin
            @test get_nanidxs(rf) == Int[]
        end

        @testset "get_hasmissing" begin
            @test get_hasmissing(rf) == Int[]
        end

        @testset "get_hasnans" begin
            @test get_hasnans(rf) == [3]
        end

        @testset "1D source" begin
            rf_1d = ReduceFeat{Float64}(
                [1], "audio", 1, identity,
                [1, 2], Int[], Int[], Int[], Int[]
            )
            @test get_dims(rf_1d) == 1
        end

        @testset "all clean" begin
            rf_clean = ReduceFeat{Float64}(
                [1], "x", 1, identity,
                [1, 2, 3], Int[], Int[], Int[], Int[]
            )
            @test isempty(get_missingidxs(rf_clean))
            @test isempty(get_nanidxs(rf_clean))
            @test isempty(get_hasmissing(rf_clean))
            @test isempty(get_hasnans(rf_clean))
        end
    end

    @testset "Getter dispatch" begin
        df = DiscreteFeat{String}([1], "a", categorical(["x"]), [1], Int[])
        cf = ContinuousFeat{Float64}([2], "b", [1], Int[], Int[])
        af = AggregateFeat{Float64}([3], "c", 1, maximum, 1, [1], Int[], Int[], Int[], Int[])
        rf = ReduceFeat{Float64}([4], "d", 1, identity, [1], Int[], Int[], Int[], Int[])

        @testset "Common getters work on all types" begin
            for f in [df, cf, af, rf]
                @test get_id(f) isa Vector
                @test get_vname(f) isa String
                @test get_valididxs(f) isa Vector{Int}
                @test get_missingidxs(f) isa Vector{Int}
            end
        end

        @testset "get_dims dispatches on Aggregate and Reduce only" begin
            @test get_dims(af) isa Int
            @test get_dims(rf) isa Int
            @test_throws MethodError get_dims(df)
            @test_throws MethodError get_dims(cf)
        end

        @testset "get_nanidxs dispatches on Continuous, Aggregate, Reduce only" begin
            @test get_nanidxs(cf) isa Vector{Int}
            @test get_nanidxs(af) isa Vector{Int}
            @test get_nanidxs(rf) isa Vector{Int}
            @test_throws MethodError get_nanidxs(df)
        end

        @testset "get_hasmissing dispatches on Aggregate and Reduce only" begin
            @test get_hasmissing(af) isa Vector{Int}
            @test get_hasmissing(rf) isa Vector{Int}
            @test_throws MethodError get_hasmissing(df)
            @test_throws MethodError get_hasmissing(cf)
        end

        @testset "get_hasnans dispatches on Aggregate and Reduce only" begin
            @test get_hasnans(af) isa Vector{Int}
            @test get_hasnans(rf) isa Vector{Int}
            @test_throws MethodError get_hasnans(df)
            @test_throws MethodError get_hasnans(cf)
        end

        @testset "get_levels dispatches on DiscreteFeat only" begin
            @test get_levels(df) isa CategoricalVector
            @test_throws MethodError get_levels(cf)
            @test_throws MethodError get_levels(af)
            @test_throws MethodError get_levels(rf)
        end

        @testset "get_feat dispatches on AggregateFeat only" begin
            @test get_feat(af) === maximum
            @test_throws MethodError get_feat(df)
            @test_throws MethodError get_feat(cf)
            @test_throws MethodError get_feat(rf)
        end

        @testset "get_nwin dispatches on AggregateFeat only" begin
            @test get_nwin(af) == 1
            @test_throws MethodError get_nwin(df)
            @test_throws MethodError get_nwin(cf)
            @test_throws MethodError get_nwin(rf)
        end

        @testset "get_reducefunc dispatches on ReduceFeat only" begin
            @test get_reducefunc(rf) === identity
            @test_throws MethodError get_reducefunc(df)
            @test_throws MethodError get_reducefunc(cf)
            @test_throws MethodError get_reducefunc(af)
        end
    end

    @testset "Subtyping" begin
        @test DiscreteFeat <: DT.AbstractDataFeature
        @test ContinuousFeat <: DT.AbstractDataFeature
        @test AggregateFeat <: DT.AbstractDataFeature
        @test ReduceFeat <: DT.AbstractDataFeature
    end
end