using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays
using Statistics

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    V3 = [3.2, 4.2, 5.2, missing, 2.4],
    V4 = [4.1, NaN, NaN, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
    ts3 = [[1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2], NaN, NaN, missing, [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]],
    ts4 = [[6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8], missing, [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8], [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8], [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

# ---------------------------------------------------------------------------- #
#                              DiscreteFeat                                    #
# ---------------------------------------------------------------------------- #
@testset "DiscreteFeat" begin
    @testset "Construction" begin
        levels_vec = categorical(["blue", "green", "red"])
        feat = DT.DiscreteFeat{String}(
            [1], "str_col", levels_vec, [2, 3, 4, 5], [1]
        )
        @test feat isa DT.DiscreteFeat{String}
        @test feat isa DT.AbstractDataFeature
    end

    @testset "Construction with CategoricalValue type" begin
        levels_vec = categorical(["large", "medium", "small"])
        feat = DT.DiscreteFeat{String}(
            [3], "cat_col", levels_vec, [1, 2, 4, 5], [3]
        )
        @test feat isa DT.DiscreteFeat{String}
    end

    @testset "Getters" begin
        levels_vec = categorical(["blue", "green", "red"])
        feat = DT.DiscreteFeat{String}(
            [1], "str_col", levels_vec, [2, 3, 4, 5], [1]
        )

        @test DT.get_id(feat) == [1]
        @test DT.get_idx(feat) == 1
        @test DT.get_vname(feat) == "str_col"
        @test DT.get_valididxs(feat) == [2, 3, 4, 5]
        @test DT.get_missingidxs(feat) == [1]
        @test DT.get_levels(feat) === levels_vec
    end

    @testset "Getters with nested id" begin
        levels_vec = categorical(["blue", "green", "red"])
        feat = DT.DiscreteFeat{String}(
            [1, 3], "str_col", levels_vec, [2, 3, 4, 5], [1]
        )
        @test DT.get_id(feat) == [1, 3]
        @test DT.get_idx(feat) == 3
    end

    @testset "Empty valid/missing indices" begin
        levels_vec = categorical(["a", "b", "c"])
        feat_no_missing = DT.DiscreteFeat{String}(
            [1], "col", levels_vec, [1, 2, 3], Int[]
        )
        @test DT.get_valididxs(feat_no_missing) == [1, 2, 3]
        @test DT.get_missingidxs(feat_no_missing) == Int[]

        feat_all_missing = DT.DiscreteFeat{String}(
            [1], "col", levels_vec, Int[], [1, 2, 3]
        )
        @test DT.get_valididxs(feat_all_missing) == Int[]
        @test DT.get_missingidxs(feat_all_missing) == [1, 2, 3]
    end

    @testset "Base.show" begin
        levels_vec = categorical(["blue", "green", "red"])
        feat = DT.DiscreteFeat{String}(
            [1], "str_col", levels_vec, [2, 3, 4, 5], [1]
        )
        str = sprint(show, feat)
        @test occursin("DiscreteFeat{String}", str)
        @test occursin("str_col", str)
        @test occursin("3 levels", str)
        @test occursin("4 valid", str)
        @test occursin("1 missing", str)
    end
end

# ---------------------------------------------------------------------------- #
#                            ContinuousFeat                                    #
# ---------------------------------------------------------------------------- #
@testset "ContinuousFeat" begin
    @testset "Construction" begin
        feat = DT.ContinuousFeat{Float64}(
            [6], "V1", [3, 4, 5], [2], [1]
        )
        @test feat isa DT.ContinuousFeat{Float64}
        @test feat isa DT.AbstractDataFeature
    end

    @testset "Construction with Int type" begin
        feat = DT.ContinuousFeat{Int}(
            [5], "int_col", [1, 2, 3, 4, 5], Int[], Int[]
        )
        @test feat isa DT.ContinuousFeat{Int}
    end

    @testset "Construction with UInt32 type" begin
        feat = DT.ContinuousFeat{UInt32}(
            [4], "uint_col", [1, 2, 3, 4, 5], Int[], Int[]
        )
        @test feat isa DT.ContinuousFeat{UInt32}
    end

    @testset "Getters" begin
        feat = DT.ContinuousFeat{Float64}(
            [6], "V1", [3, 4, 5], [2], [1]
        )

        @test DT.get_id(feat) == [6]
        @test DT.get_idx(feat) == 6
        @test DT.get_vname(feat) == "V1"
        @test DT.get_valididxs(feat) == [3, 4, 5]
        @test DT.get_missingidxs(feat) == [2]
        @test DT.get_nanidxs(feat) == [1]
    end

    @testset "Getters with nested id" begin
        feat = DT.ContinuousFeat{Float64}(
            [2, 7], "V2", [1, 3, 4], [2], [5]
        )
        @test DT.get_id(feat) == [2, 7]
        @test DT.get_idx(feat) == 7
    end

    @testset "No missing, no NaN" begin
        feat = DT.ContinuousFeat{Float64}(
            [10], "V5", [1, 2, 3, 4, 5], Int[], Int[]
        )
        @test DT.get_valididxs(feat) == [1, 2, 3, 4, 5]
        @test DT.get_missingidxs(feat) == Int[]
        @test DT.get_nanidxs(feat) == Int[]
    end

    @testset "Multiple NaN indices" begin
        # V4 = [4.1, NaN, NaN, 7.1, 5.5]
        feat = DT.ContinuousFeat{Float64}(
            [9], "V4", [1, 4, 5], Int[], [2, 3]
        )
        @test DT.get_nanidxs(feat) == [2, 3]
        @test length(DT.get_valididxs(feat)) == 3
    end

    @testset "Base.show" begin
        feat = DT.ContinuousFeat{Float64}(
            [6], "V1", [3, 4, 5], [2], [1]
        )
        str = sprint(show, feat)
        @test occursin("ContinuousFeat{Float64}", str)
        @test occursin("V1", str)
        @test occursin("3 valid", str)
        @test occursin("1 missing", str)
        @test occursin("1 NaN", str)
    end

    @testset "Base.show no issues" begin
        feat = DT.ContinuousFeat{Int}(
            [5], "int_col", [1, 2, 3, 4, 5], Int[], Int[]
        )
        str = sprint(show, feat)
        @test occursin("ContinuousFeat{Int", str)
        @test occursin("5 valid", str)
        @test occursin("0 missing", str)
        @test occursin("0 NaN", str)
    end
end

# ---------------------------------------------------------------------------- #
#                             AggregateFeat                                    #
# ---------------------------------------------------------------------------- #
@testset "AggregateFeat" begin
    @testset "Construction" begin
        feat = DT.AggregateFeat{Float64}(
            [11], "ts1", 1, maximum, 3,
            [2, 4, 5], [3], [1], Int[], Int[]
        )
        @test feat isa DT.AggregateFeat{Float64}
        @test feat isa DT.AbstractDataFeature
    end

    @testset "Construction with mean" begin
        feat = DT.AggregateFeat{Float64}(
            [12], "ts2", 1, Statistics.mean, 4,
            [1, 2, 3, 4], Int[], [5], Int[], Int[]
        )
        @test feat isa DT.AggregateFeat{Float64}
    end

    @testset "Getters - basic" begin
        feat = DT.AggregateFeat{Float64}(
            [11], "ts1", 1, maximum, 3,
            [2, 4, 5], [3], [1], Int[], Int[]
        )

        @test DT.get_id(feat) == [11]
        @test DT.get_idx(feat) == 11
        @test DT.get_vname(feat) == "ts1"
        @test DT.get_dims(feat) == 1
        @test DT.get_feat(feat) === maximum
        @test DT.get_nwin(feat) == 3
        @test DT.get_valididxs(feat) == [2, 4, 5]
        @test DT.get_missingidxs(feat) == [3]
        @test DT.get_nanidxs(feat) == [1]
        @test DT.get_hasmissing(feat) == Int[]
        @test DT.get_hasnans(feat) == Int[]
    end

    @testset "Getters - with hasmissing and hasnans" begin
        # ts3 has elements with internal NaN and missing
        feat = DT.AggregateFeat{Float64}(
            [13], "ts3", 1, minimum, 2,
            [1, 5], [4], [2, 3], [5], [1, 5]
        )

        @test DT.get_hasmissing(feat) == [5]
        @test DT.get_hasnans(feat) == [1, 5]
    end

    @testset "Getters - 2D (images)" begin
        feat = DT.AggregateFeat{Float64}(
            [15], "img1", 2, maximum, 4,
            [1, 2, 3, 4, 5], Int[], Int[], Int[], Int[]
        )

        @test DT.get_dims(feat) == 2
        @test DT.get_feat(feat) === maximum
        @test DT.get_nwin(feat) == 4
    end

    @testset "Getters with nested id" begin
        feat = DT.AggregateFeat{Float64}(
            [3, 11, 2], "ts1", 1, maximum, 3,
            [2, 4, 5], [3], [1], Int[], Int[]
        )
        @test DT.get_id(feat) == [3, 11, 2]
        @test DT.get_idx(feat) == 2
    end

    @testset "Base.show" begin
        feat = DT.AggregateFeat{Float64}(
            [11], "ts1", 1, maximum, 3,
            [2, 4, 5], [3], [1], Int[], Int[]
        )
        str = sprint(show, feat)
        @test occursin("AggregateFeat{Float64}", str)
        @test occursin("ts1", str)
        @test occursin("dims=1", str)
        @test occursin("feat=maximum", str)
        @test occursin("nwin=3", str)
        @test occursin("3 valid", str)
        @test occursin("1 missing", str)
        @test occursin("1 NaN", str)
    end

    @testset "Base.show 2D" begin
        feat = DT.AggregateFeat{Float64}(
            [15], "img1", 2, minimum, 6,
            [1, 2, 3, 4, 5], Int[], Int[], Int[], Int[]
        )
        str = sprint(show, feat)
        @test occursin("dims=2", str)
        @test occursin("feat=minimum", str)
        @test occursin("nwin=6", str)
        @test occursin("5 valid", str)
    end
end

# ---------------------------------------------------------------------------- #
#                              ReduceFeat                                      #
# ---------------------------------------------------------------------------- #
@testset "ReduceFeat" begin
    reduce_fn = x -> x[1:2:end]  # simple downsampling

    @testset "Construction" begin
        feat = DT.ReduceFeat{Float64}(
            [11], "ts1", 1, reduce_fn,
            [2, 4, 5], [3], [1], Int[], Int[]
        )
        @test feat isa DT.ReduceFeat{Float64}
        @test feat isa DT.AbstractDataFeature
    end

    @testset "Construction 2D" begin
        feat = DT.ReduceFeat{Float64}(
            [15], "img1", 2, reduce_fn,
            [1, 2, 3, 4, 5], Int[], Int[], Int[], Int[]
        )
        @test feat isa DT.ReduceFeat{Float64}
        @test DT.get_dims(feat) == 2
    end

    @testset "Getters - basic" begin
        feat = DT.ReduceFeat{Float64}(
            [11], "ts1", 1, reduce_fn,
            [2, 4, 5], [3], [1], Int[], Int[]
        )

        @test DT.get_id(feat) == [11]
        @test DT.get_idx(feat) == 11
        @test DT.get_vname(feat) == "ts1"
        @test DT.get_dims(feat) == 1
        @test DT.get_reducefunc(feat) === reduce_fn
        @test DT.get_valididxs(feat) == [2, 4, 5]
        @test DT.get_missingidxs(feat) == [3]
        @test DT.get_nanidxs(feat) == [1]
        @test DT.get_hasmissing(feat) == Int[]
        @test DT.get_hasnans(feat) == Int[]
    end

    @testset "Getters - with hasmissing and hasnans" begin
        # ts4 has elements with internal NaN and missing
        feat = DT.ReduceFeat{Float64}(
            [14], "ts4", 1, reduce_fn,
            [1, 3, 4, 5], [2], Int[], [1, 3, 4, 5], [3, 4, 5]
        )

        @test DT.get_hasmissing(feat) == [1, 3, 4, 5]
        @test DT.get_hasnans(feat) == [3, 4, 5]
    end

    @testset "Getters with nested id" begin
        feat = DT.ReduceFeat{Float64}(
            [5, 14], "ts4", 1, reduce_fn,
            [1, 3, 4, 5], [2], Int[], Int[], Int[]
        )
        @test DT.get_id(feat) == [5, 14]
        @test DT.get_idx(feat) == 14
    end

    @testset "Base.show" begin
        feat = DT.ReduceFeat{Float64}(
            [11], "ts1", 1, reduce_fn,
            [2, 4, 5], [3], [1], Int[], Int[]
        )
        str = sprint(show, feat)
        @test occursin("ReduceFeat{Float64}", str)
        @test occursin("ts1", str)
        @test occursin("dims=1", str)
        @test occursin("3 valid", str)
        @test occursin("1 missing", str)
        @test occursin("1 NaN", str)
    end

    @testset "Base.show 2D" begin
        feat = DT.ReduceFeat{Float64}(
            [16], "img2", 2, reduce_fn,
            [2, 3, 4, 5], Int[], [1], Int[], Int[]
        )
        str = sprint(show, feat)
        @test occursin("dims=2", str)
        @test occursin("4 valid", str)
        @test occursin("0 missing", str)
        @test occursin("1 NaN", str)
    end
end

# ---------------------------------------------------------------------------- #
#                          Cross-type checks                                   #
# ---------------------------------------------------------------------------- #
@testset "AbstractDataFeature interface" begin
    levels_vec = categorical(["blue", "green", "red"])
    reduce_fn = x -> x[1:2:end]

    discrete = DT.DiscreteFeat{String}([1], "str_col", levels_vec, [2, 3, 4, 5], [1])
    continuous = DT.ContinuousFeat{Float64}([6], "V1", [3, 4, 5], [2], [1])
    aggregate = DT.AggregateFeat{Float64}([11], "ts1", 1, maximum, 3, [2, 4, 5], [3], [1], Int[], Int[])
    reduce = DT.ReduceFeat{Float64}([11], "ts1", 1, reduce_fn, [2, 4, 5], [3], [1], Int[], Int[])

    @testset "All subtypes share common getters" begin
        for feat in [discrete, continuous, aggregate, reduce]
            @test DT.get_id(feat) isa Vector
            @test DT.get_idx(feat) isa Integer
            @test DT.get_vname(feat) isa String
            @test DT.get_valididxs(feat) isa Vector{Int}
            @test DT.get_missingidxs(feat) isa Vector{Int}
        end
    end

    @testset "get_nanidxs only for numeric types" begin
        @test DT.get_nanidxs(continuous) isa Vector{Int}
        @test DT.get_nanidxs(aggregate) isa Vector{Int}
        @test DT.get_nanidxs(reduce) isa Vector{Int}
        @test_throws MethodError DT.get_nanidxs(discrete)
    end

    @testset "get_dims only for multidimensional types" begin
        @test DT.get_dims(aggregate) isa Int
        @test DT.get_dims(reduce) isa Int
        @test_throws MethodError DT.get_dims(discrete)
        @test_throws MethodError DT.get_dims(continuous)
    end

    @testset "get_hasmissing/get_hasnans only for multidimensional types" begin
        @test DT.get_hasmissing(aggregate) isa Vector{Int}
        @test DT.get_hasmissing(reduce) isa Vector{Int}
        @test DT.get_hasnans(aggregate) isa Vector{Int}
        @test DT.get_hasnans(reduce) isa Vector{Int}
        @test_throws MethodError DT.get_hasmissing(discrete)
        @test_throws MethodError DT.get_hasmissing(continuous)
        @test_throws MethodError DT.get_hasnans(discrete)
        @test_throws MethodError DT.get_hasnans(continuous)
    end

    @testset "get_levels only for DiscreteFeat" begin
        @test DT.get_levels(discrete) isa CategoricalArrays.CategoricalVector
        @test_throws MethodError DT.get_levels(continuous)
        @test_throws MethodError DT.get_levels(aggregate)
        @test_throws MethodError DT.get_levels(reduce)
    end

    @testset "get_feat/get_nwin only for AggregateFeat" begin
        @test DT.get_feat(aggregate) === maximum
        @test DT.get_nwin(aggregate) == 3
        @test_throws MethodError DT.get_feat(discrete)
        @test_throws MethodError DT.get_feat(continuous)
        @test_throws MethodError DT.get_feat(reduce)
        @test_throws MethodError DT.get_nwin(discrete)
        @test_throws MethodError DT.get_nwin(continuous)
        @test_throws MethodError DT.get_nwin(reduce)
    end

    @testset "get_reducefunc only for ReduceFeat" begin
        @test DT.get_reducefunc(reduce) === reduce_fn
        @test_throws MethodError DT.get_reducefunc(discrete)
        @test_throws MethodError DT.get_reducefunc(continuous)
        @test_throws MethodError DT.get_reducefunc(aggregate)
    end
end