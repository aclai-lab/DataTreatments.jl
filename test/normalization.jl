using Test
using DataTreatments
const DT = DataTreatments

using Normalization

using Statistics: mean, std
using StatsBase: mad
using LinearAlgebra: norm
using Random

# ---------------------------------------------------------------------------- #
#                                 test data                                    #
# ---------------------------------------------------------------------------- #
function make_scalar_matrix(nrows=5, ncols=3; seed=42)
    Random.seed!(seed)
    Union{Missing,Float64}[rand(Float64, nrows, ncols)...]
    m = Matrix{Union{Missing,Float64}}(rand(Float64, nrows, ncols))
    return m
end

function make_ts_matrix(nrows=5, ncols=3, tslen=10; seed=42)
    Random.seed!(seed)
    data = Matrix{Union{Missing,Vector{Float64}}}(undef, nrows, ncols)
    for i in 1:nrows, j in 1:ncols
        data[i, j] = rand() > 0.1 ? rand(Float64, tslen) : missing
    end
    return data
end

function make_clean_ts_matrix(nrows=5, ncols=3, tslen=10; seed=42)
    Random.seed!(seed)
    [rand(Float64, tslen) for _ in 1:nrows, _ in 1:ncols]
end

# ---------------------------------------------------------------------------- #
#                         scalar normalizations                                #
# ---------------------------------------------------------------------------- #
@testset "scalar normalizations" begin

    @testset "Scale" begin
        X = make_scalar_matrix()
        T = DT.fit(Scale, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test length(Y) > 0
        @test all(isfinite, Y)
    end

    @testset "ScaleMad" begin
        X = make_scalar_matrix()
        T = DT.fit(ScaleMad, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(isfinite, Y)
    end

    @testset "ScaleFirst" begin
        X = make_scalar_matrix()
        T = DT.fit(ScaleFirst, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(isfinite, Y)
    end

    @testset "PNorm1" begin
        X = make_scalar_matrix()
        T = DT.fit(PNorm1, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(isfinite, Y)
    end

    @testset "PNorm" begin
        X = make_scalar_matrix()
        T = DT.fit(PNorm, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(isfinite, Y)
    end

    @testset "PNormInf" begin
        X = make_scalar_matrix()
        T = DT.fit(PNormInf, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(isfinite, Y)
    end

    @testset "ZScore (from Normalization.jl)" begin
        X = make_scalar_matrix()
        T = DT.fit(ZScore, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(isfinite, Y)
    end

    @testset "MinMax (from Normalization.jl)" begin
        X = make_scalar_matrix()
        T = DT.fit(MinMax, X)
        Y = DT.normalize(X, T)
        @test !isnothing(T)
        @test all(x -> 0.0 ≤ x ≤ 1.0, Y)
    end
end

# ---------------------------------------------------------------------------- #
#                    multidim fit! / fit / normalize                           #
# ---------------------------------------------------------------------------- #
@testset "multidim normalizations (NormalizationExt)" begin
    @testset "DT.fit! ZScore on ts matrix" begin
        X = make_ts_matrix()
        T = ZScore{Float64}()
        DT.fit!(T, X)
        @test !isnothing(T)
    end

    @testset "DT.fit ZScore{Float64} on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(ZScore{Float64}, X)
        @test !isnothing(T)
    end

    @testset "DT.fit ZScore (unparameterised) on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(ZScore, X)
        @test !isnothing(T)
    end

    @testset "DT.normalize preserves missing entries" begin
        X = make_ts_matrix()
        T = DT.fit(ZScore, X)
        Y = DT.normalize(X, T)
        for i in eachindex(X)
            @test ismissing(X[i]) == ismissing(Y[i])
        end
    end

    @testset "DT.normalize does not mutate original" begin
        X = make_clean_ts_matrix()
        X_copy = deepcopy(X)
        T = DT.fit(ZScore, X)
        _ = DT.normalize(X, T)
        @test all(X[i] == X_copy[i] for i in eachindex(X))
    end

    @testset "DT.normalize Scale on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(Scale, X)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.normalize ScaleMad on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(ScaleMad, X)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.normalize ScaleFirst on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(ScaleFirst, X)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.normalize PNorm1 on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(PNorm1, X)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.normalize PNorm on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(PNorm, X)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.normalize PNormInf on ts matrix" begin
        X = make_ts_matrix()
        T = DT.fit(PNormInf, X)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.fit! type mismatch throws TypeError" begin
        X = make_ts_matrix()  # inner arrays are Float64
        T = ZScore{Float32}()
        @test_throws TypeError DT.fit!(T, X)
    end

    @testset "DT.normalize with dims on ts matrix" begin
        X = make_clean_ts_matrix()
        T = DT.fit(ZScore, X; dims=1)
        Y = DT.normalize(X, T)
        valid = [y for y in Y if !ismissing(y)]
        @test all(all(isfinite, v) for v in valid)
    end

    @testset "DT.normalize all-missing column" begin
        X = Matrix{Union{Missing,Vector{Float64}}}(undef, 3, 2)
        X[:, 1] .= missing
        X[:, 2] = [rand(Float64, 5) for _ in 1:3]
        T = DT.fit(ZScore, X)
        Y = DT.normalize(X, T)
        @test all(ismissing, Y[:, 1])
        @test all(!ismissing, Y[:, 2])
    end
end