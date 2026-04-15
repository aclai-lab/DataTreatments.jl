using Test

using DataTreatments
const DT = DataTreatments

using MLJ
using DataFrames

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# create imbalanced dataset
X = vcat(Xc[1:25, :], Xc[51:100, :], Xc[101:135, :])
y = vcat(yc[1:25], yc[51:100], yc[101:135])

@testset "Imbalance structs" begin
    @testset "RandomOversampler" begin
        dt = load_dataset(
            X, y;
            balance=DT.RandomOversampler()
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    @testset "RandomWalkOversampler" begin
        dt = load_dataset(
            X, y;
            balance=DT.RandomWalkOversampler()
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    @testset "ROSE" begin
        dt = load_dataset(
            X, y;
            balance=DT.ROSE()
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    @testset "SMOTE" begin
        dt = load_dataset(
            X, y;
            balance=DT.SMOTE(k=5)
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    @testset "BorderlineSMOTE1" begin
        dt = load_dataset(
            X, y;
            balance=DT.BorderlineSMOTE1()
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    @testset "SMOTENC" begin
        dt = load_dataset(
            X, y;
            balance=DT.SMOTENC()
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    # -------------------------------------------------------------------- #
    #                 undersampling via load_dataset                        #
    # -------------------------------------------------------------------- #
    @testset "RandomUndersampler" begin
        dt = load_dataset(
            X, y;
            balance=DT.RandomUndersampler()
        )
        t = DT.get_target(dt)
        @test length(t) ≤ length(y)
    end

    @testset "ClusterUndersampler" begin
        dt = load_dataset(
            X, y;
            balance=DT.ClusterUndersampler()
        )
        t = DT.get_target(dt)
        @test length(t) ≤ length(y)
    end

    @testset "ENNUndersampler" begin
        dt = load_dataset(
            X, y;
            balance=DT.ENNUndersampler()
        )
        t = DT.get_target(dt)
        @test length(t) ≤ length(y)
    end

    @testset "TomekUndersampler" begin
        dt = load_dataset(
            X, y;
            balance=DT.TomekUndersampler()
        )
        t = DT.get_target(dt)
        @test length(t) ≤ length(y)
    end

    # -------------------------------------------------------------------- #
    #                     chained balance (tuple)                          #
    # -------------------------------------------------------------------- #
    @testset "Chained SMOTE + TomekUndersampler" begin
        dt = load_dataset(
            X, y;
            balance=(
                DT.SMOTE(k=5),
                DT.TomekUndersampler(),
            )
        )
        t = DT.get_target(dt)
        @test length(t) > 0
    end

    @testset "Chained ROSE + ENNUndersampler" begin
        dt = load_dataset(
            X, y;
            balance=(
                DT.ROSE(),
                DT.ENNUndersampler(),
            )
        )
        t = DT.get_target(dt)
        @test length(t) > 0
    end

    # -------------------------------------------------------------------- #
    #                       only dicrete dataset                           #
    # -------------------------------------------------------------------- #
    Xd = DataFrame(
        f1 = categorical(round.(Int, X[:, 1] .* 10)),
        f2 = categorical(round.(Int, X[:, 2] .* 10)),
        f3 = categorical(round.(Int, X[:, 3] .* 10)),
        f4 = categorical(round.(Int, X[:, 4] .* 10)),
    )

    @testset "SMOTEN" begin
        dt = load_dataset(
            Xd, y;
            balance=DT.SMOTEN()
        )
        t = DT.get_target(dt)
        @test length(t) ≥ length(y)
    end

    # -------------------------------------------------------------------- #
    #                       no balance (nothing)                           #
    # -------------------------------------------------------------------- #
    @testset "No balance" begin
        dt = load_dataset(X, y; balance=nothing)
        t = DT.get_target(dt)
        @test length(t) == length(y)
    end
end
