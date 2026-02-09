using Test
using DataTreatments
const DT = DataTreatments

using Normalization
using Statistics
using MLJ, DataFrames
using SoleData: Artifacts

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                       DataFrame groupby normalization                        #
# ---------------------------------------------------------------------------- #
@testset "DataFrame groupby normalization" begin
    fileds = [[:sepal_length, :petal_length], [:sepal_width]]
    groups = DT.groupby(Xc, fileds)

    normalized = DT.normalize(groups, DT.zscore())

    for i in eachindex(groups)
        group_data = vec(Matrix(groups[i]))
        n = fit(ZScore, groups[i], dims=nothing)
        test_norm = Normalization.normalize(groups[i], n)
    end

    # Group 1: sepal_length and petal_length combined
    group1_data = vec(Matrix(groups[1]))
    μ = mean(group1_data)
    σ = std(group1_data)
    group1_expected = (group1_data .- μ) ./ σ
    
    # Verify the normalized group matches
    group1_result = vec(Matrix(normalized[1]))
    @test group1_result ≈ group1_expected
    
    # Group 2: sepal_width alone
    group2_data = vec(Matrix(groups[2]))
    μ2 = mean(group2_data)
    σ2 = std(group2_data)
    group2_expected = (group2_data .- μ2) ./ σ2
    group2_result = vec(Matrix(normalized[2]))
    @test group2_result ≈ group2_expected
    
    # Alternative: test properties of zscore normalization
    @test mean(group1_result) ≈ 0.0 atol=1e-10
    @test std(group1_result) ≈ 1.0 atol=1e-10
    @test mean(group2_result) ≈ 0.0 atol=1e-10
    @test std(group2_result) ≈ 1.0 atol=1e-10
end
