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

    normalized = DT.normalize.(groups, ZScore)

    for i in eachindex(groups)
        n = fit(ZScore, groups[i], dims=nothing)
        test_norm = Normalization.normalize(groups[i], n)

        @test isapprox(normalized[i], test_norm)
    end
end

# ---------------------------------------------------------------------------- #
#                     DataTreatment groupby normalization                      #
# ---------------------------------------------------------------------------- #
@test_nowarn DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum)
)

@test_nowarn DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=ZScore
)

@test_nowarn DataTreatment(
    Xts,
    aggrtype=:reducesize,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=MinMax
)

@test_nowarn DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    groups=[:vname, :feat],
    norm=PNorm
)

@test_nowarn DataTreatment(
    Xts,
    aggrtype=:reducesize,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    groups=:vname,
    norm=Scale
)

# ---------------------------------------------------------------------------- #
#                         check against normalization                          #
# ---------------------------------------------------------------------------- #
m1 = [1.0 1.0 1.0; 1.0 2.5 1.0; 1.0 1.0 1.0]
m2 = [1.0 1.0 1.0; 1.0 7.5 1.0; 1.0 1.0 1.0]
m3 = [9.0 9.0 9.0; 9.0 2.5 9.0; 9.0 9.0 9.0]
m4 = [9.0 9.0 9.0; 9.0 7.5 9.0; 9.0 9.0 9.0]

M = reshape([m1, m2, m3, m4], 2, 2) # 2x2 matrix of matrices

test1 = DataTreatment(
    M,
    aggrtype=:reducesize;
    win=splitwindow(nwindows=3), #should actually not change the dataset
    features=(mean, maximum),
    groups=:vname,
    norm=MinMax
)

# Since the dataset is composed of two columns named :v1 and :v2 and we have grouped by :vname,
# we expect the dataset to have been normalized separately for :v1 and for :v2
@testset "DataTreatment reducesize groupby normalization" begin
    for (i, m) in enumerate(eachcol(M))
        merged = vcat(m...)
        n = fit(MinMax, merged, dims=nothing)
        test_norm = Normalization.normalize(merged, n)

        res_merged = vcat(get_X(test1)[:,i]...)
        @test isapprox(res_merged, test_norm)
    end

    @test DT.get_groups(test1) isa Vector{Vector{Int64}}
    @test DT.get_groupmethod(test1) isa Symbol
    @test DT.get_norm(test1) == MinMax
end

test2 = DataTreatment(
    M,
    aggrtype=:aggregate;
    win=splitwindow(nwindows=3), #should actually not change the dataset
    features=(mean,),
    groups=:vname,
    norm=MinMax
)

# In this case as well, since a windowing of 3 elements was used, the windowing will generate
# single elements, so we expect the normalization to have been applied separately for :v1 and :v2
@testset "DataTreatment aggregate groupby normalization" begin
    for (i, m) in enumerate(eachcol(M))
        merged = vcat(m...)
        n = fit(MinMax, merged, dims=nothing)
        test_norm = Normalization.normalize(merged, n)

        i == 1 && (i = 1:9)
        i == 2 && (i = 10:18)
        res_merged = vcat(get_X(test2)[:,i]'...)
        @test isapprox(res_merged, vcat(test_norm'...))
    end
end

# ---------------------------------------------------------------------------- #

norm_type = DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=MinMax
)
@test get_norm(norm_type) == MinMax

norm_type = DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=Scale
)
@test get_norm(norm_type) == Scale

norm_type = DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=ZScore
)
@test get_norm(norm_type) == ZScore

norm_type = DataTreatment(
    Xts,
    aggrtype=:aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=PNorm
)
@test get_norm(norm_type) == PNorm
