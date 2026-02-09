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

    normalized = DT.normalize(groups, DT.zscore(); dims=0)

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
    :aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum)
)

@test_nowarn DataTreatment(
    Xts,
    :aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=DT.minmax(lower=0.0, upper=1.0)
)

@test_nowarn DataTreatment(
    Xts,
    :reducesize,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    norm=DT.minmax(lower=0.0, upper=1.0)
)

@test_nowarn DataTreatment(
    Xts,
    :aggregate,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    groups=(:vname, :feat),
    norm=DT.minmax(lower=0.0, upper=1.0)
)

@test_nowarn DataTreatment(
    Xts,
    :reducesize,
    win=splitwindow(nwindows=2),
    features=(mean, maximum),
    groups=(:vname, :feat),
    norm=DT.minmax(lower=0.0, upper=1.0)
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
    :reducesize,
    vnames = [:V1,:V2],
    win=splitwindow(nwindows=3), #should actually not change the dataset
    features=(mean, maximum),
    groups=(:vname,),
    norm=DT.minmax(lower=0.0, upper=1.0)
)

# Since the dataset is composed of two columns named :v1 and :v2 and we have grouped by :vname,
# we expect the dataset to have been normalized separately for :v1 and for :v2

@testset "DataTreatment reducesize groupby normalization" begin
    for (i, m) in enumerate(eachcol(M))
        merged = vcat(m...)
        n = fit(MinMax, merged, dims=nothing)
        test_norm = Normalization.normalize(merged, n)

        res_merged = vcat(get_dataset(test1)[:,i]...)
        @test isapprox(res_merged, test_norm)
    end
end

test2 = DataTreatment(
    M,
    :aggregate,
    vnames = [:V1,:V2],
    win=splitwindow(nwindows=3), #should actually not change the dataset
    features=(mean,),
    groups=(:vname,),
    norm=DT.minmax(lower=0.0, upper=1.0)
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
        res_merged = vcat(get_dataset(test2)[:,i]'...)
        @test isapprox(res_merged, vcat(test_norm'...))
    end
end