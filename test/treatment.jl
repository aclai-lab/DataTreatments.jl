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

Xmatrix = [rand(Float64, 4, 2) for _ in 1:6, _ in 1:5]

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
@testset "aggregate - Flatten to Tabular" begin
    nwindows = 2
    win = splitwindow(; nwindows)
    features = (mean, maximum)
    
    result = DataTreatment(Xmatrix, aggrtype=:aggregate; win, features)
    
    @test all(isa.(get_datafeature(result), DT.AggregateFeat))
    @test size(get_X(result), 1) == size(Xmatrix, 1)  # Same number of rows
    # features are matrices, multidim windowing is: nwindows * nwindows
    @test size(get_X(result), 2) == size(Xmatrix, 2) * nwindows^2 * length(features)
    @test length(get_datafeature(result)) == 40
    @test isnothing(get_norm(result))
    @test eltype(result) == Float64
end

@testset "reducesize - Reduce elements size" begin
    vnames = Symbol.("var", 1:size(Xmatrix, 2))
    win = splitwindow(nwindows=3)
    
    result = DataTreatment(Xmatrix, aggrtype=:reducesize; vnames, win, reducefunc=Statistics.std)
    
    @test size(get_X(result)) == size(Xmatrix)
    @test typeof(first(get_X(result))) <: eltype(result)
    @test size(first(get_X(result))) == (3, 3)
    @test get_reducefuncs(result) == [std]
end

# ---------------------------------------------------------------------------- #
#                               DataFeature types                              #
# ---------------------------------------------------------------------------- #
@testset "TabularFeat" begin
    dt = DataTreatment(Xc, yc)
    feats = get_datafeature(dt)

    @test all(f -> f isa DT.TabularFeat, feats)
    @test length(feats) == 4
    @test get_vname.(feats) == [:sepal_length, :sepal_width, :petal_length, :petal_width]
    @test all(f -> get_type(f) == Float64, feats)
    @test get_id.(feats) == collect(1:4)
end

@testset "AggregateFeat" begin
    dt = DataTreatment(Xts, yts; aggrtype=:aggregate,
                       win=splitwindow(nwindows=3), features=(mean, maximum))
    feats = get_datafeature(dt)

    @test all(f -> f isa DT.AggregateFeat, feats)
    @test unique(get_vname.(feats)) == propertynames(Xts)
    @test unique(get_feat.(feats))  == [mean, maximum]
    @test all(f -> get_nwin(f) isa Int, feats)
end

@testset "ReduceFeat" begin
    dt = DataTreatment(Xts, yts; aggrtype=:reducesize,
                       win=wholewindow(), reducefunc=std)
    feats = get_datafeature(dt)

    @test all(f -> f isa DT.ReduceFeat, feats)
    @test unique(get_vname.(feats)) == propertynames(Xts)
    @test all(f -> get_reducefunc(f) == std, feats)
end
