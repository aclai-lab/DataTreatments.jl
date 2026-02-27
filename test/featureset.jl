using Test
using DataTreatments

using Statistics

X = rand(200, 120)
var = 5
Xmatrix = fill(X, 50, var)
nwindows = 3
win = splitwindow(;nwindows)

@testset "base_set features" begin
    dt = DataTreatment(Xmatrix, aggrtype=:aggregate;
                        vnames=Symbol.("var", 1:var),
                        win,
                        features=DataTreatments.base_set)
    
    @test size(dt, 1) == 50
    # 2d elements, windowing will be applied both in x and y axis, resulting in nwindows^2 elements
    @test length(get_datafeature(dt)) == (length(DataTreatments.base_set) * nwindows^2 * var)
    
    # Check all base features are present
    feature_ids = get_datafeature(dt)
    feature_funcs = unique(get_feat.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test Statistics.std in feature_funcs
end

@testset "catch9 features" begin
    dt = DataTreatment(Xmatrix, aggrtype=:aggregate;
                        vnames=Symbol.("var", 1:5),
                        win,
                        features=DataTreatments.catch9)
    
    @test size(dt, 1) == 50
    @test length(get_datafeature(dt)) == (length(DataTreatments.catch9) * nwindows^2 * var)
    
    # Check statistical features
    feature_ids = get_datafeature(dt)
    feature_funcs = unique(get_feat.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test median in feature_funcs
    @test Statistics.std in feature_funcs
    
    # Check Catch22 features
    @test DataTreatments.stretch_high in feature_funcs
    @test DataTreatments.stretch_decreasing in feature_funcs
    @test DataTreatments.entropy_pairs in feature_funcs
    @test DataTreatments.transition_variance in feature_funcs
end

@testset "catch22_set features" begin
    dt = DataTreatment(Xmatrix, aggrtype=:aggregate;
                        vnames=Symbol.("var", 1:5),
                        win,
                        features=DataTreatments.catch22_set)
    
    @test size(dt, 1) == 50
    @test length(get_datafeature(dt)) == (length(DataTreatments.catch22_set) * nwindows^2 * var)
    
    # Check a sample of Catch22 features
    feature_ids = get_datafeature(dt)
    feature_funcs = unique(get_feat.(feature_ids))
    @test DataTreatments.mode_5 in feature_funcs
    @test DataTreatments.mode_10 in feature_funcs
    @test DataTreatments.embedding_dist in feature_funcs
    @test DataTreatments.acf_timescale in feature_funcs
    @test DataTreatments.periodicity in feature_funcs
    @test DataTreatments.dfa in feature_funcs
end

@testset "complete_set features" begin
    dt = DataTreatment(Xmatrix, aggrtype=:aggregate;
                        vnames=Symbol.("var", 1:5),
                        win,
                        features=DataTreatments.complete_set)
    
    @test size(dt, 1) == 50
    @test length(get_datafeature(dt)) == (length(DataTreatments.complete_set) * nwindows^2 * var)
    
    # Check basic statistics
    feature_ids = get_datafeature(dt)
    feature_funcs = unique(get_feat.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test median in feature_funcs
    @test Statistics.std in feature_funcs
    @test cov in feature_funcs
    
    # Check all catch22 features are included
    for feat in DataTreatments.catch22_set
        @test feat in feature_funcs
    end
end
