using Test
using DataTreatments

using Statistics

X = rand(200, 120)
Xmatrix = fill(X, 50, 5)
win = splitwindow(nwindows=3)

@testset "base_set features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=base_set)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(base_set)
    @test Set(get_features(dt)) == Set(base_set)
    
    # Check all base features are present
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test Statistics.std in feature_funcs
end

@testset "catch9 features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=catch9)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(catch9)
    
    # Check statistical features
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test median in feature_funcs
    @test Statistics.std in feature_funcs
    
    # Check Catch22 features
    @test stretch_high in feature_funcs
    @test stretch_decreasing in feature_funcs
    @test entropy_pairs in feature_funcs
    @test transition_variance in feature_funcs
end

@testset "catch22_set features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=catch22_set)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(catch22_set)
    
    # Check a sample of Catch22 features
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test mode_5 in feature_funcs
    @test mode_10 in feature_funcs
    @test embedding_dist in feature_funcs
    @test acf_timescale in feature_funcs
    @test periodicity in feature_funcs
    @test dfa in feature_funcs
end

@testset "complete_set features" begin
    dt = DataTreatment(Xmatrix, :aggregate;
                        vnames=Symbol.("var", 1:5),
                        win=(win,),
                        features=complete_set)
    
    @test size(dt, 1) == 50
    @test length(get_features(dt)) == length(complete_set)
    
    # Check basic statistics
    feature_ids = get_featureid(dt)
    feature_funcs = unique(get_feature.(feature_ids))
    @test maximum in feature_funcs
    @test minimum in feature_funcs
    @test mean in feature_funcs
    @test median in feature_funcs
    @test Statistics.std in feature_funcs
    @test cov in feature_funcs
    
    # Check all catch22 features are included
    for feat in catch22_set
        @test feat in feature_funcs
    end
end
