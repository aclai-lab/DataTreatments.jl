using Test
using DataTreatments

using DataFrames
using Statistics

@testset "aggregate - Flatten to Tabular" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    wfunc = splitwindow(nwindows=3)
    intervals = @evalwindow X wfunc
    features = (mean, maximum)
    
    result = aggregate(Xmatrix, intervals; features)
    
    @test size(result, 1) == size(Xmatrix, 1)  # Same number of rows
    @test size(result, 2) == size(Xmatrix, 2) * prod(length.(intervals)) * length(features)
    @test eltype(result) == Float64
end

@testset "reducesize - Reduce elements size" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    wfunc = splitwindow(nwindows=3)
    intervals = @evalwindow X wfunc
    
    result = reducesize(Xmatrix, intervals; reducefunc=Statistics.std)
    
    @test size(result) == size(Xmatrix)
    @test eltype(result) == typeof(first(result))
    @test size(first(result)) == (3, 3)
end

@testset "FeatureId" begin
    @testset "Creation and accessors" begin
        fid = FeatureId(:temperature, mean, 1)
        
        @test get_vname(fid) == :temperature
        @test get_feature(fid) == mean
        @test get_nwin(fid) == 1
    end
    
    @testset "Display single window" begin
        fid = FeatureId(:temperature, mean, 1)
        str = sprint(show, fid)
        @test occursin("mean", str)
        @test occursin("temperature", str)
        @test !occursin("_w", str)
    end
    
    @testset "Display multi-window" begin
        fid = FeatureId(:pressure, maximum, 3)
        str = sprint(show, fid)
        @test occursin("maximum", str)
        @test occursin("pressure", str)
        @test occursin("_w3", str)
    end
end

@testset "DataTreatment - Matrix Input" begin
    Xmatrix = fill(rand(200, 120), 100, 10)
    win = splitwindow(nwindows=4)
    features = (mean, std, maximum)
    
    @testset ":reducesize mode" begin
        dt = DataTreatment(Xmatrix, :reducesize; 
                            vnames=Symbol.("var", 1:10),
                            win=(win,), 
                            reducefunc=mean)
        
        @test size(dt, 1) == 100
        @test get_aggrtype(dt) == :reducesize
        @test length(get_vnames(dt)) == 10
    end
    
    @testset ":aggregate mode" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                            vnames=Symbol.("var", 1:10),
                            win=(win,),
                            features=features)
        
        @test size(dt) == (100, 10 * length(features) * 16)  # 10 vars × 3 features × 16 windows
        @test length(get_featureid(dt)) == size(dt, 2)
        @test get_aggrtype(dt) == :aggregate
        @test length(get_vnames(dt)) == 10
        @test length(get_features(dt)) == length(features)
    end
end

@testset "FeatureId - Extended Tests" begin
    @testset "propertynames" begin
        fid = FeatureId(:temperature, mean, 1)
        props = propertynames(fid)
        
        @test :vname in props
        @test :feat in props
        @test :nwin in props
        @test length(props) == 3
    end
    
    @testset "MIME text/plain display" begin
        fid = FeatureId(:pressure, std, 5)
        str = sprint(show, MIME("text/plain"), fid)
        
        @test occursin("FeatureId:", str)
        @test occursin("std", str)
        @test occursin("pressure", str)
        @test occursin("_w5", str)
    end
    
    @testset "Single window FeatureId creation" begin
        vnames = [:var1, :var2, :var3]
        features = (mean, Statistics.std, maximum)
        
        # Simulate single window case
        feature_ids = [FeatureId(v, f, 1) for f in features, v in vnames] |> vec
        
        @test length(feature_ids) == length(vnames) * length(features)
        @test all(get_nwin(fid) == 1 for fid in feature_ids)
        
        # Check ordering: features × variables
        @test get_feature(feature_ids[1]) == mean
        @test get_vname(feature_ids[1]) == :var1
        @test get_feature(feature_ids[2]) == Statistics.std
        @test get_vname(feature_ids[2]) == :var1
    end
end

@testset "DataTreatment - DataFrame Constructor" begin
    @testset "Automatic vnames from DataFrame" begin
        df = DataFrame(
            ch1 = [rand(100) for _ in 1:50],
            ch2 = [rand(100) for _ in 1:50],
            ch3 = [rand(100) for _ in 1:50]
        )
        
        win = splitwindow(nwindows=3)
        features = (mean, std)
        
        # Test without specifying vnames (should use propertynames)
        dt = DataTreatment(df, :reducesize; win=(win,), features=features)
        
        @test Set(get_vnames(dt)) == Set([:ch1, :ch2, :ch3])
        @test size(dt, 1) == 50
        @test get_aggrtype(dt) == :reducesize
    end
    
    @testset "Explicit vnames override" begin
        df = DataFrame(
            ch1 = [rand(100) for _ in 1:50],
            ch2 = [rand(100) for _ in 1:50],
            ch3 = [rand(100) for _ in 1:50]
        )
        
        win = splitwindow(nwindows=3)
        features = (mean, std)
        
        # Override with custom names
        custom_names = [:custom1, :custom2, :custom3]
        dt = DataTreatment(df, :reducesize; 
                          vnames=custom_names, 
                          win=(win,), 
                          features=features)
        
        @test Set(get_vnames(dt)) == Set(custom_names)
    end
end

@testset "DataTreatment - propertynames" begin
    df = DataFrame(
        var1 = [rand(100) for _ in 1:30],
        var2 = [rand(100) for _ in 1:30]
    )
    
    win = splitwindow(nwindows=2)
    dt = DataTreatment(df, :reducesize; win=(win,), features=(mean,))
    
    props = propertynames(dt)
    
    @test :dataset in props
    @test :featureid in props
    @test :reducefunc in props
    @test :aggrtype in props
    @test length(props) == 4
end

@testset "DataTreatment - Accessor Functions" begin
    X = rand(200, 120)
    Xmatrix = fill(X, 50, 5)
    win = splitwindow(nwindows=3)
    features = (mean, std, maximum)
    
    dt = DataTreatment(Xmatrix, :aggregate;
                      vnames=Symbol.("var", 1:5),
                      win=(win,),
                      features=features)
    
    @testset "get_dataset" begin
        dataset = get_dataset(dt)
        @test dataset isa AbstractMatrix
        @test size(dataset) == size(dt.dataset)
        @test dataset === dt.dataset
    end
    
    @testset "get_reducefunc" begin
        reducefunc = get_reducefunc(dt)
        @test reducefunc isa Base.Callable
        @test reducefunc == mean  # default value
    end
    
    @testset "get_nwindows" begin
        nwindows = get_nwindows(dt)
        @test nwindows == 9
        @test nwindows == maximum(get_nwin.(dt.featureid))
    end
end

@testset "DataTreatment - Base Methods" begin
    X = rand(200, 120)
    Xmatrix = fill(X, 40, 4)
    win = splitwindow(nwindows=2)
    features = (mean, std)
    
    dt = DataTreatment(Xmatrix, :aggregate;
                      vnames=Symbol.("var", 1:4),
                      win=(win,),
                      features=features)
    
    @testset "length" begin
        @test length(dt) == length(dt.featureid)
        @test length(dt) == 4 * 2 * 4  # 4 vars × 2 features × 4 windows (2×2)
    end
    
    @testset "eltype" begin
        @test eltype(dt) == eltype(dt.dataset)
        @test eltype(dt) == Float64
    end
    
    @testset "Indexing - single column" begin
        col = dt[:, 1]
        @test length(col) == size(dt, 1)
        @test col == dt.dataset[:, 1]
    end
    
    @testset "Indexing - single row" begin
        row = dt[1, :]
        @test length(row) == size(dt, 2)
        @test row == dt.dataset[1, :]
    end
    
    @testset "Indexing - single element" begin
        elem = dt[1, 1]
        @test elem == dt.dataset[1, 1]
        @test elem isa Number
    end
    
    @testset "Indexing - range" begin
        subset = dt[1:10, :]
        @test size(subset, 1) == 10
        @test size(subset, 2) == size(dt, 2)
    end
    
    @testset "Indexing - general" begin
        subset = dt[1:5, 1:10]
        @test size(subset) == (5, 10)
    end
end

@testset "DataTreatment - Display Methods" begin
    X = rand(200, 120)
    Xmatrix = fill(X, 30, 3)
    win = splitwindow(nwindows=2)
    features = (mean, std, maximum)
    
    dt = DataTreatment(Xmatrix, :aggregate;
                      vnames=Symbol.("var", 1:3),
                      win=(win,),
                      features=features)
    
    @testset "Compact show" begin
        str = sprint(show, dt)
        
        @test occursin("DataTreatment", str)
        @test occursin("aggregate", str)
        @test occursin("30×", str)  # rows
        @test occursin("features", str)
    end
    
    @testset "MIME text/plain - few features" begin
        # Create a small case with ≤10 features
        X_small = rand(100)
        Xmatrix_small = fill(X_small, 20, 2)
        win_small = splitwindow(nwindows=2)
        features_small = (mean, std)
        
        dt_small = DataTreatment(Xmatrix_small, :reducesize;
                                vnames=[:v1, :v2],
                                win=(win_small,),
                                features=features_small)
        
        str = sprint(show, MIME("text/plain"), dt_small)
        
        @test occursin("DataTreatment:", str)
        @test occursin("Type: reducesize", str)
        @test occursin("Dimensions:", str)
        @test occursin("Features:", str)
        @test occursin("Reduction function:", str)
        @test occursin("Feature IDs:", str)
        
        # Check that all features are listed (≤10)
        nfeatures = length(dt_small.featureid)
        @test nfeatures <= 10
        for i in 1:nfeatures
            @test occursin("$(i).", str)
        end
    end
    
    @testset "MIME text/plain - many features" begin
        # Use the original dt with more features
        str = sprint(show, MIME("text/plain"), dt)
        
        @test occursin("DataTreatment:", str)
        @test occursin("Dimensions:", str)
        @test occursin("Features:", str)
    end
    
    @testset "Display with different aggrtype" begin
        dt_agg = DataTreatment(Xmatrix, :aggregate;
                              vnames=Symbol.("var", 1:3),
                              win=(win,),
                              features=features)
        
        str = sprint(show, dt_agg)
        @test occursin("aggregate", str)
        
        str_plain = sprint(show, MIME("text/plain"), dt_agg)
        @test occursin("Type: aggregate", str_plain)
    end
end

@testset "DataTreatment - Edge Cases" begin
    @testset "Single variable" begin
        X = rand(100)
        Xmatrix = fill(X, 50, 1)
        win = splitwindow(nwindows=3)
        
        dt = DataTreatment(Xmatrix, :reducesize;
                          vnames=[:single],
                          win=(win,),
                          features=(mean,))
        
        @test get_vnames(dt) == [:single]
        @test size(dt, 1) == 50
    end
    
    @testset "Single feature" begin
        X = rand(100)
        Xmatrix = fill(X, 30, 3)
        win = splitwindow(nwindows=2)
        
        dt = DataTreatment(Xmatrix, :reducesize;
                          vnames=Symbol.("var", 1:3),
                          win=(win,),
                          features=(mean,))
        
        feature_funcs = get_features(dt)
        @test length(feature_funcs) == 1
        @test mean in feature_funcs
    end
    
    @testset "Single window (wholewindow)" begin
        X = rand(100)
        Xmatrix = fill(X, 40, 2)
        win = wholewindow()
        features = (mean, std)
        
        dt = DataTreatment(Xmatrix, :reducesize;
                          vnames=[:v1, :v2],
                          win=(win,),
                          features=features)
        
        @test get_nwindows(dt) == 1
        @test all(get_nwin(fid) == 1 for fid in dt.featureid)
    end
end