using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Statistics

@testset "aggregate - Flatten to Tabular" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    vnames = Symbol.("var", 1:size(Xmatrix, 2))
    nwindows = 3
    win = splitwindow(; nwindows)
    features = (mean, maximum)
    
    result = DataTreatment(Xmatrix, :aggregate; vnames, win, features)
    
    @test get_aggrtype(result) == :aggregate
    @test size(get_dataset(result), 1) == size(Xmatrix, 1)  # Same number of rows
    # features are matrices, multidim windowing is: nwindows * nwindows
    @test size(get_dataset(result), 2) == size(Xmatrix, 2) * nwindows^2 * length(features)
    @test length(get_featureid(result)) == 180
    @test isnothing(get_norm(result))
    @test eltype(result) == Float64
end

@testset "reducesize - Reduce elements size" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    vnames = Symbol.("var", 1:size(Xmatrix, 2))
    win = splitwindow(nwindows=3)
    
    result = DataTreatment(Xmatrix, :reducesize; vnames, win, reducefunc=Statistics.std)
    
    @test size(get_dataset(result)) == size(Xmatrix)
    @test eltype(result) == typeof(first(get_dataset(result)))
    @test size(first(get_dataset(result))) == (3, 3)
    @test get_reducefunc(result) == std
end

@testset "FeatureId" begin
    @testset "Creation and accessors" begin
        fid = FeatureId(:temperature, mean, 1)
        
        @test get_vname(fid) == :temperature
        @test get_feat(fid) == mean
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
        @test get_feat(feature_ids[1]) == mean
        @test get_vname(feature_ids[1]) == :var1
        @test get_feat(feature_ids[2]) == Statistics.std
        @test get_vname(feature_ids[2]) == :var1
    end

    @testset "get FeatureId vectors" begin
        X = fill(rand(10), 8, 3)
        win = splitwindow(nwindows=2)
        features = (mean, std)

        dt = DataTreatment(X, :aggregate;
                            vnames=Symbol.("var", 1:4),
                            win=(win,),
                            features=features)
    
        vnames_vec = get_vecvnames(dt.featureid)
        @test length(vnames_vec) == 16
        @test unique(vnames_vec) == [:var1, :var2, :var3, :var4]

        feat_vec = get_vecfeatures(dt.featureid)
        @test unique(feat_vec) == [mean, std]

        win_vec = get_vecnwins(dt.featureid)
        @test unique(win_vec) == [1, 2]
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
    @test length(props) == 6
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

@testset "DataTreatment - aggregate" begin
    X = rand(200, 120)
    Xmatrix = fill(X, 50, 5)
    win = splitwindow(nwindows=3)
    features = (mean, std, maximum)
    
    @testset "element_norm - Z-score" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                          vnames=Symbol.("var", 1:5),
                          win=(win,),
                          features=features,
                          norm=zscore())
        
        @test abs(mean(dt.dataset)) < 1e-10  # mean ≈ 0
        @test abs(std(dt.dataset) - 1.0) < 1e-3  # std ≈ 1
    end
    
    @testset "element_norm - MinMax" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                          vnames=Symbol.("var", 1:5),
                          win=(win,),
                          features=features,
                          norm=DT.minmax())
        
        @test minimum(dt.dataset) ≈ 0.0 atol=1e-10
        @test maximum(dt.dataset) ≈ 1.0 atol=1e-10
    end
    
    @testset "element_norm - Sigmoid" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                          vnames=Symbol.("var", 1:5),
                          win=(win,),
                          features=features,
                          norm=sigmoid())
        
        @test all(0 .< dt.dataset .< 1)
    end
    
    @testset "element_norm - Center" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                          vnames=Symbol.("var", 1:5),
                          win=(win,),
                          features=features,
                          norm=center())
        
        @test abs(mean(dt.dataset)) < 1e-10
    end
    
    @testset "element_norm - Unit Power" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                          vnames=Symbol.("var", 1:5),
                          win=(win,),
                          features=features,
                          norm=unitpower())
        
        rms = sqrt(mean(abs2, dt.dataset))
        @test rms ≈ 1.0 atol=1e-10
    end
    
    @testset "element_norm - Outlier Suppress" begin
        # Create data with outliers
        X_outlier = rand(200, 120)
        X_outlier[1, 1] = 1000.0  # Add outlier
        Xmatrix_outlier = fill(X_outlier, 50, 5)
        
        dt = DataTreatment(Xmatrix_outlier, :aggregate;
                          vnames=Symbol.("var", 1:5),
                          win=(win,),
                          features=features,
                          norm=outliersuppress(;thr=0.3))
        
        @test maximum(dt.dataset) < maximum(X_outlier)
    end
end

@testset "DataTreatment - reducesize" begin
    # Create nested array structure
    X_nested = [rand(100) * 100 for _ in 1:30, _ in 1:3]
    win = splitwindow(nwindows=2)
    features = (mean, std)
    
    @testset "Basic ds_norm" begin
        dt = DataTreatment(X_nested, :reducesize;
                          vnames=Symbol.("ch", 1:3),
                          win=(win,),
                          features=features,
                          norm=DT.minmax())
        
        @test minimum(minimum.(dt.dataset)) ≈ 0.0 atol=1e-10
        @test maximum(maximum.(dt.dataset)) ≈ 1.0 atol=1e-10
    end
end

@testset "has_uniform_element_size - DataFrame" begin
    # empty DataFrame
    @test has_uniform_element_size(DataFrame()) === true

    # uniform element sizes (all 2x3 matrices)
    df_uniform = DataFrame(
        a = [rand(2,3) for _ in 1:3],
        b = [rand(2,3) for _ in 1:3]
    )
    @test has_uniform_element_size(df_uniform) === true

    # mismatch in one element
    df_mismatch = DataFrame(
        a = [rand(2,3), rand(3,3), rand(2,3)],
        b = [rand(2,3) for _ in 1:3]
    )
    @test has_uniform_element_size(df_mismatch) === false
end

@testset "has_uniform_element_size - AbstractArray" begin
    # empty array of arrays
    empty_arr = Matrix{Vector{Float64}}(undef, 0, 0)
    @test has_uniform_element_size(empty_arr) === true

    # uniform sizes (all 2x3 matrices)
    A = [rand(2,3) for _ in 1:4, _ in 1:2]
    @test has_uniform_element_size(A) === true

    # introduce a mismatch
    A_bad = copy(A)
    A_bad[3, 2] = rand(3,3)
    @test has_uniform_element_size(A_bad) === false
end

# helper to build independent copies
function make_uniform_matrix(vec, rows, cols)
    [copy(vec) for _ in 1:rows, _ in 1:cols]
end

@testset "aggregate/reducesize - uniform element sizes" begin
    v10 = collect(1.0:10.0)
    Xu  = make_uniform_matrix(v10, 2, 3)
    vnames = Symbol.("var", 1:size(Xu, 2))

    nwindows = 2
    win = splitwindow(; nwindows)
    features = (mean, maximum)

    # aggregate with uniform=true
    Au = DataTreatment(Xu, :aggregate; vnames, features, win)
    @test size(get_dataset(Au)) == (2, 3 * length(features) * nwindows)

    # reducesize with uniform=true
    Ru = DataTreatment(Xu, :reducesize; vnames, reducefunc=mean, win)
    @test size(get_dataset(Ru)) == size(Xu)
    @test length(get_dataset(Ru)[1,1]) == nwindows
    intervals_u = first(@evalwindow v10 win)
    @test get_dataset(Ru)[1,1] ≈ [mean(v10[r]) for r in intervals_u]
end

@testset "aggregate/reducesize - non-uniform element sizes (adaptivewindow)" begin
    v8  = collect(1.0:8.0)
    v10 = collect(1.0:10.0)
    v12 = collect(1.0:12.0)

    # non-uniform matrix of vectors (avoid hvcat)
    Xn = Matrix{Vector{Float64}}(undef, 2, 3)
    Xn[1,1] = copy(v10); Xn[1,2] = copy(v12); Xn[1,3] = copy(v8)
    Xn[2,1] = copy(v12); Xn[2,2] = copy(v8);  Xn[2,3] = copy(v10)
    vnames = Symbol.("var", 1:size(Xn, 2))

    nwindows = 2
    win = adaptivewindow(; nwindows, overlap=0.0)
    features = (mean, maximum)

    An = DataTreatment(Xn, :aggregate; vnames, win, features)
    @test size(get_dataset(An)) == (2, 3 * length(features) * nwindows)

    # reducesize with uniform=false and adaptivewindow
    Rn = DataTreatment(Xn, :reducesize; vnames, win, reducefunc=mean)
    @test size(get_dataset(Rn)) == size(Xn)
    intervals_e = first(@evalwindow Xn[1,1] win)
    @test get_dataset(Rn)[1,1] ≈ [mean(Xn[1,1][r]) for r in intervals_e]
end

@testset "dataset utilities" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    @test is_multidim_dataset(Xmatrix) == true
    
    intervals = (UnitRange{Int}[1:50, 51:100, 101:150, 151:200],
            UnitRange{Int}[1:30, 31:60, 61:90, 91:120])
    @test nvals(intervals) == 16
end

@testset "wholewindow" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    vnames = Symbol.("var", 1:size(Xmatrix, 2))
    win = wholewindow()
    features = (mean, maximum)
    
    result = DataTreatment(Xmatrix, :aggregate; vnames, win, features)
    
    @test length(get_featureid(result)) == (length(vnames) * length(features))
end

@testset "DataTreatment getindex (single Int)" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    vnames = Symbol.("var", 1:size(Xmatrix, 2))
    dt = DataTreatment(Xmatrix, :aggregate; vnames, win=wholewindow(), features=(mean,))

    i = 1
    @test dt[i] == get_dataset(dt)[:, i]
end
