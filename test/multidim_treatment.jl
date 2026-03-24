using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays

using InteractiveUtils

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
    DataFrame(
        str_col  = [missing, "blue", "green", "red", "blue"],
        sym_col  = [:circle, :square, :triangle, :square, missing],
        img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5],
        int_col  = Int[10, 20, 30, 40, 50],
        V1 = [NaN, missing, 3.0, 4.0, 5.6],
        V2 = [2.5, missing, 4.5, 5.5, NaN],
        ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
        V4 = [4.1, NaN, NaN, 7.1, 5.5],
        V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
        ts2 = [
            collect(2.0:0.5:5.5),
            collect(1.0:0.5:4.5),
            collect(3.0:0.5:6.5),
            collect(4.0:0.5:7.5),
            NaN
        ],
        ts3 = [
            [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2],
            NaN, NaN, missing,
            [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]
        ],
        V3 = [3.2, 4.2, 5.2, missing, 2.4],
        ts4 = [
            [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8],
            missing,
            [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8],
            [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]
        ],
        img1 = [create_image(i) for i in 1:5],
        cat_col  = categorical(["small", "medium", missing, "small", "large"]),
        uint_col = UInt32[1, 2, 3, 4, 5],
        img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
    )
end

df = build_test_df()

datastruct = DT._inspecting(Matrix(df))

ts = ["ts1","ts2","ts3","ts4"]
ts_idxs = [7, 10, 11, 13]
ts_v = datastruct.valididxs[ts_idxs]
img = ["img4","img1","img2","img3"]
img_idxs = [3, 14, 17, 18]
img_v = datastruct.valididxs[img_idxs]

a = aggregate(
    Matrix(df[!, ts]),
    ts_v,
    Float64;
    win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
    features=(mean, maximum)
)

a = aggregate(
    Matrix(df[!, img]),
    img_v,
    Float64;
    win=(DT.splitwindow(nwindows=2),),
    features=(mean,)
)

a = reducesize(
    Matrix(df[!, ts]),
    ts_v,
    Float64;
    win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
    reducefunc=maximum
)

a = reducesize(
    Matrix(df[!, img]),
    img_v,
    Float64;
    win=(DT.splitwindow(nwindows=2),),
    reducefunc=mean
)

@testset "aggregate output shape and type" begin
    # Test 1: ts columns, adaptivewindow, mean+maximum
    a1, nw1 = aggregate(
        Matrix(df[!, ts]),
        ts_v,
        Float64;
        win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
        features=(mean, maximum)
    )
    @test size(a1, 1) == nrow(df)
    @test size(a1, 2) == sum(nw1) * 2  # 2 features (mean, maximum)
    @test eltype(a1) <: Union{Missing, Float64}
    @test length(nw1) == length(ts)

    # Test 2: img columns, splitwindow, mean
    a2, nw2 = aggregate(
        Matrix(df[!, img]),
        img_v,
        Float64;
        win=(DT.splitwindow(nwindows=2),),
        features=(mean,)
    )
    @test size(a2, 1) == nrow(df)
    @test size(a2, 2) == sum(nw2) * 1  # 1 feature (mean)
    @test eltype(a2) <: Union{Missing, Float64}
    @test length(nw2) == length(img)

    # Spot check: known values for a windowed mean (if possible)
    # For example, check that the first row, first feature is the mean of the first window of ts1
    # (if ts1[1] is NaN, should be missing or NaN)
    if !ismissing(df.ts1[1]) && !(df.ts1[1] isa AbstractFloat && isnan(df.ts1[1]))
        # Compute expected mean for first window of ts1
        win1 = DT.adaptivewindow(nwindows=5, overlap=0.4)(df.ts1[1])
        expected = mean(skipmissing(win1[1]))
        @test a1[1, 1] ≈ expected
    end
end

@testset "reducesize output shape and type" begin
    # Test 3: ts columns, adaptivewindow, maximum
    r1, _ = reducesize(
        Matrix(df[!, ts]),
        ts_v,
        Float64;
        win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
        reducefunc=maximum
    )
    @test size(r1, 1) == nrow(df)
    @test size(r1, 2) == length(ts)
    @test eltype(r1) <: Union{Missing, Float64, Array{Float64}}

    # Test 4: img columns, splitwindow, mean
    r2, _ = reducesize(
        Matrix(df[!, img]),
        img_v,
        Float64;
        win=(DT.splitwindow(nwindows=2),),
        reducefunc=mean
    )
    @test size(r2, 1) == nrow(df)
    @test size(r2, 2) == length(img)
    @test eltype(r2) <: Union{Missing, Float64, Array{Float64}}
end

@testset "get_window_ranges" begin
    @test DT._get_window_ranges(([1:5, 6:10, 11:15],), CartesianIndex(2)) == (6:10,)
    @test DT._get_window_ranges(([1:3, 4:6], [1:2, 3:4, 5:6]), CartesianIndex(1, 3)) == (1:3, 5:6)
    @test DT._get_window_ranges(([1:2, 3:4], [1:3, 4:6], [1:5]), CartesianIndex(2, 1, 1)) == (3:4, 1:3, 1:5)
end

@testset "safe_feat" begin
    @test DT._safe_feat([1.0, 2.0, 3.0, 4.0, 5.0], maximum) == 5.0
    @test DT._safe_feat([1.0, 2.0, 3.0, 4.0, 5.0], mean) == 3.0

    # missing filtered
    @test DT._safe_feat([1.0, missing, 3.0, missing, 5.0], maximum) == 5.0
    @test DT._safe_feat([1.0, missing, 3.0, missing, 5.0], mean) == 3.0

    # NaN filtered
    @test DT._safe_feat([1.0, NaN, 3.0, NaN, 5.0], maximum) == 5.0
    @test DT._safe_feat([1.0, NaN, 3.0, NaN, 5.0], mean) == 3.0

    # both missing and NaN
    @test DT._safe_feat([1.0, missing, NaN, 4.0, missing, NaN, 7.0], maximum) == 7.0
    @test DT._safe_feat([1.0, missing, NaN, 4.0, missing, NaN, 7.0], mean) == 4.0

    # all invalid → empty vector
    @test_throws Exception DT._safe_feat(Union{Missing,Float64}[missing, missing], maximum)
    @test isnan(DT._safe_feat([NaN, NaN, NaN], mean))

    # single valid element
    @test DT._safe_feat([missing, NaN, 42.0, missing], maximum) == 42.0

    # integers (no NaN possible)
    @test DT._safe_feat(Union{Missing,Int}[1, missing, 3, missing, 5], sum) == 9

    # custom function
    @test DT._safe_feat([1.0, missing, 3.0, NaN, 5.0], x -> length(x)) == 3
    @test DT._safe_feat([1.0, missing, 3.0, NaN, 5.0], std) ≈ std([1.0, 3.0, 5.0])
end

@testset "curried constructors" begin
    @test aggregate() isa Function
    @test aggregate(win=(movingwindow(winsize=3, winstep=2),), features=(sum, mean)) isa Function
    @test aggregate(features=(maximum,)) isa Function
    @test aggregate(win=(splitwindow(nwindows=3),), features=(mean,)) isa Function

    @test reducesize() isa Function
    @test reducesize(win=(splitwindow(nwindows=4),), reducefunc=mean) isa Function
    @test reducesize(reducefunc=sum) isa Function
end

@test_nowarn @inferred aggregate(
    Matrix(df[!, ts]),
    ts_v,
    Float64;
    win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
    features=(mean, maximum)
)

@test_nowarn @inferred aggregate(
    Matrix(df[!, img]),
    img_v,
    Float64;
    win=(DT.splitwindow(nwindows=2),),
    features=(mean,)
)

@test_nowarn @inferred reducesize(
    Matrix(df[!, ts]),
    ts_v,
    Float64;
    win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
    reducefunc=maximum
)

@test_nowarn @inferred reducesize(
    Matrix(df[!, img]),
    img_v,
    Float64;
    win=(DT.splitwindow(nwindows=2),),
    reducefunc=mean
)

@test_nowarn InteractiveUtils.@code_warntype aggregate(
    Matrix(df[!, ts]),
    ts_v,
    Float64;
    win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
    features=(mean, maximum)
)

@test_nowarn InteractiveUtils.@code_warntype aggregate(
    Matrix(df[!, img]),
    img_v,
    Float64;
    win=(DT.splitwindow(nwindows=2),),
    features=(mean,)
)

@test_nowarn InteractiveUtils.@code_warntype reducesize(
    Matrix(df[!, ts]),
    ts_v,
    Float64;
    win=(DT.adaptivewindow(nwindows=5, overlap=0.4),),
    reducefunc=maximum
)

@test_nowarn InteractiveUtils.@code_warntype reducesize(
    Matrix(df[!, img]),
    img_v,
    Float64;
    win=(DT.splitwindow(nwindows=2),),
    reducefunc=mean
)
