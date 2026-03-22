using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays
using Statistics
using Test

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    V3 = [3.2, 4.2, 5.2, missing, 2.4],
    V4 = [4.1, NaN, NaN, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
    ts3 = [[1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2], NaN, NaN, missing, [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]],
    ts4 = [[6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8], missing, [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8], [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8], [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

ds_struct = DT.DataStructure(df)


"""Build X matrix and idx vector for the given column names."""
function _build_Xidx(colnames::Vector{String})
    colidxs = [findfirst(==(c), names(df)) for c in colnames]
    X = Matrix{Any}(undef, nrow(df), length(colidxs))
    for (j, c) in enumerate(colidxs)
        X[:, j] = df[!, c]
    end
    idx = [_get_valid(df, c) for c in colidxs]
    X, idx
end

"""Build X matrix and idx vector for all columns selected by a TreatmentGroup."""
function _build_Xidx(tg::TreatmentGroup)
    cols = DT.get_idxs(tg)
    X = Matrix{Any}(undef, nrow(df), length(cols))
    for (j, c) in enumerate(cols)
        X[:, j] = df[!, c]
    end
    idx = [_get_valid(df, c) for c in cols]
    X, idx
end

"""Return indices of rows that are neither missing nor scalar NaN."""
function _get_valid(dataframe::DataFrame, colidx::Int)
    col = dataframe[!, colidx]
    [i for i in eachindex(col) if !ismissing(col[i]) && !(col[i] isa Number && isnan(col[i]))]
end

# ---------------------------------------------------------------------------- #
#                              get_window_ranges                               #
# ---------------------------------------------------------------------------- #
@testset "get_window_ranges" begin
    @test DT.get_window_ranges(([1:5, 6:10, 11:15],), CartesianIndex(2)) == (6:10,)
    @test DT.get_window_ranges(([1:3, 4:6], [1:2, 3:4, 5:6]), CartesianIndex(1, 3)) == (1:3, 5:6)
    @test DT.get_window_ranges(([1:2, 3:4], [1:3, 4:6], [1:5]), CartesianIndex(2, 1, 1)) == (3:4, 1:3, 1:5)
end

# ---------------------------------------------------------------------------- #
#                                 safe_feat                                    #
# ---------------------------------------------------------------------------- #
@testset "safe_feat" begin
    @test DT.safe_feat([1.0, 2.0, 3.0, 4.0, 5.0], maximum) == 5.0
    @test DT.safe_feat([1.0, 2.0, 3.0, 4.0, 5.0], mean) == 3.0

    # missing filtered
    @test DT.safe_feat([1.0, missing, 3.0, missing, 5.0], maximum) == 5.0
    @test DT.safe_feat([1.0, missing, 3.0, missing, 5.0], mean) == 3.0

    # NaN filtered
    @test DT.safe_feat([1.0, NaN, 3.0, NaN, 5.0], maximum) == 5.0
    @test DT.safe_feat([1.0, NaN, 3.0, NaN, 5.0], mean) == 3.0

    # both missing and NaN
    @test DT.safe_feat([1.0, missing, NaN, 4.0, missing, NaN, 7.0], maximum) == 7.0
    @test DT.safe_feat([1.0, missing, NaN, 4.0, missing, NaN, 7.0], mean) == 4.0

    # all invalid → empty vector
    @test_throws Exception DT.safe_feat(Union{Missing,Float64}[missing, missing], maximum)
    @test isnan(DT.safe_feat([NaN, NaN, NaN], mean))

    # single valid element
    @test DT.safe_feat([missing, NaN, 42.0, missing], maximum) == 42.0

    # integers (no NaN possible)
    @test DT.safe_feat(Union{Missing,Int}[1, missing, 3, missing, 5], sum) == 9

    # custom function
    @test DT.safe_feat([1.0, missing, 3.0, NaN, 5.0], x -> length(x)) == 3
    @test DT.safe_feat([1.0, missing, 3.0, NaN, 5.0], std) ≈ std([1.0, 3.0, 5.0])
end

# ---------------------------------------------------------------------------- #
#                        aggregate — curried constructor                        #
# ---------------------------------------------------------------------------- #
@testset "aggregate — curried constructor" begin
    @test aggregate() isa Function
    @test aggregate(win=(movingwindow(winsize=3, winstep=2),), features=(sum, mean)) isa Function
    @test aggregate(features=(maximum,)) isa Function
    @test aggregate(win=(splitwindow(nwindows=3),), features=(mean,)) isa Function
end

# ---------------------------------------------------------------------------- #
#                     aggregate — 1D time series                               #
# ---------------------------------------------------------------------------- #
@testset "aggregate — 1D time series" begin
    tg1d = TreatmentGroup(ds_struct; dims=1)

    @testset "wholewindow + 3 features" begin
        X, idx = _build_Xidx(tg1d)
        Xa, nw = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))(X, idx, Float64)

        @test Xa isa Matrix{Union{Missing,Float64}}
        @test size(Xa, 1) == nrow(df)
        @test size(Xa, 2) == sum(nw) * 3
        @test all(w -> w >= 1, nw)
    end

    @testset "splitwindow(2) + 1 feature" begin
        X, idx = _build_Xidx(tg1d)
        Xa, nw = aggregate(win=(splitwindow(nwindows=2),), features=(mean,))(X, idx, Float64)

        @test all(==(2), nw)
        @test size(Xa, 2) == sum(nw)
    end

    @testset "missing/NaN row propagation" begin
        X, idx = _build_Xidx(["ts1"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(maximum,))(X, idx, Float64)

        @test !ismissing(Xa[1, 1]) && isnan(Xa[1, 1])  # ts1 row 1 = NaN
        @test ismissing(Xa[3, 1])                        # ts1 row 3 = missing
    end

    @testset "correct aggregated values" begin
        X, idx = _build_Xidx(["ts1"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))(X, idx, Float64)

        # row 2: ts1 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        @test Xa[2, 1] == 7.0   # max
        @test Xa[2, 2] == 2.0   # min
        @test Xa[2, 3] ≈ 4.5    # mean
    end

    @testset "internal NaN/missing handled by safe_feat" begin
        # ts3 row 1 = [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2]
        X, idx = _build_Xidx(["ts3"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(maximum, minimum))(X, idx, Float64)
        @test Xa[1, 1] == 4.2
        @test Xa[1, 2] == 1.0

        # ts4 row 1 = [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8]
        X, idx = _build_Xidx(["ts4"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(sum,))(X, idx, Float64)
        @test Xa[1, 1] ≈ 23.2
    end
end

# ---------------------------------------------------------------------------- #
#                     aggregate — 2D images                                    #
# ---------------------------------------------------------------------------- #
@testset "aggregate — 2D images" begin
    tg2d = TreatmentGroup(ds_struct; dims=2)

    @testset "wholewindow + 3 features" begin
        X, idx = _build_Xidx(tg2d)
        Xa, nw = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))(X, idx, Float64)

        @test size(Xa, 1) == nrow(df)
        @test size(Xa, 2) == sum(nw) * 3
    end

    @testset "splitwindow per dimension (2×3)" begin
        X, idx = _build_Xidx(tg2d)
        Xa, nw = aggregate(
            win=(splitwindow(nwindows=2), splitwindow(nwindows=3)),
            features=(mean,)
        )(X, idx, Float64)

        @test all(==(6), nw)  # 2 × 3 = 6 windows per column
        @test size(Xa, 2) == sum(nw)
    end

    @testset "missing/NaN image propagation" begin
        X, idx = _build_Xidx(["img4"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(maximum,))(X, idx, Float64)
        @test ismissing(Xa[3, 1])  # img4 row 3 = missing

        X, idx = _build_Xidx(["img2"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(maximum,))(X, idx, Float64)
        @test !ismissing(Xa[1, 1]) && isnan(Xa[1, 1])  # img2 row 1 = NaN
    end

    @testset "correct values for valid image" begin
        Random.seed!(1)
        expected = rand(Float64, 6, 6)

        X, idx = _build_Xidx(["img1"])
        Xa, _ = aggregate(win=(wholewindow(),), features=(maximum, minimum))(X, idx, Float64)

        @test Xa[1, 1] ≈ maximum(expected)
        @test Xa[1, 2] ≈ minimum(expected)
    end

    @testset "single window func reused across 2D" begin
        X, idx = _build_Xidx(["img1"])
        Xa, nw = aggregate(win=(splitwindow(nwindows=2),), features=(mean,))(X, idx, Float64)

        @test nw[1] == 4       # 2 × 2 = 4
        @test size(Xa, 2) == 4
    end
end

# ---------------------------------------------------------------------------- #
#                     aggregate — output shape & types                         #
# ---------------------------------------------------------------------------- #
@testset "aggregate — output shape & types" begin

    @testset "float_type propagation" begin
        X, idx = _build_Xidx(["ts1"])

        Xa32, _ = aggregate(features=(mean,))(X, idx, Float32)
        @test eltype(Xa32) == Union{Missing,Float32}

        Xa64, _ = aggregate(features=(mean,))(X, idx, Float64)
        @test eltype(Xa64) == Union{Missing,Float64}
    end

    @testset "adaptivewindow same shape as splitwindow" begin
        X, idx = _build_Xidx(["ts1"])
        _, nw1 = aggregate(win=(splitwindow(nwindows=3),), features=(mean,))(X, idx, Float64)
        _, nw2 = aggregate(win=(adaptivewindow(nwindows=3, overlap=0.5),), features=(mean,))(X, idx, Float64)
        @test nw1 == nw2
    end
end

# ---------------------------------------------------------------------------- #
#                       reducesize — curried constructor                        #
# ---------------------------------------------------------------------------- #
@testset "reducesize — curried constructor" begin
    @test reducesize() isa Function
    @test reducesize(win=(splitwindow(nwindows=4),), reducefunc=mean) isa Function
    @test reducesize(reducefunc=sum) isa Function
end

# ---------------------------------------------------------------------------- #
#                       reducesize — 1D time series                            #
# ---------------------------------------------------------------------------- #
@testset "reducesize — 1D time series" begin
    @testset "splitwindow(3) reduces length" begin
        X, idx = _build_Xidx(["ts1"])
        Xr, unused = reducesize(win=(splitwindow(nwindows=3),), reducefunc=mean)(X, idx, Float64)

        @test unused == 0
        @test size(Xr) == (nrow(df), 1)
        @test Xr[2, 1] isa Array{Float64}
        @test length(Xr[2, 1]) == 3  # row 2: [2..7] → 3 windows
    end

    @testset "missing/NaN row propagation" begin
        X, idx = _build_Xidx(["ts1"])
        Xr, _ = reducesize(win=(splitwindow(nwindows=2),), reducefunc=mean)(X, idx, Float64)

        @test ismissing(Xr[3, 1])                              # ts1 row 3 = missing
        @test !ismissing(Xr[1, 1]) && isnan(Xr[1, 1])          # ts1 row 1 = NaN
    end

    @testset "correct reduced values" begin
        X, idx = _build_Xidx(["ts1"])
        Xr, _ = reducesize(win=(splitwindow(nwindows=2),), reducefunc=mean)(X, idx, Float64)

        # row 4: ts1 = [4,5,6,7,8,9] → win1=[4,5,6]→5.0, win2=[7,8,9]→8.0
        @test length(Xr[4, 1]) == 2
        @test Xr[4, 1][1] ≈ 5.0
        @test Xr[4, 1][2] ≈ 8.0
    end

    @testset "internal NaN/missing handled by safe_feat" begin
        # ts3 row 1 = [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2]
        X, idx = _build_Xidx(["ts3"])
        Xr, _ = reducesize(win=(wholewindow(),), reducefunc=maximum)(X, idx, Float64)
        @test Xr[1, 1][1] ≈ 4.2

        # ts4 row 1 = [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8]
        X, idx = _build_Xidx(["ts4"])
        Xr, _ = reducesize(win=(wholewindow(),), reducefunc=sum)(X, idx, Float64)
        @test Xr[1, 1][1] ≈ 23.2
    end
end

# ---------------------------------------------------------------------------- #
#                       reducesize — 2D images                                 #
# ---------------------------------------------------------------------------- #
@testset "reducesize — 2D images" begin
    @testset "splitwindow(2)×splitwindow(3) → 2×3 output" begin
        X, idx = _build_Xidx(["img1"])
        Xr, unused = reducesize(
            win=(splitwindow(nwindows=2), splitwindow(nwindows=3)),
            reducefunc=mean
        )(X, idx, Float64)

        @test unused == 0
        for row in 1:nrow(df)
            @test size(Xr[row, 1]) == (2, 3)
        end
    end

    @testset "missing/NaN image propagation" begin
        X, idx = _build_Xidx(["img4"])
        Xr, _ = reducesize(win=(splitwindow(nwindows=2),), reducefunc=mean)(X, idx, Float64)
        @test ismissing(Xr[3, 1])

        X, idx = _build_Xidx(["img2"])
        Xr, _ = reducesize(win=(splitwindow(nwindows=2),), reducefunc=mean)(X, idx, Float64)
        @test !ismissing(Xr[1, 1]) && isnan(Xr[1, 1])
    end

    @testset "splitwindow(3)×splitwindow(2) → 3×2 output" begin
        X, idx = _build_Xidx(["img3"])
        Xr, _ = reducesize(
            win=(splitwindow(nwindows=3), splitwindow(nwindows=2)),
            reducefunc=mean
        )(X, idx, Float64)

        for row in 1:nrow(df)
            @test size(Xr[row, 1]) == (3, 2)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                  reducesize — multiple columns & types                       #
# ---------------------------------------------------------------------------- #
@testset "reducesize — multiple columns & types" begin
    @testset "all 1D columns" begin
        tg = TreatmentGroup(ds_struct; dims=1)
        X, idx = _build_Xidx(tg)
        Xr, unused = reducesize(win=(splitwindow(nwindows=3),), reducefunc=mean)(X, idx, Float64)

        @test unused == 0
        @test size(Xr) == (nrow(df), length(DT.get_idxs(tg)))
    end

    @testset "float_type propagation" begin
        X, idx = _build_Xidx(["ts1"])

        Xr32, _ = reducesize(win=(splitwindow(nwindows=2),), reducefunc=mean)(X, idx, Float32)
        @test eltype(Xr32) == Union{Missing, Float32, Array{Float32}}

        Xr64, _ = reducesize(win=(splitwindow(nwindows=2),), reducefunc=mean)(X, idx, Float64)
        @test eltype(Xr64) == Union{Missing, Float64, Array{Float64}}
    end
end

# ---------------------------------------------------------------------------- #
#                 return signature consistency                                 #
# ---------------------------------------------------------------------------- #
@testset "return signature consistency" begin
    X, idx = _build_Xidx(["ts1"])

    result_agg = aggregate(features=(mean,))(X, idx, Float64)
    @test result_agg isa Tuple{Matrix, Vector{Int}}

    result_red = reducesize(reducefunc=mean)(X, idx, Float64)
    @test result_red isa Tuple
    @test result_red[1] isa Array
    @test result_red[2] == 0
end