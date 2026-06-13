using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using Statistics: mean

# ---------------------------------------------------------------------------- #
#                                 test data                                    #
# ---------------------------------------------------------------------------- #
function make_scalar_col()
    [1.0, NaN, missing, 4.0, 5.0]
end

function make_ts_col()
    [
        [1.0, NaN, 3.0, 4.0],
        missing,
        [2.0, 3.0, NaN, 5.0],
        [1.0, 2.0, 3.0, 4.0],
        [NaN, 2.0, 3.0, missing],
    ]
end

function make_matrix(col1=make_scalar_col(), col2=make_scalar_col())
    hcat(col1, col2)
end

function make_matrix(nrows=5, ncols=3)
    data = Matrix{Union{Missing,Float64}}(undef, nrows, ncols)
    for j in 1:ncols
        col = rand(Float64, nrows)
        # sprinkle some NaN and missing
        col[rand(1:nrows)] = NaN
        data[:, j] = col
        data[rand(1:nrows), j] = missing
    end
    return data
end

# ---------------------------------------------------------------------------- #
#                            AbstractArray dispatch                            #
# ---------------------------------------------------------------------------- #
@testset "_impute AbstractMatrix" begin
    @testset "LOCF + NOCB chain" begin
        data = make_matrix()
        result = DT._impute(data, (LOCF(), NOCB()))
        @test !any(ismissing, result)
    end

    @testset "Interpolate" begin
        data = Union{Missing,Float64}[
            1.0  1.0  1.0;
            NaN  2.0  2.0;
            3.0  NaN  3.0;
            4.0  4.0  missing;
            5.0  5.0  5.0
        ]
        result = DT._impute(data, (Interpolate(),))
        @test !any(ismissing, result)
        @test result[2, 1] ≈ 2.0
        @test result[3, 2] ≈ 3.0
        @test result[4, 3] ≈ 4.0
    end

    @testset "no missing values unchanged type" begin
        data = Float64[
            1.0  2.0  3.0;
            4.0  5.0  6.0;
            7.0  8.0  9.0
        ]
        result = DT._impute(data, (LOCF(),))
        @test eltype(result) <: AbstractFloat
    end
end

# ---------------------------------------------------------------------------- #
#                       AbstractMatrix{Float} dispatch                         #
# ---------------------------------------------------------------------------- #
@testset "_impute AbstractMatrix{Float}" begin
    @testset "LOCF on float matrix" begin
        data = Union{Missing,Float64}[
            1.0  2.0;
            NaN  missing;
            3.0  4.0;
            4.0  5.0;
            5.0  6.0
        ]
        result = DT._impute(data, (LOCF(),))
        @test !any(ismissing, result)
    end

    @testset "SVD imputation" begin
        data = Union{Missing,Float64}[
            1.0  2.0  3.0;
            NaN  5.0  6.0;
            7.0  missing  9.0;
            1.0  2.0  3.0;
            4.0  5.0  6.0
        ]
        result = DT._impute(data, (SVD(),))
        @test !any(ismissing, result)
    end

    @testset "Substitute with mean" begin
        data = Union{Missing,Float64}[
            1.0  2.0;
            NaN  missing;
            3.0  4.0;
            4.0  5.0;
            5.0  6.0
        ]
        result = DT._impute(data, (DT.Substitute(statistic=mean),))
        @test !any(ismissing, result)
    end
end

# ---------------------------------------------------------------------------- #
#                     scalar/array-valued float dispatch                       #
# ---------------------------------------------------------------------------- #
@testset "_impute scalar/array element" begin
    @testset "missing scalar returned unchanged" begin
        result = DT._impute(missing, (LOCF(),))
        @test ismissing(result)
    end

    @testset "float scalar returned unchanged" begin
        result = DT._impute(3.14, (LOCF(),))
        @test result == 3.14
    end

    @testset "array element with NaN" begin
        data = Union{Missing,Float64}[
            1.0  NaN  3.0  4.0;
            1.0  2.0  3.0  4.0
        ]
        result = DT._impute(data, (Interpolate(), LOCF(), NOCB()))
        @test !any(ismissing, result)
        @test result[1, 2] ≈ 2.0
    end

    @testset "array element with missing" begin
        data = Union{Missing,Float64}[
            1.0  2.0  missing  4.0;
            1.0 2.0 3.0 4.0
        ]
        result = DT._impute(data, (Interpolate(), LOCF(), NOCB()))
        @test !any(ismissing, result)
        @test result[1, 3] == 3.0
    end

    @testset "array element no missing unchanged type" begin
        data = Union{Missing,Float64}[
            1.0  2.0  3.0  4.0;
            1.0 2.0 3.0 4.0
        ]
        result = DT._impute(data, (LOCF(),))
        @test eltype(result) <: AbstractFloat
    end
end

# ---------------------------------------------------------------------------- #
#                        integration with load_dataset                         #
# ---------------------------------------------------------------------------- #
@testset "_impute via load_dataset" begin
    Random.seed!(42)

    df = DataFrame(
        V1=Union{Missing,Float64}[NaN, missing, 3.0, 4.0, 5.6],
        V2=Union{Missing,Float64}[2.5, missing, 4.5, 5.5, NaN],
        V3=Union{Missing,Float64}[3.2, 4.2, 5.2, missing, 2.4],
        ts1=[
            missing,
            collect(2.0:7.0),
            missing,
            collect(4.0:9.0),
            collect(5.0:10.0)
        ],
        ts2=[
            Union{Missing,Float64}[1.0, NaN, 3.0, 4.0],
            Union{Missing,Float64}[2.0, 3.0, missing, 5.0],
            collect(Float64, 1:4),
            collect(Float64, 2:5),
            collect(Float64, 3:6),
        ],
    )
    target = ["a", "b", "a", "b", "a"]

    @testset "scalar LOCF" begin
        dt = load_dataset(
            df, target,
            TreatmentGroup(dims=0, impute=(LOCF(),),
                datatype=:continuous)
        )
        data = get_continuous(dt)
        @test !isnothing(data)
    end

    @testset "scalar LOCF + NOCB" begin
        dt = load_dataset(
            df, target,
            TreatmentGroup(dims=0, impute=(LOCF(), NOCB()),
                datatype=:continuous)
        )
        data = get_continuous(dt)
        @test !isnothing(data)
    end

    @testset "scalar Interpolate + LOCF + NOCB" begin
        dt = load_dataset(
            df, target,
            TreatmentGroup(dims=0, impute=(Interpolate(), LOCF(), NOCB()),
                datatype=:continuous)
        )
        data = get_continuous(dt)
        @test !isnothing(data)
    end

    @testset "timeseries SVD" begin
        dt = load_dataset(
            df, target,
            TreatmentGroup(dims=1, impute=(SVD(),))
        )
        data = get_multidim(dt)
        @test !isnothing(data)
    end

    @testset "timeseries Substitute" begin
        dt = load_dataset(
            df, target,
            TreatmentGroup(
                dims=1,
                impute=(DT.Substitute(statistic=mean),),
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=2),)
                )
            )
        )
        data = get_multidim(dt)
        @test !isnothing(data)
    end
end