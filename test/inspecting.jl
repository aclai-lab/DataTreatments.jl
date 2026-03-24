using Test
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
        str_col=[missing, "blue", "green", "red", "blue"],
        sym_col=[:circle, :square, :triangle, :square, missing],
        img4=[i == 3 ? missing : create_image(i + 30) for i in 1:5],
        int_col=Int[10, 20, 30, 40, 50],
        V1=[NaN, missing, 3.0, 4.0, 5.6],
        V2=[2.5, missing, 4.5, 5.5, NaN],
        ts1=[NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
        V4=[4.1, NaN, NaN, 7.1, 5.5],
        V5=[5.0, 6.0, 7.0, 8.0, 1.8],
        ts2=[
            collect(2.0:0.5:5.5),
            collect(1.0:0.5:4.5),
            collect(3.0:0.5:6.5),
            collect(4.0:0.5:7.5),
            NaN
        ],
        ts3=[
            [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2],
            NaN, NaN, missing,
            [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]
        ],
        V3=[3.2, 4.2, 5.2, missing, 2.4],
        ts4=[
            [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8],
            missing,
            [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8],
            [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]
        ],
        img1=[create_image(i) for i in 1:5],
        cat_col=categorical(["small", "medium", missing, "small", "large"]),
        uint_col=UInt32[1, 2, 3, 4, 5],
        img2=[i == 1 ? NaN : create_image(i + 10) for i in 1:5],
        img3=[create_image(i + 20) for i in 1:5],
    )
end

df = build_test_df()

inspect = DT._inspecting(Matrix(df))

@test_nowarn @inferred DT._inspecting(Matrix(df))
@test_nowarn InteractiveUtils.@code_warntype DT._inspecting(Matrix(df))

@testset "_inspecting returns expected results" begin
    # Check number of columns
    @test length(inspect.id) == ncol(df)
    @test inspect.id == collect(1:ncol(df))

    # Check that types are inferred correctly for some columns
    @test inspect.datatype[df |> names .== "int_col"][1] == Int
    @test inspect.datatype[df |> names .== "str_col"][1] == String
    @test inspect.datatype[df |> names .== "img1"][1] <: Array

    # Check that dims is 0 for scalar columns and >0 for arrays
    @test inspect.dims[df |> names .== "int_col"][1] == 0
    @test inspect.dims[df |> names .== "img1"][1] == 2

    # Check missing and nan indices for a known column
    v1_idx = findfirst(==("V1"), names(df))
    @test inspect.missingidxs[v1_idx] == [2]
    @test inspect.nanidxs[v1_idx] == [1]

    # Check valid indices for a column with no missing/nan
    int_idx = findfirst(==("int_col"), names(df))
    @test inspect.valididxs[int_idx] == [1,2,3,4,5]

    # Check hasmissing and hasnans for a multidim column
    ts4_idx = findfirst(==("ts4"), names(df))
    @test inspect.hasmissing[ts4_idx] == [1,3,4,5]
    @test inspect.hasnans[ts4_idx] == [3,4,5]
end

@testset "_discrete_encode works as expected" begin
    target = [:circle, :square, :triangle, :square, missing]
    encoded, levels = DT._discrete_encode(target)

    @test length(encoded) == length(target)
    @test all(ismissing(target[i]) ? ismissing(encoded[i]) : isa(encoded[i], Int) for i in eachindex(target))

    @test encoded[2] == encoded[4]

    @test encoded[1] != encoded[2]
    @test encoded[3] != encoded[1]

    @test length(levels) == 3
end
