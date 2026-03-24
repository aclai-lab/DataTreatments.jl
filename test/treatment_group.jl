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
vnames = names(df)

# @test_nowarn @inferred TreatmentGroup(datastruct, vnames; dims=1, groupby=:vname)
@test_nowarn InteractiveUtils.@code_warntype TreatmentGroup(datastruct, vnames; dims=1, groupby=:vname)

@testset "TreatmentGroup basic construction and filtering" begin
    tg = TreatmentGroup(datastruct, vnames)
    @test tg isa TreatmentGroup
    @test length(tg.ids) == ncol(df)

    tg0 = TreatmentGroup(datastruct, vnames; dims=0)
    tg1 = TreatmentGroup(datastruct, vnames; dims=1)
    tg2 = TreatmentGroup(datastruct, vnames; dims=2)
    @test length(tg0.ids) == 10
    @test length(tg1.ids) == 4
    @test length(tg2.ids) == 4

    tg_v = TreatmentGroup(datastruct, vnames; name_expr=r"^V")
    @test all(startswith(n, "V") for n in tg_v.vnames)

    tg_img = TreatmentGroup(datastruct, vnames; name_expr=r"^img")
    @test all(startswith(n, "img") for n in tg_img.vnames)

    tg_vec = TreatmentGroup(datastruct, vnames; name_expr=["V1", "V3"])
    @test tg_vec.vnames == ["V1", "V3"]

    tg_fun = TreatmentGroup(datastruct, vnames; name_expr=n -> endswith(n, "col"))
    @test all(endswith(n, "col") for n in tg_fun.vnames)

    tg_empty = TreatmentGroup(datastruct, vnames; name_expr=r"^ZZZZZ")
    @test length(tg_empty.ids) == 0
end

@testset "TreatmentGroup type parameter and aggregation" begin
    tg_v = TreatmentGroup(datastruct, vnames; dims=0, name_expr=r"^V")
    @test tg_v.datatype == Float64
    tg_int = TreatmentGroup(datastruct, vnames; name_expr=["int_col"])
    @test tg_int.datatype == Int64
    tg_any = TreatmentGroup(datastruct, vnames; name_expr=r"^ZZZZZ")
    @test tg_any.datatype == Any
    tg_agg = TreatmentGroup(datastruct, vnames; dims=1)
    @test DT.get_aggrfunc(tg_agg) isa Base.Callable
end

@testset "TreatmentGroup groupby and grouped" begin
    tg = TreatmentGroup(datastruct, vnames; dims=1, groupby=:vname)
    @test DT.has_groupby(tg)
    @test DT.get_groupby(tg) == (:vname,)
    tg2 = TreatmentGroup(datastruct, vnames; dims=1, groupby=(:vname, :feature))
    @test DT.get_groupby(tg2) == (:vname, :feature)
    tg3 = TreatmentGroup(datastruct, vnames)
    @test !DT.has_groupby(tg3)
end
