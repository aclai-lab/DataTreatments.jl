using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays
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

ds_struct = DT.DatasetStructure(df)

# ============================================================================ #
#                     TreatmentGroup - DataFrame constructor                   #
# ============================================================================ #
@testset "TreatmentGroup - DataFrame constructor" begin
    @testset "Default kwargs (all columns)" begin
        tg = TreatmentGroup(df)
        @test tg isa TreatmentGroup
        @test length(tg) == ncol(df)
        @test DT.get_dims(tg) == -1
    end

    @testset "dims=0 selects scalar columns" begin
        tg = TreatmentGroup(df; dims=0)
        @test DT.get_dims(tg) == 0
        # scalar columns: str_col, sym_col, cat_col, uint_col, int_col, V1..V5 = 10
        @test length(tg) == 10
        @test all(n -> n in DT.get_vnames(tg),
            ["str_col", "sym_col", "cat_col", "uint_col", "int_col",
             "V1", "V2", "V3", "V4", "V5"])
    end

    @testset "dims=1 selects 1D columns" begin
        tg = TreatmentGroup(df; dims=1)
        @test DT.get_dims(tg) == 1
        # 1D columns: ts1, ts2, ts3, ts4
        @test length(tg) == 4
        @test all(n -> n in DT.get_vnames(tg), ["ts1", "ts2", "ts3", "ts4"])
    end

    @testset "dims=2 selects 2D columns" begin
        tg = TreatmentGroup(df; dims=2)
        @test DT.get_dims(tg) == 2
        # 2D columns: img1, img2, img3, img4
        @test length(tg) == 4
        @test all(n -> n in DT.get_vnames(tg), ["img1", "img2", "img3", "img4"])
    end
end

# ============================================================================ #
#                  TreatmentGroup - DatasetStructure constructor                #
# ============================================================================ #
@testset "TreatmentGroup - DatasetStructure constructor" begin
    @testset "Basic construction" begin
        tg = TreatmentGroup(ds_struct)
        @test tg isa TreatmentGroup
        @test length(tg) == ncol(df)
    end

    @testset "dims filter" begin
        tg0 = TreatmentGroup(ds_struct; dims=0)
        tg1 = TreatmentGroup(ds_struct; dims=1)
        tg2 = TreatmentGroup(ds_struct; dims=2)
        @test length(tg0) + length(tg1) + length(tg2) == ncol(df)
    end
end

# ============================================================================ #
#                    TreatmentGroup - name_expr filtering                       #
# ============================================================================ #
@testset "TreatmentGroup - name_expr filtering" begin
    @testset "Regex filter" begin
        tg = TreatmentGroup(df; name_expr=r"^V")
        @test length(tg) == 5
        @test all(n -> startswith(n, "V"), DT.get_vnames(tg))
    end

    @testset "Regex filter - ts columns" begin
        tg = TreatmentGroup(df; name_expr=r"^ts")
        @test length(tg) == 4
        @test all(n -> startswith(n, "ts"), DT.get_vnames(tg))
    end

    @testset "Regex filter - img columns" begin
        tg = TreatmentGroup(df; name_expr=r"^img")
        @test length(tg) == 4
        @test all(n -> startswith(n, "img"), DT.get_vnames(tg))
    end

    @testset "Vector{String} filter" begin
        tg = TreatmentGroup(df; name_expr=["V1", "V3", "V5"])
        @test length(tg) == 3
        @test DT.get_vnames(tg) == ["V1", "V3", "V5"]
    end

    @testset "Function filter" begin
        tg = TreatmentGroup(df; name_expr=n -> endswith(n, "col"))
        @test length(tg) == 5
        @test all(n -> endswith(n, "col"), DT.get_vnames(tg))
    end

    @testset "Regex matching nothing" begin
        tg = TreatmentGroup(df; name_expr=r"^ZZZZZ")
        @test length(tg) == 0
        @test isempty(DT.get_idxs(tg))
    end

    @testset "Vector{String} with nonexistent names" begin
        tg = TreatmentGroup(df; name_expr=["nonexistent1", "nonexistent2"])
        @test length(tg) == 0
    end

    @testset "Combined dims + name_expr" begin
        tg = TreatmentGroup(df; dims=0, name_expr=r"^V")
        @test length(tg) == 5
        @test all(n -> startswith(n, "V"), DT.get_vnames(tg))
    end

    @testset "Combined dims + name_expr - no overlap" begin
        # V columns are dims=0, so dims=1 + V regex → empty
        tg = TreatmentGroup(df; dims=1, name_expr=r"^V")
        @test length(tg) == 0
    end
end

# ============================================================================ #
#                    TreatmentGroup - datatype filtering                        #
# ============================================================================ #
@testset "TreatmentGroup - datatype filtering" begin
    @testset "Float64 filter" begin
        tg = TreatmentGroup(df; datatype=Float64)
        @test length(tg) > 0
        @test all(n -> n in ["V1", "V2", "V3", "V4", "V5",
                             "ts1", "ts2", "ts3", "ts4",
                             "img1", "img2", "img3", "img4"],
                  DT.get_vnames(tg))
    end

    @testset "Int filter" begin
        tg = TreatmentGroup(df; datatype=Int)
        @test length(tg) >= 1
        @test "int_col" in DT.get_vnames(tg)
    end

    @testset "UInt32 filter" begin
        tg = TreatmentGroup(df; datatype=UInt32)
        @test length(tg) >= 1
        @test "uint_col" in DT.get_vnames(tg)
    end

    @testset "Combined dims + datatype" begin
        tg = TreatmentGroup(df; dims=0, datatype=Float64)
        @test length(tg) == 5
        @test all(n -> n in ["V1", "V2", "V3", "V4", "V5"], DT.get_vnames(tg))
    end

    @testset "Nonexistent datatype" begin
        tg = TreatmentGroup(df; datatype=Complex{Float64})
        @test length(tg) == 0
    end
end

# ============================================================================ #
#                    TreatmentGroup - type parameter T                         #
# ============================================================================ #
@testset "TreatmentGroup - type parameter T" begin
    @testset "Float64 columns" begin
        tg = TreatmentGroup(df; dims=0, name_expr=r"^V")
        @test tg isa TreatmentGroup{Float64}
    end

    @testset "Int columns" begin
        tg = TreatmentGroup(df; name_expr=["int_col"])
        @test tg isa TreatmentGroup{Int64}
    end

    @testset "Mixed types → typejoin" begin
        tg = TreatmentGroup(df; dims=0)
        # Mixing String, Symbol, CategoricalValue, UInt32, Int, Float64 → broader type
        @test tg isa TreatmentGroup{T} where T
    end

    @testset "Empty selection → Any" begin
        tg = TreatmentGroup(df; name_expr=r"^ZZZZZ")
        @test tg isa TreatmentGroup{Any}
    end
end

# ============================================================================ #
#                    TreatmentGroup - aggrfunc & groupby                        #
# ============================================================================ #
@testset "TreatmentGroup - aggrfunc and groupby" begin
    @testset "Default aggrfunc" begin
        tg = TreatmentGroup(df; dims=1)
        @test DT.get_aggrfunc(tg) isa Base.Callable
    end

    @testset "Custom aggrfunc" begin
        custom_agg = aggregate(win=(wholewindow(),), features=(sum,))
        tg = TreatmentGroup(df; dims=1, aggrfunc=custom_agg)
        @test DT.get_aggrfunc(tg) === custom_agg
    end

    @testset "Default groupby is nothing" begin
        tg = TreatmentGroup(df; dims=1)
        @test DT.get_groupby(tg) === nothing
        @test DT.has_groupby(tg) == false
    end

    @testset "Symbol groupby is wrapped in tuple" begin
        tg = TreatmentGroup(df; dims=1, groupby=:vname)
        @test DT.get_groupby(tg) == (:vname,)
        @test DT.has_groupby(tg) == true
    end

    @testset "Tuple groupby" begin
        tg = TreatmentGroup(df; dims=1, groupby=(:vname, :feature))
        @test DT.get_groupby(tg) == (:vname, :feature)
        @test DT.has_groupby(tg) == true
    end
end

# ============================================================================ #
#                    TreatmentGroup - curried constructor                       #
# ============================================================================ #
@testset "TreatmentGroup - curried constructor" begin
    @testset "Returns a callable" begin
        tg_fn = TreatmentGroup(dims=0)
        @test tg_fn isa Function
    end

    @testset "Callable produces TreatmentGroup" begin
        tg_fn = TreatmentGroup(dims=0)
        tg = tg_fn(ds_struct)
        @test tg isa TreatmentGroup
        @test DT.get_dims(tg) == 0
        @test length(tg) == 10
    end

    @testset "Curried with name_expr" begin
        tg_fn = TreatmentGroup(dims=0, name_expr=r"^V")
        tg = tg_fn(ds_struct)
        @test length(tg) == 5
        @test all(n -> startswith(n, "V"), DT.get_vnames(tg))
    end

    @testset "Curried with all kwargs" begin
        tg_fn = TreatmentGroup(dims=1, name_expr=["ts1", "ts2"], groupby=:vname)
        tg = tg_fn(ds_struct)
        @test length(tg) == 2
        @test DT.has_groupby(tg) == true
    end
end

# ============================================================================ #
#                    TreatmentGroup - Matrix constructor                        #
# ============================================================================ #
@testset "TreatmentGroup - Matrix constructor" begin
    mat = Matrix{Any}(hcat(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        ["a", "b", "c"]
    ))
    vnames = ["col1", "col2", "col3"]

    @testset "Basic construction" begin
        tg = TreatmentGroup(mat, vnames)
        @test tg isa TreatmentGroup
        @test length(tg) == 3
    end

    @testset "With name_expr" begin
        tg = TreatmentGroup(mat, vnames; name_expr=["col1", "col2"])
        @test length(tg) == 2
        @test DT.get_vnames(tg) == ["col1", "col2"]
    end
end

# ============================================================================ #
#                         Base methods                                         #
# ============================================================================ #
@testset "TreatmentGroup - Base methods" begin
    tg = TreatmentGroup(df; dims=0, name_expr=r"^V")

    @testset "Base.length" begin
        @test length(tg) == 5
    end

    @testset "Base.iterate" begin
        collected = collect(tg)
        @test length(collected) == 5
        @test all(i -> i isa Int, collected)
        @test collected == DT.get_idxs(tg)
    end

    @testset "Base.iterate - empty" begin
        tg_empty = TreatmentGroup(df; name_expr=r"^ZZZZZ")
        collected = collect(tg_empty)
        @test isempty(collected)
    end

    @testset "Base.eachindex" begin
        ei = eachindex(tg)
        @test ei == eachindex(DT.get_idxs(tg))
        @test length(ei) == 5
    end
end

# ============================================================================ #
#                         Getter methods                                       #
# ============================================================================ #
@testset "TreatmentGroup - Getter methods" begin
    tg = TreatmentGroup(df; dims=0, name_expr=["V1", "V2", "V3"])

    @testset "get_idxs()" begin
        idxs = DT.get_idxs(tg)
        @test idxs isa Vector{Int}
        @test length(idxs) == 3
    end

    @testset "get_idxs(i)" begin
        idx = DT.get_idxs(tg, 1)
        @test idx isa Int
        @test idx == DT.get_idxs(tg)[1]
    end

    @testset "get_dims()" begin
        @test DT.get_dims(tg) == 0
    end

    @testset "get_vnames()" begin
        vnames = DT.get_vnames(tg)
        @test vnames isa Vector{String}
        @test vnames == ["V1", "V2", "V3"]
    end

    @testset "get_vnames(i)" begin
        @test DT.get_vnames(tg, 1) == "V1"
        @test DT.get_vnames(tg, 2) == "V2"
        @test DT.get_vnames(tg, 3) == "V3"
    end

    @testset "get_vnames(idxs)" begin
        result = DT.get_vnames(tg, [1, 3])
        @test result == ["V1", "V3"]
    end

    @testset "get_aggrfunc()" begin
        @test DT.get_aggrfunc(tg) isa Base.Callable
    end

    @testset "get_groupby()" begin
        @test DT.get_groupby(tg) === nothing
    end

    @testset "has_groupby() - false" begin
        @test DT.has_groupby(tg) == false
    end

    @testset "has_groupby() - true" begin
        tg_gb = TreatmentGroup(df; dims=1, groupby=:vname)
        @test DT.has_groupby(tg_gb) == true
    end
end

# ============================================================================ #
#                    get_idxs(::Vector{TreatmentGroup})                        #
# ============================================================================ #
@testset "TreatmentGroup - get_idxs overlap resolution" begin
    @testset "No overlap" begin
        tg1 = TreatmentGroup(df; dims=0, name_expr=r"^V")
        tg2 = TreatmentGroup(df; dims=1)
        result = DT.get_idxs([tg1, tg2])
        @test length(result) == 2
        @test !isempty(result[1])
        @test !isempty(result[2])
        @test isempty(intersect(result[1], result[2]))
    end

    @testset "Full overlap - later group wins" begin
        tg1 = TreatmentGroup(df; dims=0, name_expr=r"^V")
        tg2 = TreatmentGroup(df; dims=0, name_expr=r"^V")
        result = DT.get_idxs([tg1, tg2])
        @test length(result) == 2
        # tg2 (later) should keep all indices, tg1 should be empty
        @test isempty(result[1])
        @test length(result[2]) == 5
    end

    @testset "Partial overlap" begin
        tg1 = TreatmentGroup(df; dims=0, name_expr=["V1", "V2", "V3"])
        tg2 = TreatmentGroup(df; dims=0, name_expr=["V3", "V4", "V5"])
        result = DT.get_idxs([tg1, tg2])
        @test length(result) == 2
        # V3 should be in tg2 (later wins), not in tg1
        @test isempty(intersect(result[1], result[2]))
        @test length(result[1]) == 2  # V1, V2
        @test length(result[2]) == 3  # V3, V4, V5
    end

    @testset "Three groups with cascading overlap" begin
        tg1 = TreatmentGroup(df; dims=0)  # all scalar
        tg2 = TreatmentGroup(df; dims=0, name_expr=r"^V")  # V1..V5
        tg3 = TreatmentGroup(df; dims=0, name_expr=["V1"])  # just V1
        result = DT.get_idxs([tg1, tg2, tg3])
        @test length(result) == 3
        # V1 should only be in tg3
        idxs_V1 = DT.get_idxs(TreatmentGroup(df; name_expr=["V1"]))[1]
        @test idxs_V1 ∉ result[1]
        @test idxs_V1 ∉ result[2]
        @test idxs_V1 ∈ result[3]
    end

    @testset "Warning on fully consumed group" begin
        tg1 = TreatmentGroup(df; dims=0, name_expr=["V1"])
        tg2 = TreatmentGroup(df; dims=0, name_expr=["V1"])
        @test_logs (:warn, r"no columns") DT.get_idxs([tg1, tg2])
    end
end

# ============================================================================ #
#                         Base.show methods                                    #
# ============================================================================ #
@testset "TreatmentGroup - Base.show" begin
    @testset "One-line show - dims=all" begin
        tg = TreatmentGroup(df)
        str = sprint(show, tg)
        @test occursin("TreatmentGroup", str)
        @test occursin("dims=all", str)
        @test occursin("$(ncol(df)) cols", str)
    end

    @testset "One-line show - dims=0" begin
        tg = TreatmentGroup(df; dims=0)
        str = sprint(show, tg)
        @test occursin("dims=0", str)
    end

    @testset "One-line show - dims=1" begin
        tg = TreatmentGroup(df; dims=1)
        str = sprint(show, tg)
        @test occursin("dims=1", str)
    end

    @testset "One-line show - dims=2" begin
        tg = TreatmentGroup(df; dims=2)
        str = sprint(show, tg)
        @test occursin("dims=2", str)
    end

    @testset "One-line show - empty" begin
        tg = TreatmentGroup(df; name_expr=r"^ZZZZZ")
        str = sprint(show, tg)
        @test occursin("0 cols", str)
    end

    @testset "Multi-line show - dims=0 (scalar)" begin
        tg = TreatmentGroup(df; dims=0, name_expr=r"^V")
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("TreatmentGroup", str)
        @test occursin("5 columns selected", str)
        @test occursin("dims filter: 0", str)
        @test occursin("selected indices", str)
        # dims=0 should NOT show aggregation function or groupby
        @test !occursin("aggregation function", str)
        @test !occursin("groupby", str)
    end

    @testset "Multi-line show - dims=1 (multidim)" begin
        tg = TreatmentGroup(df; dims=1)
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("TreatmentGroup", str)
        @test occursin("4 columns selected", str)
        @test occursin("dims filter: 1", str)
        @test occursin("selected indices", str)
        @test occursin("aggregation function", str)
        @test occursin("groupby", str)
    end

    @testset "Multi-line show - dims=2 (multidim)" begin
        tg = TreatmentGroup(df; dims=2)
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("dims filter: 2", str)
        @test occursin("aggregation function", str)
    end

    @testset "Multi-line show - dims=-1 (all)" begin
        tg = TreatmentGroup(df)
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("dims filter: all", str)
        # dims=-1 != 0, so should show aggrfunc/groupby
        @test occursin("aggregation function", str)
        @test occursin("groupby", str)
    end

    @testset "Multi-line show - with groupby" begin
        tg = TreatmentGroup(df; dims=1, groupby=(:vname, :feature))
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("groupby", str)
        @test occursin("vname", str)
        @test occursin("feature", str)
    end

    @testset "Multi-line show - type parameter displayed" begin
        tg = TreatmentGroup(df; dims=0, name_expr=r"^V")
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("Float64", str)
    end

    @testset "Multi-line show - empty group" begin
        tg = TreatmentGroup(df; name_expr=r"^ZZZZZ")
        str = sprint(show, MIME("text/plain"), tg)
        @test occursin("0 columns selected", str)
    end
end

# ============================================================================ #
#                    TreatmentGroup - edge cases                               #
# ============================================================================ #
@testset "TreatmentGroup - edge cases" begin
    @testset "All filters combined - matching" begin
        tg = TreatmentGroup(df; dims=0, name_expr=r"^V", datatype=Float64)
        @test length(tg) == 5
    end

    @testset "All filters combined - no match" begin
        tg = TreatmentGroup(df; dims=1, name_expr=r"^V", datatype=Float64)
        @test length(tg) == 0
    end

    @testset "Single column selection" begin
        tg = TreatmentGroup(df; name_expr=["V1"])
        @test length(tg) == 1
        @test DT.get_vnames(tg) == ["V1"]
    end

    @testset "Iteration protocol consistency" begin
        tg = TreatmentGroup(df; dims=0, name_expr=r"^V")
        # iterate should yield same as get_idxs
        iterated = Int[]
        for idx in tg
            push!(iterated, idx)
        end
        @test iterated == DT.get_idxs(tg)
    end

    @testset "eachindex matches length" begin
        tg = TreatmentGroup(df; dims=1)
        @test length(eachindex(tg)) == length(tg)
    end
end