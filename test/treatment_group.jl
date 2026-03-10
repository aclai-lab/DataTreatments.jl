using Test
using DataFrames
using DataTreatments

# Setup test data
df = DataFrame(
    V1 = [1.0, 2.0, 3.0],
    V2 = [4.0, 5.0, 6.0],
    V3 = [7.0, 8.0, 9.0],
    X1 = [10, 20, 30],
    X2 = [40, 50, 60],
    name_col = ["a", "b", "c"]
)

@testset "TreatmentGroup" begin

    @testset "Construction" begin
        @testset "Basic construction with no filters" begin
            tg = TreatmentGroup(df)
            @test length(tg) == ncol(df)
            @test get_dims(tg) == -1
        end

        @testset "Filter by dims" begin
            tg = TreatmentGroup(df; dims=0)
            @test length(tg) > 0
            @test get_dims(tg) == 0
        end

        @testset "Filter by regex pattern" begin
            tg = TreatmentGroup(df; name_expr=r"^V")
            @test length(tg) == 3
            @test all(name -> startswith(name, "V"), get_vnames(tg))
        end

        @testset "Filter by function predicate" begin
            tg = TreatmentGroup(df; name_expr=name -> startswith(name, "X"))
            @test length(tg) == 2
            @test all(name -> startswith(name, "X"), get_vnames(tg))
        end

        @testset "Filter by vector of names" begin
            tg = TreatmentGroup(df; name_expr=["V1", "X1"])
            @test length(tg) == 2
            @test get_vnames(tg) == ["V1", "X1"]
        end

        @testset "Filter by datatype" begin
            tg = TreatmentGroup(df; datatype=Float64)
            @test all(name -> startswith(name, "V"), get_vnames(tg))
        end

        @testset "Combined filters" begin
            tg = TreatmentGroup(df; name_expr=r"^V", datatype=Float64)
            @test length(tg) == 3
            @test all(name -> startswith(name, "V"), get_vnames(tg))
        end

        @testset "Curried constructor" begin
            curried = TreatmentGroup(dims=0)
            @test curried isa Function
            tg = curried(df)
            @test get_dims(tg) == 0
        end
    end

    @testset "Base Methods" begin
        tg = TreatmentGroup(df; name_expr=r"^V")

        @testset "length" begin
            @test length(tg) == 3
        end

        @testset "iterate" begin
            idxs = collect(tg)
            @test length(idxs) == 3
            @test all(idx -> idx isa Int, idxs)
        end

        @testset "eachindex" begin
            ei = eachindex(tg)
            @test length(ei) == length(tg)
        end
    end

    @testset "Getter Methods" begin
        tg = TreatmentGroup(df; name_expr=r"^V")

        @testset "get_idxs - all indices" begin
            idxs = get_idxs(tg)
            @test idxs isa Vector{Int}
            @test length(idxs) == 3
        end

        @testset "get_idxs - single index" begin
            idx = get_idxs(tg, 1)
            @test idx isa Int
            @test idx == get_idxs(tg)[1]
        end

        @testset "get_dims" begin
            @test get_dims(tg) == -1
        end

        @testset "get_vnames - all names" begin
            @test get_vnames(tg) == ["V1", "V2", "V3"]
        end

        @testset "get_vnames - single name" begin
            @test get_vnames(tg, 1) == "V1"
        end

        @testset "get_vnames - subset by indices" begin
            @test get_vnames(tg, [1, 3]) == ["V1", "V3"]
        end

        @testset "get_aggrfunc" begin
            @test get_aggrfunc(tg) isa Function
        end

        @testset "get_groupby" begin
            @test get_groupby(tg) == (:vname,)
        end

        @testset "custom groupby" begin
            tg_custom = TreatmentGroup(df; groupby=(:vname, :window))
            @test get_groupby(tg_custom) == (:vname, :window)
        end
    end

    @testset "Multiple Treatment Groups" begin
        @testset "get_idxs with vector of TreatmentGroups" begin
            tg1 = TreatmentGroup(df; name_expr=r"^V")
            tg2 = TreatmentGroup(df; name_expr=r"^X")
            idxs_vec = get_idxs([tg1, tg2])
            @test length(idxs_vec) == 2
            @test all(idx -> idx isa Vector{Int}, idxs_vec)
        end

        @testset "Disjoint partitioning - no overlap" begin
            tg1 = TreatmentGroup(df; name_expr=r"^V")
            tg2 = TreatmentGroup(df; name_expr=r"^X")
            idxs_vec = get_idxs([tg1, tg2])
            combined = union(idxs_vec...)
            @test length(combined) == length(tg1) + length(tg2)
        end

        @testset "Disjoint partitioning - with overlap, later group takes precedence" begin
            tg1 = TreatmentGroup(df; name_expr=r"^V")
            tg2 = TreatmentGroup(df; name_expr=r"V|X")
            idxs_vec = get_idxs([tg1, tg2])
            @test isdisjoint(idxs_vec[1], idxs_vec[2])
            @test get_idxs(tg2) == union(idxs_vec[2], idxs_vec[1])
        end

        @testset "Three groups with cascading overlap" begin
            tg1 = TreatmentGroup(df; name_expr=r"^V")
            tg2 = TreatmentGroup(df; name_expr=r"V|X")
            tg3 = TreatmentGroup(df)
            idxs_vec = get_idxs([tg1, tg2, tg3])
            for i in 1:3, j in i+1:3
                @test isdisjoint(idxs_vec[i], idxs_vec[j])
            end
            @test union(idxs_vec...) == get_idxs(tg3)
        end
    end

    @testset "Edge Cases" begin
        @testset "Empty result from filters" begin
            tg = TreatmentGroup(df; name_expr=r"^NONEXISTENT")
            @test length(tg) == 0
            @test isempty(get_idxs(tg))
            @test isempty(get_vnames(tg))
        end

        @testset "Single column result" begin
            tg = TreatmentGroup(df; name_expr=["V1"])
            @test length(tg) == 1
            @test get_vnames(tg) == ["V1"]
        end
    end

    @testset "show method" begin
        tg = TreatmentGroup(df; name_expr=r"^V")

        @testset "one-line show" begin
            io = IOBuffer()
            show(io, tg)
            output = String(take!(io))
            @test contains(output, "TreatmentGroup")
            @test contains(output, "3")
        end

        @testset "multi-line show (text/plain)" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), tg)
            output = String(take!(io))
            @test contains(output, "TreatmentGroup")
            @test contains(output, "3 columns selected")
            @test contains(output, "dims filter:")
            @test contains(output, "selected indices:")
            @test contains(output, "[1, 2, 3]")
        end
    end
end