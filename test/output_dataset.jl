using Test
using DataTreatments
const DT = DataTreatments
using CategoricalArrays
using Statistics

@testset "OutputDataset" begin
    @testset "discrete_encode" begin
        @testset "basic encoding" begin
            X = Matrix{Any}(undef, 4, 2)
            X[:, 1] = ["a", "b", "a", "c"]
            X[:, 2] = [1, 2, 1, 3]

            codes, lvls = DT.discrete_encode(X)

            @test length(codes) == 2
            @test length(lvls) == 2
            # codes are integers
            @test all(c -> c isa Int, codes[1])
            @test all(c -> c isa Int, codes[2])
            # levels are strings (CategoricalValue wraps them)
            @test all(l -> string(l) isa String, lvls[1])
            @test all(l -> string(l) isa String, lvls[2])
            # reconstruction
            @test [string(lvls[1][c]) for c in codes[1]] == ["a", "b", "a", "c"]
        end

        @testset "missing preserved" begin
            X = Matrix{Any}(undef, 3, 1)
            X[:, 1] = ["a", missing, "b"]

            codes, lvls = DT.discrete_encode(X)

            @test ismissing(codes[1][2])
            @test !ismissing(codes[1][1])
            @test !ismissing(codes[1][3])
            @test any(l -> string(l) == "a", lvls[1])
            @test any(l -> string(l) == "b", lvls[1])
        end

        @testset "NaN treated as missing" begin
            X = Matrix{Any}(undef, 3, 1)
            X[:, 1] = [1.0, NaN, 2.0]

            codes, lvls = DT.discrete_encode(X)

            @test ismissing(codes[1][2])
            @test !any(l -> string(l) == "NaN", lvls[1])
        end
    end

    @testset "DiscreteDataset" begin

        # --- direct constructor (use unparameterized Vector{DiscreteFeat}) ---
        info_discrete = DiscreteFeat[
            DiscreteFeat{String}([1, 1], "color", categorical(["red", "blue", "green"]), [1, 2, 3], Int[]),
            DiscreteFeat{String}([1, 2], "shape", categorical(["circle", "square"]), [1, 2, 3], Int[]),
        ]
        mat_discrete = Matrix{Any}([1 1; 2 2; 3 1])
        dd = DiscreteDataset(mat_discrete, info_discrete)

        @testset "Direct construction" begin
            @test dd isa DiscreteDataset
            @test dd isa DT.AbstractDataset
        end

        @testset "Base methods" begin
            @test size(dd) == (3, 2)
            @test size(dd, 1) == 3
            @test size(dd, 2) == 2
            @test length(dd) == 2
            @test ndims(dd) == 2
            @test eachindex(dd) == Base.OneTo(2)
            @test eltype(dd) == DiscreteFeat
        end

        @testset "iterate" begin
            collected = collect(dd)
            @test length(collected) == 2
            @test all(f -> f isa DiscreteFeat, collected)
            @test collected[1] === info_discrete[1]
            @test collected[2] === info_discrete[2]
        end

        @testset "getindex" begin
            @test dd[1] === info_discrete[1]
            @test dd[2] === info_discrete[2]
            @test dd[[1, 2]] == info_discrete
        end

        @testset "Getter methods" begin
            @test get_dataset(dd) === mat_discrete
            @test get_dataset(dd, 1) == [1, 2, 3]
            @test isequal(get_dataset(dd, [1, 2]), mat_discrete)
            @test get_info(dd) === info_discrete
            @test get_info(dd, 1) === info_discrete[1]
            @test get_info(dd, [1, 2]) == info_discrete
            @test get_nrows(dd) == 3
            @test get_ncols(dd) == 2
            @test get_vnames(dd) == ["color", "shape"]
            @test get_vnames(dd, 1) == "color"
            @test get_vnames(dd, [1, 2]) == ["color", "shape"]
            @test get_idxs(dd) == [[1, 1], [1, 2]]
            @test get_idxs(dd, 1) == [1, 1]
            @test get_idxs(dd, [1, 2]) == [[1, 1], [1, 2]]
        end

        @testset "show - one-line" begin
            io = IOBuffer()
            show(io, dd)
            output = String(take!(io))
            @test contains(output, "DiscreteDataset")
            @test contains(output, "3×2")
        end

        @testset "show - multi-line" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), dd)
            output = String(take!(io))
            @test contains(output, "DiscreteDataset")
            @test contains(output, "3 rows × 2 columns")
            @test contains(output, "vnames:")
            @test contains(output, "levels per column:")
        end

        @testset "show - multi-line with missing" begin
            info_miss = DiscreteFeat[
                DiscreteFeat{String}([1, 1], "x", categorical(["a", "b"]), [1, 3], [2]),
            ]
            dd_miss = DiscreteDataset(Matrix{Any}(reshape(Any[1, missing, 2], 3, 1)), info_miss)
            io = IOBuffer()
            show(io, MIME"text/plain"(), dd_miss)
            output = String(take!(io))
            @test contains(output, "columns with missing: 1")
        end
    end

    @testset "ContinuousDataset" begin

        # --- direct constructor (use parameterized vector for ===) ---
        info_cont = ContinuousFeat{Float64}[
            ContinuousFeat{Float64}([1, 1], "temp", [1, 2, 4], [3], Int[]),
            ContinuousFeat{Float64}([1, 2], "pressure", [1, 2, 3], Int[], [4]),
        ]
        mat_cont = Matrix{Any}(Any[1.0 10.0; 2.0 20.0; missing 30.0; 4.0 NaN])
        cd = ContinuousDataset(mat_cont, info_cont)

        @testset "Direct construction" begin
            @test cd isa ContinuousDataset{Float64}
            @test cd isa DT.AbstractDataset
        end

        @testset "Base methods" begin
            @test size(cd) == (4, 2)
            @test size(cd, 1) == 4
            @test size(cd, 2) == 2
            @test length(cd) == 2
            @test ndims(cd) == 2
            @test eachindex(cd) == Base.OneTo(2)
            @test eltype(cd) == ContinuousFeat{Float64}
        end

        @testset "iterate" begin
            collected = collect(cd)
            @test length(collected) == 2
            @test all(f -> f isa ContinuousFeat, collected)
        end

        @testset "getindex" begin
            @test cd[1] === info_cont[1]
            @test cd[[1, 2]] == info_cont
        end

        @testset "Getter methods" begin
            @test get_dataset(cd) === mat_cont
            @test isequal(get_dataset(cd, 1), [1.0, 2.0, missing, 4.0])
            @test get_info(cd) == info_cont
            @test get_info(cd, 2) === info_cont[2]
            @test get_nrows(cd) == 4
            @test get_ncols(cd) == 2
            @test get_vnames(cd) == ["temp", "pressure"]
            @test get_vnames(cd, 1) == "temp"
            @test get_vnames(cd, [2]) == ["pressure"]
            @test get_idxs(cd) == [[1, 1], [1, 2]]
            @test get_idxs(cd, 2) == [1, 2]
        end

        @testset "show - one-line" begin
            io = IOBuffer()
            show(io, cd)
            output = String(take!(io))
            @test contains(output, "ContinuousDataset{Float64}")
            @test contains(output, "4×2")
        end

        @testset "show - multi-line" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), cd)
            output = String(take!(io))
            @test contains(output, "ContinuousDataset{Float64}")
            @test contains(output, "4 rows × 2 columns")
            @test contains(output, "vnames:")
            @test contains(output, "float type: Float64")
        end

        @testset "show - multi-line with missing and NaN" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), cd)
            output = String(take!(io))
            @test contains(output, "columns with missing: 1")
            @test contains(output, "columns with NaN: 1")
        end

        @testset "show - multi-line clean" begin
            info_clean = ContinuousFeat{Float64}[
                ContinuousFeat{Float64}([1, 1], "x", [1, 2, 3], Int[], Int[]),
            ]
            cd_clean = ContinuousDataset(Matrix{Any}(reshape(Any[1.0, 2.0, 3.0], 3, 1)), info_clean)
            io = IOBuffer()
            show(io, MIME"text/plain"(), cd_clean)
            output = String(take!(io))
            @test !contains(output, "columns with missing")
            @test !contains(output, "columns with NaN")
        end
    end

    @testset "MultidimDataset" begin

        # --- AggregateFeat (aggregate mode) - use parameterized vectors ---
        info_aggr = AggregateFeat{Float64}[
            AggregateFeat{Float64}([1, 1], "ts1", maximum, 2, [1, 2, 3], Int[], Int[], Int[], Int[]),
            AggregateFeat{Float64}([1, 2], "ts1", minimum, 2, [1, 2, 3], Int[], Int[], Int[], Int[]),
            AggregateFeat{Float64}([1, 3], "ts1", mean,    2, [1, 2, 3], Int[], Int[], Int[], Int[]),
        ]
        mat_aggr = Matrix{Any}(Any[5.0 1.0 3.0; 6.0 2.0 4.0; 7.0 3.0 5.0])
        md_aggr = MultidimDataset(mat_aggr, info_aggr)

        @testset "Direct construction - aggregate" begin
            @test md_aggr isa MultidimDataset{Float64}
            @test md_aggr isa DT.AbstractDataset
        end

        @testset "Base methods - aggregate" begin
            @test size(md_aggr) == (3, 3)
            @test size(md_aggr, 1) == 3
            @test size(md_aggr, 2) == 3
            @test length(md_aggr) == 3
            @test ndims(md_aggr) == 2
            @test eachindex(md_aggr) == Base.OneTo(3)
            @test eltype(md_aggr) == Union{AggregateFeat{Float64}, ReduceFeat{Float64}}
        end

        @testset "iterate - aggregate" begin
            collected = collect(md_aggr)
            @test length(collected) == 3
            @test all(f -> f isa AggregateFeat, collected)
        end

        @testset "getindex - aggregate" begin
            @test md_aggr[1] === info_aggr[1]
            @test md_aggr[[1, 3]] == [info_aggr[1], info_aggr[3]]
        end

        @testset "Getter methods - aggregate" begin
            @test get_dataset(md_aggr) === mat_aggr
            @test get_dataset(md_aggr, 1) == [5.0, 6.0, 7.0]
            @test isequal(get_dataset(md_aggr, [1, 2]), mat_aggr[:, 1:2])
            @test get_info(md_aggr) == info_aggr
            @test get_info(md_aggr, 1) === info_aggr[1]
            @test get_nrows(md_aggr) == 3
            @test get_ncols(md_aggr) == 3
            @test get_vnames(md_aggr) == ["ts1", "ts1", "ts1"]
            @test get_vnames(md_aggr, 1) == "ts1"
            @test get_vnames(md_aggr, [1, 2]) == ["ts1", "ts1"]
            @test get_idxs(md_aggr) == [[1, 1], [1, 2], [1, 3]]
            @test get_idxs(md_aggr, 2) == [1, 2]
            @test get_idxs(md_aggr, [1, 3]) == [[1, 1], [1, 3]]
        end

        @testset "show - one-line aggregate" begin
            io = IOBuffer()
            show(io, md_aggr)
            output = String(take!(io))
            @test contains(output, "MultidimDataset{Float64}")
            @test contains(output, "3×3")
            @test contains(output, "aggregate")
        end

        @testset "show - multi-line aggregate" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), md_aggr)
            output = String(take!(io))
            @test contains(output, "MultidimDataset{Float64}")
            @test contains(output, "3 rows × 3 columns")
            @test contains(output, "mode: aggregate")
            @test contains(output, "vnames:")
            @test contains(output, "features:")
            @test contains(output, "windows:")
        end

        @testset "show - multi-line aggregate with missing/NaN" begin
            info_aggr_dirty = AggregateFeat{Float64}[
                AggregateFeat{Float64}([1, 1], "ts", maximum, 1, [1], [2], [3], [4], [5]),
            ]
            md_dirty = MultidimDataset(Matrix{Any}(reshape(Any[5.0], 1, 1)), info_aggr_dirty)
            io = IOBuffer()
            show(io, MIME"text/plain"(), md_dirty)
            output = String(take!(io))
            @test contains(output, "columns with missing: 1")
            @test contains(output, "columns with NaN: 1")
            @test contains(output, "columns with internal missing: 1")
            @test contains(output, "columns with internal NaN: 1")
        end

        # --- ReduceFeat (reducesize mode) ---
        downsample = x -> x[1:2:end]
        info_reduce = ReduceFeat{Float64}[
            ReduceFeat{Float64}([2, 1], "signal1", downsample, [1, 2, 3], Int[], Int[], Int[], Int[]),
            ReduceFeat{Float64}([2, 2], "signal2", downsample, [1, 2, 3], Int[], Int[], Int[], Int[]),
        ]
        mat_reduce = Matrix{Any}(undef, 3, 2)
        mat_reduce[:, 1] = [collect(1.0:5.0), collect(2.0:6.0), collect(3.0:7.0)]
        mat_reduce[:, 2] = [collect(10.0:14.0), collect(11.0:15.0), collect(12.0:16.0)]
        md_reduce = MultidimDataset(mat_reduce, info_reduce)

        @testset "Direct construction - reducesize" begin
            @test md_reduce isa MultidimDataset{Float64}
            @test md_reduce isa DT.AbstractDataset
        end

        @testset "Base methods - reducesize" begin
            @test size(md_reduce) == (3, 2)
            @test length(md_reduce) == 2
        end

        @testset "iterate - reducesize" begin
            collected = collect(md_reduce)
            @test length(collected) == 2
            @test all(f -> f isa ReduceFeat, collected)
        end

        @testset "Getter methods - reducesize" begin
            @test get_vnames(md_reduce) == ["signal1", "signal2"]
            @test get_idxs(md_reduce) == [[2, 1], [2, 2]]
        end

        @testset "show - one-line reducesize" begin
            io = IOBuffer()
            show(io, md_reduce)
            output = String(take!(io))
            @test contains(output, "MultidimDataset{Float64}")
            @test contains(output, "3×2")
            @test contains(output, "reducesize")
        end

        @testset "show - multi-line reducesize" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), md_reduce)
            output = String(take!(io))
            @test contains(output, "mode: reducesize")
            @test contains(output, "vnames:")
            @test contains(output, "reduce function:")
        end

        @testset "show - multi-line reducesize with missing/NaN" begin
            info_reduce_dirty = ReduceFeat{Float64}[
                ReduceFeat{Float64}([1, 1], "sig", downsample, [1], [2], [3], [4], [5]),
            ]
            md_reduce_dirty = MultidimDataset(Matrix{Any}(reshape(Any[collect(1.0:5.0)], 1, 1)), info_reduce_dirty)
            io = IOBuffer()
            show(io, MIME"text/plain"(), md_reduce_dirty)
            output = String(take!(io))
            @test contains(output, "columns with missing: 1")
            @test contains(output, "columns with NaN: 1")
            @test contains(output, "columns with internal missing: 1")
            @test contains(output, "columns with internal NaN: 1")
        end
    end

    @testset "Lazy constructors" begin

        @testset "DiscreteDataset lazy constructor" begin
            # Note: the lazy constructor has a bug where it produces
            # Matrix{Union{Missing,Int}} which doesn't match the direct
            # constructor signature Matrix{Any} -> Vector{DiscreteFeat}.
            # Skipping until source is fixed.
            @test_broken begin
                raw = Matrix{Any}(undef, 4, 2)
                raw[:, 1] = ["a", "b", missing, "a"]
                raw[:, 2] = ["x", "y", "x", "z"]
                ds_struct = DatasetStructure(raw, ["color", "shape"])
                dd = DiscreteDataset([1], raw, ds_struct, [1, 2])
                dd isa DiscreteDataset
            end
        end

        @testset "ContinuousDataset lazy constructor" begin
            raw = Matrix{Any}(undef, 4, 2)
            raw[:, 1] = [1.0, 2.0, missing, 4.0]
            raw[:, 2] = [10.0, NaN, 30.0, 40.0]

            ds_struct = DatasetStructure(raw, ["temp", "pressure"])
            cd = ContinuousDataset([2], raw, ds_struct, [1, 2], Float64)

            @test cd isa ContinuousDataset{Float64}
            @test size(cd, 2) == 2
            @test get_vnames(cd) == ["temp", "pressure"]
            # missing preserved
            @test ismissing(get_dataset(cd)[3, 1])
            # converted to Float64
            valid_vals = filter(!ismissing, get_dataset(cd)[:, 1])
            @test all(v -> v isa Float64, valid_vals)
            # info populated
            @test length(get_info(cd)) == 2
            @test all(f -> f isa ContinuousFeat, get_info(cd))
        end

        @testset "ContinuousDataset lazy constructor - Float32" begin
            raw = Matrix{Any}(undef, 3, 1)
            raw[:, 1] = [1.0, 2.0, 3.0]

            ds_struct = DatasetStructure(raw, ["val"])
            cd32 = ContinuousDataset([3], raw, ds_struct, [1], Float32)

            @test cd32 isa ContinuousDataset{Float32}
            valid_vals = filter(!ismissing, get_dataset(cd32)[:, 1])
            @test all(v -> v isa Float32, valid_vals)
        end
    end

    @testset "Subtyping" begin
        @test DiscreteDataset <: DT.AbstractDataset
        @test ContinuousDataset <: DT.AbstractDataset
        @test MultidimDataset <: DT.AbstractDataset
    end

    @testset "Cross-type getter consistency" begin
        info_d = DiscreteFeat[DiscreteFeat{String}([1, 1], "a", categorical(["x"]), [1], Int[])]
        info_c = ContinuousFeat{Float64}[ContinuousFeat{Float64}([2, 1], "b", [1], Int[], Int[])]
        info_a = AggregateFeat{Float64}[AggregateFeat{Float64}([3, 1], "c", maximum, 1, [1], Int[], Int[], Int[], Int[])]

        dd = DiscreteDataset(Matrix{Any}(reshape(Any[1], 1, 1)), info_d)
        cd = ContinuousDataset(Matrix{Any}(reshape(Any[1.0], 1, 1)), info_c)
        md = MultidimDataset(Matrix{Any}(reshape(Any[1.0], 1, 1)), info_a)

        for (ds, expected_name) in [(dd, "a"), (cd, "b"), (md, "c")]
            @test get_vnames(ds) == [expected_name]
            @test get_vnames(ds, 1) == expected_name
            @test get_nrows(ds) == 1
            @test get_ncols(ds) == 1
            @test size(ds) == (1, 1)
            @test length(ds) == 1
            @test eachindex(ds) == Base.OneTo(1)
            @test ds[1] isa DT.AbstractDataFeature
        end
    end
end