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

# ---------------------------------------------------------------------------- #
#                            discrete_encode                                   #
# ---------------------------------------------------------------------------- #
@testset "discrete_encode" begin
    @testset "Matrix overload" begin
        X = Matrix{Any}(hcat(
            [missing, "blue", "green", "red", "blue"],
            [:circle, :square, :triangle, :square, missing]
        ))
        codes, lvls = DT.discrete_encode(X)

        @test length(codes) == 2
        @test length(lvls) == 2

        # first column: missing at row 1
        @test ismissing(codes[1][1])
        @test all(!ismissing, codes[1][2:end])
        @test "blue" in lvls[1]
        @test "green" in lvls[1]
        @test "red" in lvls[1]

        # second column: missing at row 5
        @test ismissing(codes[2][5])
        @test all(!ismissing, codes[2][1:4])
    end

    @testset "Vector overload" begin
        x = [missing, "blue", "green", "red", "blue"]
        codes, lvls = DT.discrete_encode(x)

        @test ismissing(codes[1])
        @test all(!ismissing, codes[2:end])
        @test sort(lvls) == ["blue", "green", "red"]
    end

    @testset "NaN treated as missing" begin
        x = [1.0, NaN, 2.0, 3.0]
        codes, lvls = DT.discrete_encode(x)

        @test ismissing(codes[2])
        @test all(!ismissing, codes[[1, 3, 4]])
    end

    @testset "No missing values" begin
        x = ["a", "b", "c", "a"]
        codes, lvls = DT.discrete_encode(x)

        @test all(!ismissing, codes)
        @test sort(lvls) == ["a", "b", "c"]
        # same label → same code
        @test codes[1] == codes[4]
    end

    @testset "All missing" begin
        x = [missing, missing, missing]
        codes, lvls = DT.discrete_encode(x)

        @test all(ismissing, codes)
        @test isempty(lvls)
    end
end

# ---------------------------------------------------------------------------- #
#                           _reindex_groups                                    #
# ---------------------------------------------------------------------------- #
@testset "_reindex_groups" begin
    @testset "Nothing input" begin
        @test DT._reindex_groups(nothing, [1, 2, 3]) === nothing
    end

    @testset "Full subset" begin
        groups = [[1, 2], [3, 4]]
        result = DT._reindex_groups(groups, [1, 2, 3, 4])
        @test result == [[1, 2], [3, 4]]
    end

    @testset "Partial subset" begin
        groups = [[1, 2, 3], [4, 5, 6]]
        result = DT._reindex_groups(groups, [2, 4, 5])
        # old 2→new 1, old 4→new 2, old 5→new 3
        @test result == [[1], [2, 3]]
    end

    @testset "Group entirely removed" begin
        groups = [[1, 2], [3, 4]]
        result = DT._reindex_groups(groups, [3, 4])
        # group [1,2] gone; old 3→new 1, old 4→new 2
        @test result == [[1, 2]]
    end

    @testset "Empty result" begin
        groups = [[1, 2], [3, 4]]
        result = DT._reindex_groups(groups, [5, 6])
        @test result === nothing
    end
end

# ---------------------------------------------------------------------------- #
#                            _callable_name                                    #
# ---------------------------------------------------------------------------- #
@testset "_callable_name" begin
    @testset "Named function" begin
        @test DT._callable_name(maximum) == "maximum"
        @test DT._callable_name(minimum) == "minimum"
        @test DT._callable_name(sum) == "sum"
    end

    @testset "Anonymous function" begin
        anon = x -> x + 1
        name = DT._callable_name(anon)
        # should not error; returns some string representation
        @test name isa String
        @test !isempty(name)
    end

    @testset "Callable struct" begin
        struct MyCallable end
        (::MyCallable)(x) = x
        name = DT._callable_name(MyCallable())
        @test name == "MyCallable"
    end
end

# ---------------------------------------------------------------------------- #
#                            DiscreteDataset                                   #
# ---------------------------------------------------------------------------- #
@testset "DiscreteDataset" begin
    # Build a DiscreteDataset via direct constructor
    codes_mat = Union{Missing,Int}[missing 1; 1 2; 2 3; 3 2; 1 missing]
    info_disc = [
        DT.DiscreteFeat{String}([1], "str_col", categorical(["blue", "green", "red"]), [2, 3, 4, 5], [1]),
        DT.DiscreteFeat{Symbol}([2], "sym_col", categorical(string.([:circle, :square, :triangle])), [1, 2, 3, 4], [5]),
    ]
    ds = DT.DiscreteDataset(codes_mat, info_disc)

    @testset "Construction" begin
        @test ds isa DT.DiscreteDataset
        @test ds isa DT.AbstractDataset
    end

    @testset "Base.size" begin
        @test size(ds) == (5, 2)
        @test size(ds, 1) == 5
        @test size(ds, 2) == 2
    end

    @testset "Base.length" begin
        @test length(ds) == 2
    end

    @testset "Base.ndims" begin
        @test ndims(ds) == 2
    end

    @testset "Base.eachindex" begin
        @test collect(eachindex(ds)) == [1, 2]
    end

    @testset "Base.iterate" begin
        feats = collect(ds)
        @test length(feats) == 2
        @test feats[1] === info_disc[1]
        @test feats[2] === info_disc[2]
    end

    @testset "Base.eltype" begin
        @test eltype(ds) == DT.DiscreteFeat
    end

    @testset "Base.getindex - single" begin
        sub = ds[1]
        @test sub isa DT.DiscreteDataset
        @test size(sub) == (5, 1)
        @test length(sub) == 1
    end

    @testset "Base.getindex - vector" begin
        sub = ds[[1, 2]]
        @test sub isa DT.DiscreteDataset
        @test size(sub, 2) == 2
    end

    @testset "Base.view" begin
        sub = @view ds[1]
        @test sub isa DT.DiscreteDataset
        @test length(sub) == 1

        sub2 = @view ds[1:2]
        @test length(sub2) == 2

        sub3 = @view ds[:]
        @test length(sub3) == 2
    end

    @testset "Getter methods" begin
        @test DT.get_data(ds) === codes_mat
        @test isequal(DT.get_data(ds, 1), @view codes_mat[:, 1])
        @test isequal(DT.get_data(ds, [1, 2]), @view codes_mat[:, [1, 2]])
        @test DT.get_info(ds) === info_disc
        @test DT.get_info(ds, 1) === info_disc[1]
        @test DT.get_nrows(ds) == 5
        @test DT.get_ncols(ds) == 2
        @test DT.get_vnames(ds) == ["str_col", "sym_col"]
        @test DT.get_vnames(ds, 1) == "str_col"
        @test DT.get_vnames(ds, [1, 2]) == ["str_col", "sym_col"]
        @test DT.get_idxs(ds) == [[1], [2]]
        @test DT.get_idxs(ds, 1) == [1]
    end

    @testset "Base.show - one line" begin
        str = sprint(show, ds)
        @test occursin("DiscreteDataset", str)
        @test occursin("5×2", str)
    end

    @testset "Base.show - multi-line" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("DiscreteDataset", str)
        @test occursin("5 rows × 2 columns", str)
        @test occursin("str_col", str)
        @test occursin("sym_col", str)
        @test occursin("levels per column", str)
        @test occursin("columns with missing", str)
    end

    @testset "Base.show - no missing" begin
        info_no_miss = [
            DT.DiscreteFeat{String}([1], "col1", categorical(["a", "b"]), [1, 2, 3], Int[]),
        ]
        ds_no_miss = DT.DiscreteDataset(Union{Missing,Int}[1; 2; 1;;], info_no_miss)
        str = sprint(show, MIME("text/plain"), ds_no_miss)
        @test !occursin("columns with missing", str)
    end
end

# ---------------------------------------------------------------------------- #
#                          ContinuousDataset                                   #
# ---------------------------------------------------------------------------- #
@testset "ContinuousDataset" begin
    data_cont = Union{Missing,Float64}[
        NaN     2.5   3.2   4.1   5.0;
        missing missing 4.2 NaN   6.0;
        3.0     4.5   5.2   NaN   7.0;
        4.0     5.5   missing 7.1 8.0;
        5.6     NaN   2.4   5.5   1.8
    ]
    info_cont = [
        DT.ContinuousFeat{Float64}([1], "V1", [3, 4, 5], [2], [1]),
        DT.ContinuousFeat{Float64}([2], "V2", [1, 3, 4], [2], [5]),
        DT.ContinuousFeat{Float64}([3], "V3", [1, 2, 3, 5], [4], Int[]),
        DT.ContinuousFeat{Float64}([4], "V4", [1, 4, 5], Int[], [2, 3]),
        DT.ContinuousFeat{Float64}([5], "V5", [1, 2, 3, 4, 5], Int[], Int[]),
    ]
    ds = DT.ContinuousDataset(data_cont, info_cont)

    @testset "Construction" begin
        @test ds isa DT.ContinuousDataset{Float64}
        @test ds isa DT.AbstractDataset
    end

    @testset "Base.size" begin
        @test size(ds) == (5, 5)
        @test size(ds, 1) == 5
        @test size(ds, 2) == 5
    end

    @testset "Base.length" begin
        @test length(ds) == 5
    end

    @testset "Base.eltype" begin
        @test eltype(ds) == DT.ContinuousFeat{Float64}
    end

    @testset "Base.getindex - single" begin
        sub = ds[3]
        @test sub isa DT.ContinuousDataset
        @test size(sub, 2) == 1
        @test DT.get_vnames(sub) == ["V3"]
    end

    @testset "Base.getindex - vector" begin
        sub = ds[[1, 3, 5]]
        @test sub isa DT.ContinuousDataset
        @test size(sub, 2) == 3
    end

    @testset "Base.iterate" begin
        feats = collect(ds)
        @test length(feats) == 5
        @test all(f -> f isa DT.ContinuousFeat{Float64}, feats)
    end

    @testset "Getter methods" begin
        @test DT.get_data(ds) === data_cont
        @test DT.get_info(ds) === info_cont
        @test DT.get_nrows(ds) == 5
        @test DT.get_ncols(ds) == 5
        @test DT.get_vnames(ds) == ["V1", "V2", "V3", "V4", "V5"]
        @test DT.get_idxs(ds) == [[1], [2], [3], [4], [5]]
    end

    @testset "Base.show - one line" begin
        str = sprint(show, ds)
        @test occursin("ContinuousDataset{Float64}", str)
        @test occursin("5×5", str)
    end

    @testset "Base.show - multi-line" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("ContinuousDataset{Float64}", str)
        @test occursin("5 rows × 5 columns", str)
        @test occursin("V1", str)
        @test occursin("columns with missing", str)
        @test occursin("columns with NaN", str)
        @test occursin("float type: Float64", str)
    end

    @testset "Base.show - no issues" begin
        info_clean = [
            DT.ContinuousFeat{Float64}([1], "clean", [1, 2, 3], Int[], Int[]),
        ]
        ds_clean = DT.ContinuousDataset(Float64[1.0; 2.0; 3.0;;], info_clean)
        str = sprint(show, MIME("text/plain"), ds_clean)
        @test !occursin("columns with missing", str)
        @test !occursin("columns with NaN", str)
    end

    @testset "Base.view" begin
        sub = @view ds[1]
        @test sub isa DT.ContinuousDataset
        @test length(sub) == 1

        sub2 = @view ds[1:3]
        @test sub2 isa DT.ContinuousDataset
        @test length(sub2) == 3
        @test DT.get_vnames(sub2) == ["V1", "V2", "V3"]

        sub3 = @view ds[[2, 4]]
        @test sub3 isa DT.ContinuousDataset
        @test length(sub3) == 2
        @test DT.get_vnames(sub3) == ["V2", "V4"]

        sub4 = @view ds[:]
        @test length(sub4) == 5
    end

    @testset "Getter methods" begin
        @test DT.get_data(ds) === data_cont
        @test DT.get_info(ds) === info_cont
        @test DT.get_nrows(ds) == 5
        @test DT.get_ncols(ds) == 5
        @test DT.get_vnames(ds) == ["V1", "V2", "V3", "V4", "V5"]
        @test DT.get_idxs(ds) == [[1], [2], [3], [4], [5]]
    end

    @testset "get_info - vector of indices" begin
        sub_info = DT.get_info(ds, [1, 3, 5])
        @test length(sub_info) == 3
        @test DT.get_vname(sub_info[1]) == "V1"
        @test DT.get_vname(sub_info[2]) == "V3"
        @test DT.get_vname(sub_info[3]) == "V5"
    end

    @testset "get_idxs - vector of indices" begin
        sub_idxs = DT.get_idxs(ds, [2, 4])
        @test length(sub_idxs) == 2
        @test sub_idxs == [[2], [4]]
    end
end

# ---------------------------------------------------------------------------- #
#                          MultidimDataset (Aggregate)                         #
# ---------------------------------------------------------------------------- #
@testset "MultidimDataset - Aggregate" begin
    # Simulate an aggregated dataset: 2 original columns, 2 features, 3 windows each = 12 columns
    nrows = 5
    ncols_agg = 12
    data_agg = rand(Float64, nrows, ncols_agg)

    info_agg = DT.AggregateFeat{Float64}[]
    col_idx = 1
    for c in 1:2
        vname = c == 1 ? "ts1" : "ts2"
        valid = c == 1 ? [2, 4, 5] : [1, 2, 3, 4]
        miss = c == 1 ? [3] : Int[]
        nan = c == 1 ? [1] : [5]
        for feat_fn in [maximum, minimum]
            for w in 1:3
                push!(info_agg, DT.AggregateFeat{Float64}(
                    [1, col_idx], vname, 1, feat_fn, w,
                    valid, miss, nan, Int[], Int[]
                ))
                col_idx += 1
            end
        end
    end

    groups = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    ds = DT.MultidimDataset(data_agg, info_agg, groups)

    @testset "Construction" begin
        @test ds isa DT.MultidimDataset
        @test ds isa DT.AbstractDataset
    end

    @testset "Base.size" begin
        @test size(ds) == (5, 12)
        @test size(ds, 1) == 5
        @test size(ds, 2) == 12
    end

    @testset "Base.length" begin
        @test length(ds) == 12
    end

    @testset "Base.eltype" begin
        ET = eltype(ds)
        @test ET <: Union{<:DT.AggregateFeat, <:DT.ReduceFeat}
    end

    @testset "Base.getindex - single" begin
        sub = ds[1]
        @test sub isa DT.MultidimDataset
        @test length(sub) == 1
    end

    @testset "Base.getindex - vector" begin
        sub = ds[[1, 7]]
        @test sub isa DT.MultidimDataset
        @test length(sub) == 2
    end

    @testset "Base.getindex - vector with group reindexing" begin
        sub = ds[[1, 2, 3]]  # first group only
        @test sub isa DT.MultidimDataset
        @test length(sub) == 3
    end

    @testset "has_groups / get_groups" begin
        @test DT.has_groups(ds) == true
        @test DT.get_groups(ds) == groups
    end

    @testset "has_groups - nothing" begin
        ds_nogrp = DT.MultidimDataset(data_agg, info_agg, nothing)
        @test DT.has_groups(ds_nogrp) == false
    end

    @testset "Getter methods" begin
        @test DT.get_data(ds) === data_agg
        @test DT.get_info(ds) === info_agg
        @test DT.get_nrows(ds) == 5
        @test DT.get_ncols(ds) == 12
        @test DT.get_dims(ds) == fill(1, 12)
        @test DT.get_dims(ds, 1) == 1
        @test DT.get_dims(ds, [1, 7]) == [1, 1]
    end

    @testset "get_vnames - aggregate mode" begin
        vnames = DT.get_vnames(ds)
        @test length(vnames) == 12
        # Should include feature name and window info
        @test all(v -> occursin(",", v), vnames)
    end

    @testset "get_vnames - aggregate groupby_split" begin
        vnames_grouped = DT.get_vnames(ds; groupby_split=true)
        @test length(vnames_grouped) == 2  # 2 groups
        @test length(vnames_grouped[1]) == 6
        @test length(vnames_grouped[2]) == 6
    end

    @testset "get_data - aggregate groupby_split" begin
        data_grouped = DT.get_data(ds; groupby_split=true)
        @test length(data_grouped) == 2
        @test size(data_grouped[1]) == (5, 6)
        @test size(data_grouped[2]) == (5, 6)
    end

    @testset "get_data - aggregate groupby_split=false" begin
        data_full = DT.get_data(ds; groupby_split=false)
        @test data_full === data_agg
    end

    @testset "get_idxs" begin
        idxs = DT.get_idxs(ds)
        @test length(idxs) == 12
        @test all(id -> length(id) == 2, idxs)
    end

    @testset "Base.show - one line" begin
        str = sprint(show, ds)
        @test occursin("MultidimDataset", str)
        @test occursin("5×12", str)
        @test occursin("dims=1", str)
        @test occursin("aggregate", str)
    end

    @testset "Base.show - multi-line" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("MultidimDataset", str)
        @test occursin("5 rows × 12 columns", str)
        @test occursin("mode: aggregate", str)
        @test occursin("ts1", str)
        @test occursin("ts2", str)
        @test occursin("features:", str)
        @test occursin("maximum", str)
        @test occursin("minimum", str)
        @test occursin("windows:", str)
    end

    @testset "Base.show - with missing and NaN" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("columns with missing", str) || occursin("columns with NaN", str)
    end

    @testset "Base.show - with internal hasmissing/hasnans" begin
        info_internal = [
            DT.AggregateFeat{Float64}(
                [1], "ts3", 1, maximum, 1,
                [1, 5], [4], [2, 3], [5], [1, 5]
            ),
        ]
        ds_internal = DT.MultidimDataset(rand(5, 1), info_internal, nothing)
        str = sprint(show, MIME("text/plain"), ds_internal)
        @test occursin("columns with internal missing", str)
        @test occursin("columns with internal NaN", str)
    end

    @testset "Base.view" begin
        sub = @view ds[1]
        @test sub isa DT.MultidimDataset
        @test length(sub) == 1

        sub2 = @view ds[1:6]
        @test sub2 isa DT.MultidimDataset
        @test length(sub2) == 6

        sub3 = @view ds[[1, 7]]
        @test sub3 isa DT.MultidimDataset
        @test length(sub3) == 2

        sub4 = @view ds[:]
        @test length(sub4) == 12
    end

    @testset "get_info - vector of indices" begin
        sub_info = DT.get_info(ds, [1, 7])
        @test length(sub_info) == 2
        @test DT.get_vname(sub_info[1]) == "ts1"
        @test DT.get_vname(sub_info[2]) == "ts2"
    end

    @testset "get_idxs - vector of indices" begin
        sub_idxs = DT.get_idxs(ds, [1, 12])
        @test length(sub_idxs) == 2
        @test sub_idxs[1] == [1, 1]
        @test sub_idxs[2] == [1, 12]
    end
end

# ---------------------------------------------------------------------------- #
#                          MultidimDataset (Reduce)                            #
# ---------------------------------------------------------------------------- #
@testset "MultidimDataset - Reduce" begin
    reduce_fn = x -> x[1:2:end]

    # Simulate a reduced dataset: 3 columns, each cell is a small vector
    nrows = 5
    data_red = Matrix{Any}(undef, nrows, 3)
    for r in 1:nrows, c in 1:3
        data_red[r, c] = rand(Float64, 4)
    end

    info_red = [
        DT.ReduceFeat{Float64}([1], "ts1", 1, reduce_fn, [2, 4, 5], [3], [1], Int[], Int[]),
        DT.ReduceFeat{Float64}([2], "ts2", 1, reduce_fn, [1, 2, 3, 4], Int[], [5], Int[], Int[]),
        DT.ReduceFeat{Float64}([3], "ts3", 1, reduce_fn, [1, 5], [4], [2, 3], [5], [1, 5]),
    ]
    ds = DT.MultidimDataset(data_red, info_red)

    @testset "Construction" begin
        @test ds isa DT.MultidimDataset
        @test ds isa DT.AbstractDataset
    end

    @testset "Base.size" begin
        @test size(ds, 1) == 5
    end

    @testset "Base.length" begin
        @test length(ds) == 3
    end

    @testset "has_groups default nothing" begin
        @test DT.has_groups(ds) == false
    end

    @testset "Getter methods" begin
        @test DT.get_data(ds) === data_red
        @test DT.get_info(ds) === info_red
        @test DT.get_nrows(ds) == 5
        @test DT.get_dims(ds) == [1, 1, 1]
        @test DT.get_vnames(ds) == ["ts1", "ts2", "ts3"]
    end

    @testset "Base.getindex - single" begin
        sub = ds[2]
        @test sub isa DT.MultidimDataset
        @test length(sub) == 1
        @test DT.get_vnames(sub) == ["ts2"]
    end

    @testset "Base.getindex - vector" begin
        sub = ds[[1, 3]]
        @test sub isa DT.MultidimDataset
        @test length(sub) == 2
    end

    @testset "Base.show - one line" begin
        str = sprint(show, ds)
        @test occursin("MultidimDataset", str)
        @test occursin("dims=1", str)
        @test occursin("reducesize", str)
    end

    @testset "Base.show - multi-line" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("MultidimDataset", str)
        @test occursin("mode: reducesize", str)
        @test occursin("ts1", str)
        @test occursin("ts2", str)
        @test occursin("ts3", str)
        @test occursin("reduce function:", str)
    end

    @testset "Base.show - with internal issues" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("columns with missing", str) || occursin("columns with NaN", str)
        @test occursin("columns with internal missing", str)
        @test occursin("columns with internal NaN", str)
    end
end

# ---------------------------------------------------------------------------- #
#                     MultidimDataset - 2D (images)                            #
# ---------------------------------------------------------------------------- #
@testset "MultidimDataset - 2D Aggregate" begin
    nrows = 5
    ncols_agg = 8  # 2 columns × 2 features × 2 windows
    data_2d = rand(Float64, nrows, ncols_agg)

    info_2d = DT.AggregateFeat{Float64}[]
    col_idx = 1
    for c in 1:2
        vname = c == 1 ? "img1" : "img2"
        valid = c == 1 ? [1, 2, 3, 4, 5] : [2, 3, 4, 5]
        miss = Int[]
        nan = c == 1 ? Int[] : [1]
        for feat_fn in [maximum, minimum]
            for w in 1:2
                push!(info_2d, DT.AggregateFeat{Float64}(
                    [1, col_idx], vname, 2, feat_fn, w,
                    valid, miss, nan, Int[], Int[]
                ))
                col_idx += 1
            end
        end
    end

    ds = DT.MultidimDataset(data_2d, info_2d, nothing)

    @testset "Construction" begin
        @test ds isa DT.MultidimDataset
    end

    @testset "Dims = 2" begin
        @test all(d -> d == 2, DT.get_dims(ds))
    end

    @testset "Base.show - one line (dims=2)" begin
        str = sprint(show, ds)
        @test occursin("dims=2", str)
        @test occursin("aggregate", str)
    end

    @testset "Base.show - multi-line (dims=2)" begin
        str = sprint(show, MIME("text/plain"), ds)
        @test occursin("mode: aggregate", str)
        @test occursin("img1", str)
        @test occursin("img2", str)
    end
end

# ---------------------------------------------------------------------------- #
#                     MultidimDataset - mixed dims                             #
# ---------------------------------------------------------------------------- #
@testset "MultidimDataset - mixed dims show" begin
    data_mix = rand(Float64, 5, 2)
    info_mix = [
        DT.AggregateFeat{Float64}([1], "ts1", 1, maximum, 1, [1, 2, 3, 4, 5], Int[], Int[], Int[], Int[]),
        DT.AggregateFeat{Float64}([2], "img1", 2, maximum, 1, [1, 2, 3, 4, 5], Int[], Int[], Int[], Int[]),
    ]
    ds = DT.MultidimDataset(data_mix, info_mix, nothing)

    @testset "One-line show displays multiple dims" begin
        str = sprint(show, ds)
        # Should show both dims, not just a single number
        @test occursin("dims=", str)
        @test occursin("[1, 2]", str)
    end
end

# ---------------------------------------------------------------------------- #
#                     _callable_name edge cases                                #
# ---------------------------------------------------------------------------- #
@testset "_callable_name edge cases" begin
    @testset "do-block anonymous" begin
        f = function(x) x^2 end
        name = DT._callable_name(f)
        @test name isa String
    end

    @testset "Composed function" begin
        f = sum ∘ abs
        name = DT._callable_name(f)
        @test name isa String
        @test !isempty(name)
    end
end