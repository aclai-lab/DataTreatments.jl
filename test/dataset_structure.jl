using Test
using DataTreatments

using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

@testset "DatasetStructure" begin
    # ------------------------------------------------------------------ #
    #                        shared rich dataset                          #
    # ------------------------------------------------------------------ #
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

    ds_rich = DatasetStructure(df)

    # ------------------------------------------------------------------ #
    #                           constructors                              #
    # ------------------------------------------------------------------ #
    @testset "Constructor - Matrix" begin
        dataset = Matrix{Any}(undef, 5, 3)
        dataset[:, 1] = [1, 2, missing, 4, 5]
        dataset[:, 2] = [1.0, NaN, 3.0, missing, 5.0]
        dataset[:, 3] = [collect(1.0:3.0), collect(2.0:4.0), collect(3.0:5.0), missing, NaN]
        col_names = ["a", "b", "c"]

        ds = DatasetStructure(dataset, col_names)

        @test length(ds) == 3
        @test size(ds) == (3,)
        @test get_vnames(ds) == col_names
    end

    @testset "Constructor - DataFrame" begin
        @test length(ds_rich) == 18
        @test size(ds_rich) == (18,)
        @test get_vnames(ds_rich) == names(df)
    end

    @testset "Constructor - default vnames" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, 2, 3]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset)

        @test get_vnames(ds) == ["V1", "V2"]
    end

    # ------------------------------------------------------------------ #
    #                       size and length methods                        #
    # ------------------------------------------------------------------ #
    @testset "Size and length methods" begin
        @test size(ds_rich) == (18,)
        @test length(ds_rich) == 18
        @test eachindex(ds_rich) == Base.OneTo(18)
        @test collect(eachindex(ds_rich)) == collect(1:18)
    end

    # ------------------------------------------------------------------ #
    #                  scalar columns - missing and NaN                    #
    # ------------------------------------------------------------------ #
    @testset "Scalar columns - missing and NaN" begin
        # str_col (col 1): missing at row 1
        @test get_missingidxs(ds_rich, 1) == [1]
        @test get_nanidxs(ds_rich, 1) == Int[]
        @test get_valididxs(ds_rich, 1) == [2, 3, 4, 5]
        @test get_dims(ds_rich, 1) == 0

        # sym_col (col 2): missing at row 5
        @test get_missingidxs(ds_rich, 2) == [5]
        @test get_nanidxs(ds_rich, 2) == Int[]
        @test get_valididxs(ds_rich, 2) == [1, 2, 3, 4]
        @test get_dims(ds_rich, 2) == 0

        # cat_col (col 3): missing at row 3
        @test get_missingidxs(ds_rich, 3) == [3]
        @test get_nanidxs(ds_rich, 3) == Int[]
        @test get_valididxs(ds_rich, 3) == [1, 2, 4, 5]
        @test get_dims(ds_rich, 3) == 0

        # uint_col (col 4): all valid
        @test get_missingidxs(ds_rich, 4) == Int[]
        @test get_nanidxs(ds_rich, 4) == Int[]
        @test get_valididxs(ds_rich, 4) == [1, 2, 3, 4, 5]
        @test get_dims(ds_rich, 4) == 0

        # int_col (col 5): all valid
        @test get_missingidxs(ds_rich, 5) == Int[]
        @test get_nanidxs(ds_rich, 5) == Int[]
        @test get_valididxs(ds_rich, 5) == [1, 2, 3, 4, 5]
        @test get_dims(ds_rich, 5) == 0

        # V1 (col 6): NaN at row 1, missing at row 2
        @test get_nanidxs(ds_rich, 6) == [1]
        @test get_missingidxs(ds_rich, 6) == [2]
        @test get_valididxs(ds_rich, 6) == [3, 4, 5]
        @test get_dims(ds_rich, 6) == 0

        # V2 (col 7): missing at row 2, NaN at row 5
        @test get_missingidxs(ds_rich, 7) == [2]
        @test get_nanidxs(ds_rich, 7) == [5]
        @test get_valididxs(ds_rich, 7) == [1, 3, 4]
        @test get_dims(ds_rich, 7) == 0

        # V3 (col 8): missing at row 4
        @test get_missingidxs(ds_rich, 8) == [4]
        @test get_nanidxs(ds_rich, 8) == Int[]
        @test get_valididxs(ds_rich, 8) == [1, 2, 3, 5]
        @test get_dims(ds_rich, 8) == 0

        # V4 (col 9): NaN at rows 2, 3
        @test get_missingidxs(ds_rich, 9) == Int[]
        @test get_nanidxs(ds_rich, 9) == [2, 3]
        @test get_valididxs(ds_rich, 9) == [1, 4, 5]
        @test get_dims(ds_rich, 9) == 0

        # V5 (col 10): all valid
        @test get_missingidxs(ds_rich, 10) == Int[]
        @test get_nanidxs(ds_rich, 10) == Int[]
        @test get_valididxs(ds_rich, 10) == [1, 2, 3, 4, 5]
        @test get_dims(ds_rich, 10) == 0
    end

    # ------------------------------------------------------------------ #
    #              array columns - dims, hasmissing, hasnans              #
    # ------------------------------------------------------------------ #
    @testset "Array columns - time series" begin
        # ts1 (col 11): NaN at row 1, missing at row 3, vectors at rows 2,4,5
        @test get_dims(ds_rich, 11) == 1
        @test get_nanidxs(ds_rich, 11) == [1]
        @test get_missingidxs(ds_rich, 11) == [3]
        @test get_valididxs(ds_rich, 11) == [2, 4, 5]
        @test get_hasmissing(ds_rich, 11) == Int[]
        @test get_hasnans(ds_rich, 11) == Int[]

        # ts2 (col 12): NaN at row 5, all others are valid vectors
        @test get_dims(ds_rich, 12) == 1
        @test get_nanidxs(ds_rich, 12) == [5]
        @test get_missingidxs(ds_rich, 12) == Int[]
        @test get_valididxs(ds_rich, 12) == [1, 2, 3, 4]
        @test get_hasmissing(ds_rich, 12) == Int[]
        @test get_hasnans(ds_rich, 12) == Int[]

        # ts3 (col 13): NaN at rows 2,3; missing at row 4
        # row 1: [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2] — internal NaN
        # row 5: [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2] — internal NaN + missing
        @test get_dims(ds_rich, 13) == 1
        @test get_nanidxs(ds_rich, 13) == [2, 3]
        @test get_missingidxs(ds_rich, 13) == [4]
        @test get_valididxs(ds_rich, 13) == [1, 5]
        @test get_hasmissing(ds_rich, 13) == [5]
        @test get_hasnans(ds_rich, 13) == [1, 5]

        # ts4 (col 14): missing at row 2
        # row 1: [6.0, 5.2, missing, ...] — internal missing
        # row 3: [5.0, 4.2, NaN, 3.4, missing, ...] — internal NaN + missing
        # row 4: [8.0, 7.2, missing, 6.4, NaN, ...] — internal missing + NaN
        # row 5: [9.0, NaN, 8.2, missing, ...] — internal NaN + missing
        @test get_dims(ds_rich, 14) == 1
        @test get_nanidxs(ds_rich, 14) == Int[]
        @test get_missingidxs(ds_rich, 14) == [2]
        @test get_valididxs(ds_rich, 14) == [1, 3, 4, 5]
        @test get_hasmissing(ds_rich, 14) == [1, 3, 4, 5]
        @test get_hasnans(ds_rich, 14) == [3, 4, 5]
    end

    @testset "Array columns - images" begin
        # img1 (col 15): all valid 6x6 matrices
        @test get_dims(ds_rich, 15) == 2
        @test get_missingidxs(ds_rich, 15) == Int[]
        @test get_nanidxs(ds_rich, 15) == Int[]
        @test get_valididxs(ds_rich, 15) == [1, 2, 3, 4, 5]
        @test get_hasmissing(ds_rich, 15) == Int[]
        @test get_hasnans(ds_rich, 15) == Int[]

        # img2 (col 16): NaN at row 1, valid matrices at rows 2-5
        @test get_dims(ds_rich, 16) == 2
        @test get_nanidxs(ds_rich, 16) == [1]
        @test get_missingidxs(ds_rich, 16) == Int[]
        @test get_valididxs(ds_rich, 16) == [2, 3, 4, 5]
        @test get_hasmissing(ds_rich, 16) == Int[]
        @test get_hasnans(ds_rich, 16) == Int[]

        # img3 (col 17): all valid 6x6 matrices
        @test get_dims(ds_rich, 17) == 2
        @test get_missingidxs(ds_rich, 17) == Int[]
        @test get_nanidxs(ds_rich, 17) == Int[]
        @test get_valididxs(ds_rich, 17) == [1, 2, 3, 4, 5]
        @test get_hasmissing(ds_rich, 17) == Int[]
        @test get_hasnans(ds_rich, 17) == Int[]

        # img4 (col 18): missing at row 3, valid matrices elsewhere
        @test get_dims(ds_rich, 18) == 2
        @test get_missingidxs(ds_rich, 18) == [3]
        @test get_nanidxs(ds_rich, 18) == Int[]
        @test get_valididxs(ds_rich, 18) == [1, 2, 4, 5]
        @test get_hasmissing(ds_rich, 18) == Int[]
        @test get_hasnans(ds_rich, 18) == Int[]
    end

    # ------------------------------------------------------------------ #
    #                        datatype inference                           #
    # ------------------------------------------------------------------ #
    @testset "Datatype inference" begin
        # str_col: String
        @test get_datatype(ds_rich, 1) == String

        # sym_col: Symbol
        @test get_datatype(ds_rich, 2) == Symbol

        # cat_col: CategoricalValue{String, UInt32}
        @test get_datatype(ds_rich, 3) <: CategoricalValue

        # uint_col: UInt32
        @test get_datatype(ds_rich, 4) == UInt32

        # int_col: Int
        @test get_datatype(ds_rich, 5) == Int

        # V1-V5 (cols 6-10): Float64
        for i in 6:10
            @test get_datatype(ds_rich, i) == Float64
        end

        # ts1-ts4 (cols 11-14): Vector (1D array)
        for i in 11:14
            @test get_datatype(ds_rich, i) <: AbstractVector
        end

        # img1-img4 (cols 15-18): Matrix (2D array)
        for i in 15:18
            @test get_datatype(ds_rich, i) <: AbstractMatrix
        end
    end

    # ------------------------------------------------------------------ #
    #                    getter methods - full vectors                     #
    # ------------------------------------------------------------------ #
    @testset "Getter methods - full vectors" begin
        @test get_vnames(ds_rich) == names(df)
        @test length(get_datatype(ds_rich)) == 18
        @test length(get_dims(ds_rich)) == 18
        @test length(get_valididxs(ds_rich)) == 18
        @test length(get_missingidxs(ds_rich)) == 18
        @test length(get_nanidxs(ds_rich)) == 18
        @test length(get_hasmissing(ds_rich)) == 18
        @test length(get_hasnans(ds_rich)) == 18
    end

    # ------------------------------------------------------------------ #
    #                     getter methods - by index                       #
    # ------------------------------------------------------------------ #
    @testset "Getter methods - by index" begin
        @test get_vnames(ds_rich, 1) == "str_col"
        @test get_vnames(ds_rich, 5) == "int_col"
        @test get_vnames(ds_rich, 11) == "ts1"
        @test get_vnames(ds_rich, [1, 5, 11]) == ["str_col", "int_col", "ts1"]

        @test get_datatype(ds_rich, [4, 5]) == [UInt32, Int]

        @test get_dims(ds_rich, [6, 11, 15]) == [0, 1, 2]

        @test get_missingidxs(ds_rich, [1, 4]) == [get_missingidxs(ds_rich, 1), get_missingidxs(ds_rich, 4)]

        @test get_nanidxs(ds_rich, [6, 9]) == [get_nanidxs(ds_rich, 6), get_nanidxs(ds_rich, 9)]

        @test get_valididxs(ds_rich, [10, 15]) == [get_valididxs(ds_rich, 10), get_valididxs(ds_rich, 15)]

        @test get_hasmissing(ds_rich, [13, 14]) == [get_hasmissing(ds_rich, 13), get_hasmissing(ds_rich, 14)]

        @test get_hasnans(ds_rich, [13, 14]) == [get_hasnans(ds_rich, 13), get_hasnans(ds_rich, 14)]
    end

    # ------------------------------------------------------------------ #
    #                         all clean dataset                           #
    # ------------------------------------------------------------------ #
    @testset "All clean dataset" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, 2, 3]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset, ["a", "b"])

        @test get_missingidxs(ds, 1) == Int[]
        @test get_missingidxs(ds, 2) == Int[]
        @test get_nanidxs(ds, 1) == Int[]
        @test get_nanidxs(ds, 2) == Int[]
        @test get_hasmissing(ds, 1) == Int[]
        @test get_hasnans(ds, 2) == Int[]
        @test get_valididxs(ds, 1) == [1, 2, 3]
        @test get_valididxs(ds, 2) == [1, 2, 3]
    end

    # ------------------------------------------------------------------ #
    #                           show methods                              #
    # ------------------------------------------------------------------ #
    @testset "show methods" begin
        # Test one-line show
        io = IOBuffer()
        show(io, ds_rich)
        output = String(take!(io))
        @test contains(output, "DatasetStructure(18 cols)")

        # Test multi-line show (text/plain)
        io = IOBuffer()
        show(io, MIME"text/plain"(), ds_rich)
        output = String(take!(io))
        @test contains(output, "DatasetStructure(18 columns)")
        @test contains(output, "datatypes by columns:")
        @test contains(output, "missing at:")
        @test contains(output, "NaN at:")
    end

    @testset "show method - clean dataset" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, 2, 3]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset, ["a", "b"])

        io = IOBuffer()
        show(io, MIME"text/plain"(), ds)
        output = String(take!(io))
        @test contains(output, "DatasetStructure(2 columns)")
        @test contains(output, "datatypes by columns:")
        @test !contains(output, "missing at:")
        @test !contains(output, "NaN at:")
    end

    @testset "show method - only missing, no NaN" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, missing, 3]
        dataset[:, 2] = ["a", "b", "c"]

        ds = DatasetStructure(dataset, ["a", "b"])

        io = IOBuffer()
        show(io, MIME"text/plain"(), ds)
        output = String(take!(io))
        @test contains(output, "missing at:")
        @test !contains(output, "NaN at:")
    end

    @testset "show method - only NaN, no missing" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1.0, NaN, 3.0]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset, ["a", "b"])

        io = IOBuffer()
        show(io, MIME"text/plain"(), ds)
        output = String(take!(io))
        @test !contains(output, "missing at:")
        @test contains(output, "NaN at:")
    end
end