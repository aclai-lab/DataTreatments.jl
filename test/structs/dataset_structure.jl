using Test
using DataTreatments

using DataFrames

@testset "DatasetStructure" begin
    
    @testset "Constructor" begin
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        @test length(ds) == 3
        
        @test_throws DimensionMismatch DatasetStructure(
            [Int64, Float64],
            [0, 0, 0],
            [[1, 2], [1, 2, 3], [1]],
            missingidxs, nanidxs, hasmissing, hasnans
        )
    end
    
    @testset "Size and length methods" begin
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test size(ds) == (3,)
        @test length(ds) == 3
        @test ndims(ds) == 1
    end
    
    @testset "Getter methods - full vectors" begin
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_datatype(ds) == datatype
        @test get_dims(ds) == dims
        @test get_valididxs(ds) == valididxs
        @test get_missingidxs(ds) == missingidxs
        @test get_nanidxs(ds) == nanidxs
        @test get_hasmissing(ds) == hasmissing
        @test get_hasnans(ds) == hasnans
    end
    
    @testset "Getter methods - by index" begin
        datatype = [Int64, Float64, String]
        dims = [0, 1, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_datatype(ds, 1) == Int64
        @test get_datatype(ds, 2) == Float64
        @test get_datatype(ds, 3) == String
        
        @test get_dims(ds, 1) == 0
        @test get_dims(ds, 2) == 1
        @test get_dims(ds, 3) == 0
        
        @test get_valididxs(ds, 1) == [1, 2]
        @test get_missingidxs(ds, 3) == [2]
        @test get_nanidxs(ds, 2) == [3]
    end
    
    @testset "Iteration support" begin
        datatype = [Int64, Float64, String]
        dims = [0, 1, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test collect(ds) == [Int64, Float64, String]
        @test collect(eachindex(ds)) == [1, 2, 3]
    end

    @testset "Dims handling - vector/array dimensions" begin
        datatype = [Float64, Vector{Float64}, Matrix{Float64}]
        dims = [0, 1, 2]
        valididxs = [[1, 2, 3], [1, 2], [1, 2, 3]]
        missingidxs = [Int[], [3], Int[]]
        nanidxs = [Int[], Int[], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], [2], [1]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_dims(ds, 1) == 0
        @test get_dims(ds, 2) == 1
        @test get_dims(ds, 3) == 2
        @test get_dims(ds) == [0, 1, 2]
    end

    @testset "get_dataset_structure - Matrix" begin
        dataset = Matrix{Any}(undef, 5, 3)
        dataset[:, 1] = [1, 2, missing, 4, 5]
        dataset[:, 2] = [1.0, NaN, 3.0, missing, 5.0]
        dataset[:, 3] = [collect(1.0:3.0), collect(2.0:4.0), collect(3.0:5.0), missing, NaN]

        ds = get_dataset_structure(dataset)

        @test length(ds) == 3

        # col 1: Int with one missing
        @test get_missingidxs(ds, 1) == [3]
        @test get_nanidxs(ds, 1) == Int[]
        @test get_valididxs(ds, 1) == [1, 2, 4, 5]

        # col 2: Float with one NaN and one missing
        @test get_missingidxs(ds, 2) == [4]
        @test get_nanidxs(ds, 2) == [2]
        @test get_valididxs(ds, 2) == [1, 3, 5]

        # col 3: Vector column with one missing and one top-level NaN
        @test get_missingidxs(ds, 3) == [4]
        @test get_nanidxs(ds, 3) == [5]
        @test get_valididxs(ds, 3) == [1, 2, 3]
        @test get_dims(ds, 3) == 1
    end

    @testset "get_dataset_structure - DataFrame" begin
        df = DataFrame(
            a = [1, 2, missing, 4],
            b = [1.0, NaN, 3.0, 4.0],
            c = [collect(1.0:3.0), collect(2.0:4.0), missing, NaN]
        )

        ds = get_dataset_structure(df)

        @test length(ds) == 3
        @test get_missingidxs(ds, 1) == [3]
        @test get_nanidxs(ds, 2) == [2]
        @test get_dims(ds, 3) == 1
        @test get_valididxs(ds, 3) == [1, 2]
    end

    @testset "show method" begin
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        io = IOBuffer()
        show(io, ds)
        output = String(take!(io))
        
        @test contains(output, "DatasetStructure(3 columns)")
        @test contains(output, "datatypes by columns:")
        @test contains(output, "missing at:")
        @test contains(output, "NaN at:")
    end
end