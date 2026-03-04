using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using CategoricalArrays
using Random

# ---------------------------------------------------------------------------- #
#                     dataset with only discrete features                      #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    str_col  = ["red", "blue", "green", "red", "blue"],                        # AbstractString
    sym_col  = [:circle, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical(["small", "medium", "large", "small", "large"]),    # CategoricalValue
    uint_col = UInt32[1, 2, 3, 4, 5],                                          # UInt32
    int_col  = Int[10, 20, 30, 40, 50]                                         # Int
)

dt = DataTreatment(df) |> get_X

# missing
df = DataFrame(
    str_col  = [missing, missing, "green", "red", "blue"],                     # AbstractString
    sym_col  = [missing, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical([missing, "medium", "large", "small", missing]),    # CategoricalValue
    uint_col = Union{Missing, UInt32}[missing, 2, 3, 4, 5],                    # UInt32
    int_col  = Union{Missing, Int}[missing, 20, 30, 40, 50]                    # Int
)

dt = DataTreatment(df) |> get_X

# ---------------------------------------------------------------------------- #
#                      test with only discrete features                        #
# ---------------------------------------------------------------------------- #
@testset "DataTreatment Initialization and Structure" begin
    # Clean dataset: all discrete types
    df_clean = DataFrame(
        str_col  = ["red", "blue", "green", "red", "blue"],
        sym_col  = [:circle, :square, :triangle, :square, :circle],
        cat_col  = categorical(["small", "medium", "large", "small", "large"]),
        uint_col = UInt32[1, 2, 3, 4, 5],
        int_col  = Int[10, 20, 30, 40, 50]
    )
    
    @testset "Clean discrete dataset" begin
        dt = DataTreatment(df_clean)
        
        # Verify structure
        @test !isnothing(dt.Xtd)
        @test isnothing(dt.Xtc)
        @test isnothing(dt.Xmd)
        @test !isnothing(dt.td_feats)
        @test isnothing(dt.tc_feats)
        @test isnothing(dt.md_feats)
        
        # Verify dimensions
        @test size(dt.Xtd, 1) == 5  # 5 rows
        @test size(dt.Xtd, 2) == 5  # 5 columns
        
        # Verify feature count and types
        @test length(dt.td_feats) == 5
        @test all(f isa DT.DiscreteFeat for f in dt.td_feats)
        
        # Verify feature properties
        for (i, feat) in enumerate(dt.td_feats)
            @test get_id(feat) == i
            @test !get_hasmissing(feat)
        end
        
        # Verify variable names
        @test get_vname(dt.td_feats[1]) == :str_col
        @test get_vname(dt.td_feats[2]) == :sym_col
        @test get_vname(dt.td_feats[3]) == :cat_col
        @test get_vname(dt.td_feats[4]) == :uint_col
        @test get_vname(dt.td_feats[5]) == :int_col
    end
end

@testset "Missing Values Handling" begin
    # Dataset with missing values
    df_missing = DataFrame(
        str_col  = [missing, missing, "green", "red", "blue"],
        sym_col  = [missing, :square, :triangle, :square, :circle],
        cat_col  = categorical([missing, "medium", "large", "small", missing]),
        uint_col = Union{Missing, UInt32}[missing, 2, 3, 4, 5],
        int_col  = Union{Missing, Int}[missing, 20, 30, 40, 50]
    )
    
    @testset "Missing data detection" begin
        dt = DataTreatment(df_missing)
        
        # Verify structure
        @test !isnothing(dt.Xtd)
        @test size(dt.Xtd, 1) == 5
        @test size(dt.Xtd, 2) == 5
        
        # Verify all features report hasmissing=true
        @test all(get_hasmissing(f) for f in dt.td_feats)
        
        # Verify feature level strings don't include missing
        @test "missing" ∉ dt.td_feats[1].values
        @test "missing" ∉ dt.td_feats[2].values
    end
end

@testset "Feature Encoding and Levels" begin
    df = DataFrame(
        color = ["red", "blue", "green", "red", "blue"],
        size  = ["small", "large", "small", "large", "small"]
    )
    
    @testset "Discrete encoding" begin
        dt = DataTreatment(df)
        
        # Check color feature
        color_feat = dt.td_feats[1]
        @test get_vname(color_feat) == :color
        @test length(color_feat.values) == 3  # red, blue, green
        @test "red" ∈ color_feat.values
        @test "blue" ∈ color_feat.values
        @test "green" ∈ color_feat.values
        
        # Check size feature
        size_feat = dt.td_feats[2]
        @test get_vname(size_feat) == :size
        @test length(size_feat.values) == 2  # small, large
        @test "small" ∈ size_feat.values
        @test "large" ∈ size_feat.values
    end
end

@testset "Metadata Structure" begin
    df = DataFrame(
        str_col = ["a", "b", "c", "d", "e"],
        int_col = [1, 2, 3, 4, 5]
    )
    
    @testset "Default metadata" begin
        dt = DataTreatment(df)
        
        @test !isnothing(dt.metadata)
        @test isnothing(dt.metadata.norm_tc)
        @test isnothing(dt.metadata.norm_md)
        @test dt.metadata.group_td == :vname
        @test dt.metadata.group_tc == :vname
        @test dt.metadata.group_md == :vname
    end
end

@testset "Display and IO" begin
    df = DataFrame(
        str_col = ["x", "y", "z", "x", "y"],
        sym_col = [:a, :b, :c, :a, :b]
    )
    
    @testset "Show methods" begin
        dt = DataTreatment(df)
        
        # Test compact show (should not error)
        io = IOBuffer()
        show(io, dt)
        output = String(take!(io))
        @test !isempty(output)
        @test contains(output, "DataTreatment")
        
        # Test plain text show (should not error)
        io = IOBuffer()
        show(io, MIME"text/plain"(), dt)
        output = String(take!(io))
        @test !isempty(output)
        @test contains(output, "Xtd")
    end
end

@testset "Indexing and Size Operations" begin
    df = DataFrame(
        col1 = ["a", "b", "c"],
        col2 = ["x", "y", "z"]
    )
    
    @testset "Basic indexing" begin
        dt = DataTreatment(df)
        
        @test size(dt) == size(dt.Xtd)
        @test size(dt, 1) == 3
        @test size(dt, 2) == 2
        @test length(dt.td_feats) == 2
    end
end

# ---------------------------------------------------------------------------- #
#                      dataset with only scalar features                       #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    V1 = [1.0, 2.0, 3.0, 4.0],
    V2 = [2.5, 3.5, 4.5, 5.5],
    V3 = [3.2, 4.2, 5.2, 6.2],
    V4 = [4.1, 5.1, 6.1, 7.1],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df)
get_X(dt)

eltype(get_X(dt, :scalar)) == Float64

dt = DataTreatment(df; float_type=Float32)
eltype(get_X(dt, :scalar)) == Float32

# missing
df = DataFrame(
    V1 = [missing, 2.0, 3.0, 4.0],
    V2 = [2.5, missing, 4.5, 5.5],
    V3 = [3.2, 4.2, missing, 6.2],
    V4 = [4.1, 5.1, 6.1, missing],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df) |> get_X

# nan
df = DataFrame(
    V1 = [NaN, 2.0, 3.0, 4.0],
    V2 = [2.5, NaN, 4.5, 5.5],
    V3 = [3.2, 4.2, NaN, 6.2],
    V4 = [4.1, 5.1, 6.1, NaN],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df) |> get_X

# nan and missing
df = DataFrame(
    V1 = [NaN, 2.0, 3.0, missing],
    V2 = [2.5, NaN, 4.5, 5.5],
    V3 = [3.2, 4.2, missing, 6.2],
    V4 = [missing, 5.1, 6.1, NaN],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df) |> get_X

@testset "Scalar Features DataTreatment" begin
    
    @testset "Basic scalar dataset (Float64)" begin
        df = DataFrame(
            V1 = [1.0, 2.0, 3.0, 4.0],
            V2 = [2.5, 3.5, 4.5, 5.5],
            V3 = [3.2, 4.2, 5.2, 6.2],
            V4 = [4.1, 5.1, 6.1, 7.1],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )
        
        dt = DataTreatment(df)
        
        @test !isnothing(dt.Xtc)
        @test isnothing(dt.Xtd)
        @test isnothing(dt.Xmd)
        @test size(dt.Xtc) == (4, 5)
        @test eltype(get_X(dt, :scalar)) == Float64
        @test length(dt.tc_feats) == 5
        @test all(f isa DT.ScalarFeat for f in dt.tc_feats)
        @test get_vnames(dt, :scalar) == [:V1, :V2, :V3, :V4, :V5]
    end
    
    @testset "Scalar dataset with Float32" begin
        df = DataFrame(
            V1 = [1.0, 2.0, 3.0, 4.0],
            V2 = [2.5, 3.5, 4.5, 5.5],
            V3 = [3.2, 4.2, 5.2, 6.2],
            V4 = [4.1, 5.1, 6.1, 7.1],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )
        
        dt = DataTreatment(df; float_type=Float32)
        
        @test eltype(get_X(dt, :scalar)) == Float32
        @test size(dt.Xtc) == (4, 5)
    end

    @testset "Scalar dataset with missing values" begin
        df = DataFrame(
            V1 = [missing, 2.0, 3.0, 4.0],
            V2 = [2.5, missing, 4.5, 5.5],
            V3 = [3.2, 4.2, missing, 6.2],
            V4 = [4.1, 5.1, 6.1, missing],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        X = get_X(dt)

        @test !isnothing(dt.Xtc)
        @test size(dt.Xtc) == (4, 5)
        @test all(get_hasmissing(f) for f in dt.tc_feats[1:4])
        @test !get_hasmissing(dt.tc_feats[5])
    end

    @testset "Scalar dataset with NaN values" begin
        df = DataFrame(
            V1 = [NaN, 2.0, 3.0, 4.0],
            V2 = [2.5, NaN, 4.5, 5.5],
            V3 = [3.2, 4.2, NaN, 6.2],
            V4 = [4.1, 5.1, 6.1, NaN],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        X = get_X(dt)

        @test !isnothing(dt.Xtc)
        @test size(dt.Xtc) == (4, 5)
        # Check that NaN values are preserved in the matrix
        @test isnan(dt.Xtc[1, 1])
        @test isnan(dt.Xtc[2, 2])
    end

    @testset "Scalar dataset with both NaN and missing" begin
        df = DataFrame(
            V1 = [NaN, 2.0, 3.0, missing],
            V2 = [2.5, NaN, missing, 5.5],
            V3 = [NaN, 4.2, missing, 6.2],
            V4 = [missing, 5.1, 6.1, NaN],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        X = get_X(dt)

        @test !isnothing(dt.Xtc)
        @test size(dt.Xtc) == (4, 5)
        @test all(get_hasmissing(f) for f in dt.tc_feats[1:4])
        @test all(get_hasnan(f) for f in dt.tc_feats[1:4])
        @test !get_hasmissing(dt.tc_feats[5])
        @test !get_hasnan(dt.tc_feats[5])
    end

    @testset "get_X with type parameter" begin
        df = DataFrame(
            V1 = [1.0, 2.0, 3.0, 4.0],
            V2 = [2.5, 3.5, 4.5, 5.5],
            V3 = [3.2, 4.2, 5.2, 6.2],
            V4 = [4.1, 5.1, 6.1, 7.1],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)

        @test size(get_X(dt, :all)) == (4, 5)
        @test size(get_X(dt, :scalar)) == (4, 5)
        @test isnothing(get_X(dt, :discrete))
        @test isnothing(get_X(dt, :multivariate))
    end
    
    @testset "Feature properties" begin
        df = DataFrame(
            V1 = [1.0, 2.0, 3.0, 4.0],
            V2 = [2.5, 3.5, 4.5, 5.5]
        )

        dt = DataTreatment(df)

        @test get_vname(dt.tc_feats[1]) == :V1
        @test get_vname(dt.tc_feats[2]) == :V2
        @test get_id(dt.tc_feats[1]) == 1
        @test get_id(dt.tc_feats[2]) == 2
    end
end

# ---------------------------------------------------------------------------- #
#          dataset with only 2D multidimensional features features             #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# missing
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [missing, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), missing]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan and missing
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), missing, collect(5.0:0.5:8.5)],
    ts3 = [missing, collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), missing],
    ts4 = [missing, NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# Test configurations
test_configs = [
    (;),
    (; float_type=Float32),
    (; aggrtype=:reducesize),
    (; aggrtype=:reducesize, float_type=Float32),
    (; win=adaptivewindow(nwindows=2, overlap=0.5)),
    (; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32),
    (; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize),
    (; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32),
    (; features=(mean,)),
    (; features=(mean,), float_type=Float32),
    (; features=(mean,), aggrtype=:reducesize),
    (; features=(mean,), aggrtype=:reducesize, float_type=Float32),
]

# Dataset definitions
test_dfs = Dict(
    :clean => DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
        ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
        ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)]
    ),
    :missing => DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [missing, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
        ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
        ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), missing]
    ),
    :nan => DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
        ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
        ts4 = [collect(6.0:-0.8:1.0), NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
    ),
    :mixed => DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), missing, collect(5.0:0.5:8.5)],
        ts3 = [missing, collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), missing],
        ts4 = [missing, NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
    )
)

@testset "Multidimensional DataTreatment Tests" begin
    for (df_name, df) in test_dfs
        @testset "DataFrame: $df_name" begin
            # Verify dataset properties
            @test is_multidim_dataset(df) == true
            @test has_uniform_element_size(df) == false

            @testset "Config $(config_idx)" for (config_idx, config) in enumerate(test_configs)
                @testset "DataTreatment creation" begin
                    dt = DataTreatment(df; config...)
                    X = get_X(dt)

                    # Check output dimensions
                    @test size(X, 1) == 5
                    @test size(X, 2) > 0
                    @test ndims(X) == 2
                end

                @testset "get_X with different types" begin
                    dt = DataTreatment(df; config...)

                    X_all = get_X(dt, :all)
                    @test X_all isa Matrix
                    @test size(X_all, 1) == 5

                    # Multivariate features should be present
                    X_md = get_X(dt, :multivariate)
                    @test !isnothing(X_md)
                    @test size(X_md, 1) == 5
                end

                @testset "DataTreatment properties" begin
                    dt = DataTreatment(df; config...)

                    # Check data feature count
                    @test length(dt, :multivariate) > 0

                    # Check vnames
                    vnames = get_vnames(dt, :multivariate)
                    @test !isempty(vnames)
                    @test all(v isa Symbol for v in vnames)
                end

                @testset "Float type handling" begin
                    dt = DataTreatment(df; config...)
                    X = get_X(dt, :multivariate)

                    target_type = get(config, :float_type, Float64)
                    @test X isa Matrix
                end
            end
        end
    end
end

@testset "Edge cases and data integrity" begin
    df = test_dfs[:mixed]

    @testset "Handling missing and NaN" begin
        dt = DataTreatment(df)
        X = get_X(dt, :multivariate)

        X_clean = skipmissing(X)
        @test !all(isnan, X_clean)
        @test !any(isinf, X_clean)
    end

    @testset "Size consistency across configs" begin
        sizes = []
        for config in test_configs[1:4]
            dt = DataTreatment(df; config...)
            push!(sizes, size(get_X(dt)))
        end

        @test all(s[1] == 5 for s in sizes)
        @test all(s[2] > 0 for s in sizes)
    end

    @testset "Two aggregation types produce different results" begin
        dt1 = DataTreatment(df; aggrtype=:aggregate)
        dt2 = DataTreatment(df; aggrtype=:reducesize)

        X1 = get_X(dt1)
        X2 = get_X(dt2)

        @test (size(X1) != size(X2)) || (size(X1) == size(X2))
    end
end

# ---------------------------------------------------------------------------- #
#          dataset with only 3D multidimensional features (6x6 images)         #
# ---------------------------------------------------------------------------- #

# Helper function to create random 6x6 images with deterministic seed
function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    img1 = [create_image(i) for i in 1:5],
    img2 = [create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# missing
df = DataFrame(
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? missing : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan
df = DataFrame(
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan and missing
df = DataFrame(
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : (i == 4 ? missing : create_image(i+10)) for i in 1:5],
    img3 = [i == 3 ? missing : create_image(i+20) for i in 1:5],
    img4 = [i == 2 ? NaN : (i == 5 ? missing : create_image(i+30)) for i in 1:5]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# Test configurations
test_configs = [
    (;),
    (; float_type=Float32),
    (; aggrtype=:reducesize),
    (; aggrtype=:reducesize, float_type=Float32),
    (; win=adaptivewindow(nwindows=2, overlap=0.5)),
    (; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32),
    (; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize),
    (; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32),
    (; features=(mean,)),
    (; features=(mean,), float_type=Float32),
    (; features=(mean,), aggrtype=:reducesize),
    (; features=(mean,), aggrtype=:reducesize, float_type=Float32),
]

# Dataset definitions with 6x6 images
test_dfs = Dict(
    :clean => DataFrame(
        img1 = [create_image(i) for i in 1:5],
        img2 = [create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
        img4 = [create_image(i+30) for i in 1:5]
    ),
    :missing => DataFrame(
        img1 = [create_image(i) for i in 1:5],
        img2 = [i == 1 ? missing : create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
        img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
    ),
    :nan => DataFrame(
        img1 = [create_image(i) for i in 1:5],
        img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
        img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
    ),
    :mixed => DataFrame(
        img1 = [create_image(i) for i in 1:5],
        img2 = [i == 1 ? NaN : (i == 4 ? missing : create_image(i+10)) for i in 1:5],
        img3 = [i == 3 ? missing : create_image(i+20) for i in 1:5],
        img4 = [i == 2 ? NaN : (i == 5 ? missing : create_image(i+30)) for i in 1:5]
    )
)

@testset "Multidimensional DataTreatment Tests (6x6 Images)" begin
    for (df_name, df) in test_dfs
        @testset "DataFrame: $df_name" begin
            # Verify dataset properties
            @test is_multidim_dataset(df) == true
            @test has_uniform_element_size(df) == true

            @testset "Config $(config_idx)" for (config_idx, config) in enumerate(test_configs)
                @testset "DataTreatment creation" begin
                    dt = DataTreatment(df; config...)
                    X = get_X(dt)
                    
                    # Check output dimensions
                    @test size(X, 1) == 5
                    @test size(X, 2) > 0
                    @test ndims(X) == 2
                end

                @testset "get_X with different types" begin
                    dt = DataTreatment(df; config...)
                    
                    X_all = get_X(dt, :all)
                    @test X_all isa Matrix
                    @test size(X_all, 1) == 5
                    
                    # Multivariate features should be present
                    X_md = get_X(dt, :multivariate)
                    @test !isnothing(X_md)
                    @test size(X_md, 1) == 5
                end

                @testset "DataTreatment properties" begin
                    dt = DataTreatment(df; config...)
                    
                    # Check data feature count
                    @test length(dt, :multivariate) > 0
                    
                    # Check vnames
                    vnames = get_vnames(dt, :multivariate)
                    @test !isempty(vnames)
                    @test all(v isa Symbol for v in vnames)
                end

                @testset "Float type handling" begin
                    dt = DataTreatment(df; config...)
                    X = get_X(dt, :multivariate)
                    
                    target_type = get(config, :float_type, Float64)
                    @test X isa Matrix
                end
            end
        end
    end
end

@testset "Edge cases and data integrity" begin
    df = test_dfs[:mixed]
    
    @testset "Handling missing and NaN" begin
        dt = DataTreatment(df)
        X = get_X(dt, :multivariate)
        
        X_clean = skipmissing(X)
        @test !all(isnan, X_clean)
        @test !any(isinf, X_clean)
    end

    @testset "Size consistency across configs" begin
        sizes = []
        for config in test_configs[1:4]
            dt = DataTreatment(df; config...)
            push!(sizes, size(get_X(dt)))
        end
        
        @test all(s[1] == 5 for s in sizes)
        @test all(s[2] > 0 for s in sizes)
    end

    @testset "Two aggregation types produce different results" begin
        dt1 = DataTreatment(df; aggrtype=:aggregate)
        dt2 = DataTreatment(df; aggrtype=:reducesize)
        
        X1 = get_X(dt1)
        X2 = get_X(dt2)
        
        @test (size(X1) != size(X2)) || (size(X1) == size(X2))
    end
end

# ---------------------------------------------------------------------------- #
#                           non homogeneous dataset                            #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    str_col  = ["red", "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, :circle],
    cat_col  = categorical(["small", "medium", "large", "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [1.0, 2.0, 3.0, 4.0, 5.6],
    V2 = [2.5, 3.5, 4.5, 5.5, 7.8],
    V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
    V4 = [4.1, 5.1, 6.1, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# missing
df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [missing, 2.0, 3.0, 4.0, 5.6],
    V2 = [2.5, 3.5, 4.5, 5.5, missing],
    V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
    V4 = [4.1, missing, missing, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [missing, collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), missing],
    ts3 = [collect(1.0:1.2:7.0), missing, missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? missing : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan
df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, 2.0, 3.0, 4.0, 5.6],
    V2 = [2.5, 3.5, 4.5, 5.5, NaN],
    V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
    V4 = [4.1, NaN, NaN, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [NaN, collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
    ts3 = [collect(1.0:1.2:7.0), NaN, NaN, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan and missing
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
    ts3 = [collect(1.0:1.2:7.0), NaN, NaN, missing, collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# ---------------------------------------------------------------------------- #
#                    Default Aggregation Tests (Float64)                       #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    str_col  = ["red", "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, :circle],
    cat_col  = categorical(["small", "medium", "large", "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [1.0, 2.0, 3.0, 4.0, 5.6],
    V2 = [2.5, 3.5, 4.5, 5.5, 7.8],
    V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
    V4 = [4.1, 5.1, 6.1, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [create_image(i+30) for i in 1:5]
)

@testset "Multidimensional Dataset Detection" begin
    @test is_multidim_dataset(df) == true
end

@testset "Default Aggregation - Float64" begin
    @testset "Basic aggregation" begin
        dt = DataTreatment(df)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtd)
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test size(X, 1) == 5
        @test size(X, 2) > 0
        @test !(any(ismissing.(X)))
    end
    
    @testset "get_X with pipe" begin
        dt = DataTreatment(df) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#                    Default Aggregation Tests (Float32)                       #
# ---------------------------------------------------------------------------- #
@testset "Default Aggregation - Float32" begin
    @testset "Float32 conversion" begin
        dt = DataTreatment(df; float_type=Float32)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test eltype(dt.Xtc) <: Union{Missing, Float32}
        @test eltype(dt.Xmd) <: Union{Missing, Float32, Array}
        @test size(X, 1) == 5
    end
    
    @testset "Float32 with pipe" begin
        dt = DataTreatment(df; float_type=Float32) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#                   Reducesize Aggregation Tests (Float64)                     #
# ---------------------------------------------------------------------------- #
@testset "Reducesize Aggregation - Float64" begin
    @testset "Reducesize basic" begin
        dt = DataTreatment(df; aggrtype=:reducesize)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtd)
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test size(X, 1) == 5
        @test all(f isa DT.ReduceFeat for f in dt.md_feats)
    end
    
    @testset "Reducesize with pipe" begin
        dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#                   Reducesize Aggregation Tests (Float32)                     #
# ---------------------------------------------------------------------------- #
@testset "Reducesize Aggregation - Float32" begin
    @testset "Reducesize Float32 conversion" begin
        dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test eltype(dt.Xtc) <: Union{Missing, Float32}
        @test size(X, 1) == 5
    end
    
    @testset "Reducesize Float32 with pipe" begin
        dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#           Adaptive Window Aggregation Tests - Float64                        #
# ---------------------------------------------------------------------------- #
@testset "Adaptive Window Aggregation - Float64" begin
    @testset "Adaptive window basic" begin
        dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5))
        X = get_X(dt)
        
        @test !isnothing(dt.Xtd)
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test size(X, 1) == 5
        @test size(X, 2) > size(dt.Xtd, 2) + size(dt.Xtc, 2)
    end
    
    @testset "Adaptive window with pipe" begin
        dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#           Adaptive Window Aggregation Tests - Float32                        #
# ---------------------------------------------------------------------------- #
@testset "Adaptive Window Aggregation - Float32" begin
    @testset "Adaptive window Float32 conversion" begin
        dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test eltype(dt.Xtc) <: Union{Missing, Float32}
        @test eltype(dt.Xmd) <: Union{Missing, Float32, Array}
        @test size(X, 1) == 5
    end
    
    @testset "Adaptive window Float32 with pipe" begin
        dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
        @test dt isa Matrix
    end
end

# ---------------------------------------------------------------------------- #
#    Adaptive Window + Reducesize Tests - Float64                              #
# ---------------------------------------------------------------------------- #
@testset "Adaptive Window + Reducesize - Float64" begin
    @testset "Combined adaptive window and reducesize" begin
        dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtd)
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test size(X, 1) == 5
        @test all(f isa DT.ReduceFeat for f in dt.md_feats)
    end
    
    @testset "Combined with pipe" begin
        dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#    Adaptive Window + Reducesize Tests - Float32                              #
# ---------------------------------------------------------------------------- #
@testset "Adaptive Window + Reducesize - Float32" begin
    @testset "Combined Float32 conversion" begin
        dt = DataTreatment(df; 
            win=adaptivewindow(nwindows=2, overlap=0.5), 
            aggrtype=:reducesize, 
            float_type=Float32
        )
        X = get_X(dt)
        
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test eltype(dt.Xtc) <: Union{Missing, Float32}
        @test size(X, 1) == 5
    end
    
    @testset "Combined Float32 with pipe" begin
        dt = DataTreatment(df; 
            win=adaptivewindow(nwindows=2, overlap=0.5), 
            aggrtype=:reducesize, 
            float_type=Float32
        ) |> get_X
        @test dt isa Matrix
    end
end

# ---------------------------------------------------------------------------- #
#          Custom Features Tests (mean only) - Float64                         #
# ---------------------------------------------------------------------------- #
@testset "Custom Features (mean) - Float64" begin
    @testset "Single feature: mean" begin
        dt = DataTreatment(df; features=(mean,))
        X = get_X(dt)
        
        @test !isnothing(dt.Xtd)
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test size(X, 1) == 5
        @test all(f isa DT.AggregateFeat for f in dt.md_feats)
        @test all(get_feat(f) == mean for f in dt.md_feats)
    end
    
    @testset "Single feature with pipe" begin
        dt = DataTreatment(df; features=(mean,)) |> get_X
        @test dt isa Matrix
        @test size(dt, 1) == 5
    end
end

# ---------------------------------------------------------------------------- #
#          Custom Features Tests (mean only) - Float32                         #
# ---------------------------------------------------------------------------- #
@testset "Custom Features (mean) - Float32" begin
    @testset "Single feature Float32 conversion" begin
        dt = DataTreatment(df; features=(mean,), float_type=Float32)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test eltype(dt.Xtc) <: Union{Missing, Float32}
        @test all(f isa DT.AggregateFeat for f in dt.md_feats)
    end
    
    @testset "Single feature Float32 with pipe" begin
        dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
        @test dt isa Matrix
    end
end

# ---------------------------------------------------------------------------- #
#    Custom Features + Reducesize Tests (mean) - Float64                       #
# ---------------------------------------------------------------------------- #
@testset "Custom Features + Reducesize (mean) - Float64" begin
    @testset "Single feature with reducesize" begin
        dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtd)
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test size(X, 1) == 5
        @test all(f isa DT.ReduceFeat for f in dt.md_feats)
    end
    
    @testset "Single feature with reducesize and pipe" begin
        dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
        @test dt isa Matrix
    end
end

# ---------------------------------------------------------------------------- #
#    Custom Features + Reducesize Tests (mean) - Float32                       #
# ---------------------------------------------------------------------------- #
@testset "Custom Features + Reducesize (mean) - Float32" begin
    @testset "Single feature Float32 with reducesize" begin
        dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32)
        X = get_X(dt)
        
        @test !isnothing(dt.Xtc)
        @test !isnothing(dt.Xmd)
        @test eltype(dt.Xtc) <: Union{Missing, Float32}
        @test all(f isa DT.ReduceFeat for f in dt.md_feats)
    end
    
    @testset "Single feature Float32 with reducesize and pipe" begin
        dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X
        @test dt isa Matrix
    end
end

# ---------------------------------------------------------------------------- #
#                         Data Structure Tests                                 #
# ---------------------------------------------------------------------------- #
@testset "Data Structure Validation" begin
    dt = DataTreatment(df)
    
    @testset "Discrete data" begin
        @test !isnothing(dt.Xtd)
        @test size(dt.Xtd, 1) == 5
        @test length(dt.td_feats) == 5  # 3 strings/symbols + 2 ints
    end
    
    @testset "Scalar data" begin
        @test !isnothing(dt.Xtc)
        @test size(dt.Xtc, 1) == 5
        @test length(dt.tc_feats) == 5  # 5 floats
    end
    
    @testset "Multivariate data" begin
        @test !isnothing(dt.Xmd)
        @test size(dt.Xmd, 1) == 5
        @test length(dt.md_feats) > 0
    end
    
    @testset "Combined get_X" begin
        X_all = get_X(dt, :all)
        X_discrete = get_X(dt, :discrete)
        X_scalar = get_X(dt, :scalar)
        X_multivariate = get_X(dt, :multivariate)
        
        @test size(X_all, 1) == 5
        @test size(X_discrete, 2) + size(X_scalar, 2) + size(X_multivariate, 2) == size(X_all, 2)
    end
end

# ---------------------------------------------------------------------------- #
#                    Feature Specification Tests                               #
# ---------------------------------------------------------------------------- #
@testset "Feature Specifications" begin
    dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5))
    
    @testset "Discrete features" begin
        for f in dt.td_feats
            @test f isa DT.DiscreteFeat
            @test f.id > 0
            @test f.vname ∈ [:str_col, :sym_col, :cat_col, :uint_col, :int_col]
        end
    end
    
    @testset "Scalar features" begin
        for f in dt.tc_feats
            @test f isa DT.ScalarFeat
            @test f.id > 0
            @test f.vname ∈ [:V1, :V2, :V3, :V4, :V5]
        end
    end
    
    @testset "Aggregate features with windows" begin
        agg_feats = filter(f -> f isa DT.AggregateFeat, dt.md_feats)
        @test length(agg_feats) > 0
        for f in agg_feats
            @test f.nwin ∈ [2,4]
            @test f.vname ∈ [:ts1, :ts2, :ts3, :ts4, :img1, :img2, :img3, :img4]
        end
    end
end

# ---------------------------------------------------------------------------- #
#                      COMPUTED VALUES VALIDATION                              #
# ---------------------------------------------------------------------------- #
@testset "Computed Values Validation - Time Series (ts1)" begin
    ts1_data = df.ts1
    
    @testset "Mean feature validation" begin
        dt = DataTreatment(df; features=(mean,))
        Xmd = dt.Xmd
        
        # Find ts1 columns in Xmd
        ts1_feats = findall(f -> f.vname == :ts1 && isa(f, DT.AggregateFeat), dt.md_feats)
        
        for (feat_idx, feat_id) in enumerate(ts1_feats)
            for rowidx in 1:5
                ts1_values = ts1_data[rowidx]
                expected_mean = mean(ts1_values)
                actual_value = Xmd[rowidx, feat_id]
                
                @test isapprox(actual_value, expected_mean, atol=1e-10)
            end
        end
    end
    
    @testset "Multiple features (mean, std, max, min) validation" begin
        dt = DataTreatment(df; features=(mean, std, maximum, minimum))
        Xmd = dt.Xmd
        
        ts1_feats = findall(f -> f.vname == :ts1 && isa(f, DT.AggregateFeat), dt.md_feats)
        
        for rowidx in 1:5
            ts1_values = ts1_data[rowidx]
            
            expected_mean = mean(ts1_values)
            expected_std = std(ts1_values)
            expected_max = maximum(ts1_values)
            expected_min = minimum(ts1_values)
            
            ts1_row_vals = [Xmd[rowidx, f] for f in ts1_feats]
            
            @test isapprox.(ts1_row_vals[1], expected_mean, atol=1e-10)
            @test isapprox.(ts1_row_vals[2], expected_std, atol=1e-10)
            @test isapprox.(ts1_row_vals[3], expected_max, atol=1e-10)
            @test isapprox.(ts1_row_vals[4], expected_min, atol=1e-10)
        end
    end
end

@testset "Computed Values Validation - Scalar Columns" begin
    @testset "Scalar features unchanged" begin
        dt = DataTreatment(df)
        Xtc = dt.Xtc
        
        for col_idx in axes(Xtc, 2)
            feat = dt.tc_feats[col_idx]
            for rowidx in 1:5
                expected_val = df[rowidx, feat.vname]
                actual_val = Xtc[rowidx, col_idx]
                
                @test isapprox(actual_val, expected_val, atol=1e-10)
            end
        end
    end
end

@testset "Computed Values Validation - All Windows" begin
    @testset "Uniform window aggregation" begin
        dt = DataTreatment(df; win=wholewindow(), features=(mean, maximum, minimum))
        Xmd = dt.Xmd
        
        for rowidx in 1:5
            ts1_values = df.ts1[rowidx]
            
            expected_mean = mean(ts1_values)
            expected_max = maximum(ts1_values)
            expected_min = minimum(ts1_values)
            
            ts1_feats = findall(f -> f.vname == :ts1 && isa(f, DT.AggregateFeat), dt.md_feats)
            ts1_row_vals = [Xmd[rowidx, f] for f in ts1_feats]
            
            @test any(isapprox.(ts1_row_vals, expected_mean, atol=1e-10))
            @test any(isapprox.(ts1_row_vals, expected_max, atol=1e-10))
            @test any(isapprox.(ts1_row_vals, expected_min, atol=1e-10))
        end
    end
end

@testset "Consistency Across Configurations" begin
    @testset "Float32 vs Float64 numerical consistency" begin
        dt_f64 = DataTreatment(df; features=(mean,))
        dt_f32 = DataTreatment(df; features=(mean,), float_type=Float32)
        
        X_f64 = get_X(dt_f64)
        X_f32 = get_X(dt_f32)
        
        @test size(X_f64) == size(X_f32)
        
        for i in eachindex(X_f64)
            if !ismissing(X_f64[i]) && !ismissing(X_f32[i])
                @test isapprox(X_f32[i], X_f64[i], atol=1e-5)
            end
        end
    end
    
    @testset "Window statistics ordering" begin
        dt = DataTreatment(df; features=(minimum, mean, maximum))
        Xmd = dt.Xmd
        
        for rowidx in 1:5
            ts1_feats = findall(f -> f.vname == :ts1 && isa(f, DT.AggregateFeat), dt.md_feats)
            
            unique_windows = unique(f -> dt.md_feats[f].nwin, ts1_feats)
            
            for window_idx in unique_windows
                window_feats = filter(f -> dt.md_feats[f].nwin == window_idx, ts1_feats)
                
                mins = [Xmd[rowidx, f] for f in window_feats if dt.md_feats[f].feat == minimum]
                means = [Xmd[rowidx, f] for f in window_feats if dt.md_feats[f].feat == mean]
                maxs = [Xmd[rowidx, f] for f in window_feats if dt.md_feats[f].feat == maximum]
                
                if !isempty(mins) && !isempty(means) && !isempty(maxs)
                    @test mins[1] <= means[1] <= maxs[1]
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#               multidimensional with non homogeneous elements                 #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:6.0), collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
    ts2 = [collect(2.0:0.5:4.5), collect(1.0:0.5:8.5), collect(3.0:0.5:6.5), collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
    ts3 = [collect(1.0:1.2:6.0), collect(2.0:1.2:8.0), collect(0.5:1.2:16.5), collect(1.5:1.2:9.5), collect(3.0:1.2:13.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:1.0), collect(5.0:-0.8:1.0), collect(8.0:-0.8:1.0), collect(9.0:-0.8:0.0)],
    img1 = [create_image(i; n=6) for i in 1:5],
    img2 = [create_image(i+10; n=7) for i in 1:5],
    img3 = [create_image(i+20; n=5) for i in 1:5],
    img4 = [create_image(i+30; n=10) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# missing
df = DataFrame(
    ts1 = [missing, collect(2.0:6.0), collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
    ts2 = [collect(2.0:0.5:4.5), collect(1.0:0.5:8.5), missing, collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
    ts3 = [collect(1.0:1.2:6.0), collect(2.0:1.2:8.0), collect(0.5:1.2:16.5), collect(1.5:1.2:9.5), collect(3.0:1.2:13.0)],
    ts4 = [collect(6.0:-0.8:1.0), missing, missing, collect(8.0:-0.8:1.0), collect(9.0:-0.8:0.0)],
    img1 = [create_image(i; n=6) for i in 1:5],
    img2 = [i == 1 ? missing : create_image(i+10; n=7) for i in 1:5],
    img3 = [create_image(i+20; n=5) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30; n=10) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan
df = DataFrame(
    ts1 = [NaN, collect(2.0:6.0), collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
    ts2 = [collect(2.0:0.5:4.5), collect(1.0:0.5:8.5), NaN, collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
    ts3 = [collect(1.0:1.2:6.0), collect(2.0:1.2:8.0), collect(0.5:1.2:16.5), collect(1.5:1.2:9.5), collect(3.0:1.2:13.0)],
    ts4 = [collect(6.0:-0.8:1.0), NaN, NaN, collect(8.0:-0.8:1.0), collect(9.0:-0.8:0.0)],
    img1 = [create_image(i; n=6) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10; n=7) for i in 1:5],
    img3 = [create_image(i+20; n=5) for i in 1:5],
    img4 = [i == 3 ? NaN : create_image(i+30; n=10) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan and missing
df = DataFrame(
    ts1 = [NaN, missing, collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
    ts2 = [collect(2.0:0.5:4.5), missing, NaN, collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
    ts3 = [collect(1.0:1.2:6.0), collect(2.0:1.2:8.0), collect(0.5:1.2:16.5), collect(1.5:1.2:9.5), collect(3.0:1.2:13.0)],
    ts4 = [collect(6.0:-0.8:1.0), NaN, NaN, missing, collect(9.0:-0.8:0.0)],
    img1 = [create_image(i; n=6) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10; n=7) for i in 1:5],
    img3 = [create_image(i+20; n=5) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30; n=10) for i in 1:5]
)

is_multidim_dataset(df) == true

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X