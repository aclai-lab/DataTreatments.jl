using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using CategoricalArrays

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

dt = DataTreatment(df)

df = DataFrame(
    str_col  = [missing, missing, "green", "red", "blue"],                     # AbstractString
    sym_col  = [missing, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical([missing, "medium", "large", "small", missing]),    # CategoricalValue
    uint_col = Union{Missing, UInt32}[missing, 2, 3, 4, 5],                    # UInt32
    int_col  = Union{Missing, Int}[missing, 20, 30, 40, 50]                    # Int
)

dt = DataTreatment(df)

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

# @testset "Indexing and Size Operations" begin
#     df = DataFrame(
#         col1 = ["a", "b", "c"],
#         col2 = ["x", "y", "z"]
#     )
    
#     @testset "Basic indexing" begin
#         dt = DataTreatment(df)
        
#         @test size(dt) == size(dt.Xtd)
#         @test size(dt, 1) == 3
#         @test size(dt, 2) == 2
#         @test length(dt.td_feats) == 2
#     end
# end

# @testset "Unsupervised Dataset (no y)" begin
#     df = DataFrame(
#         col1 = ["a", "b", "c"],
#         col2 = [1, 2, 3]
#     )
    
#     @testset "Default unsupervised" begin
#         dt = DataTreatment(df)
        
#         @test length(dt.y) == 3
#         @test all(ismissing, dt.y)
#     end
# end
