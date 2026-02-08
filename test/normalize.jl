using Test
using DataTreatments
const DT = DataTreatments

using Normalization
using Statistics

# ---------------------------------------------------------------------------- #
#                                 normalization                                #
# ---------------------------------------------------------------------------- #
X = Float64.([8 1 6; 3 5 7; 4 9 2])
nfunc = DT.minmax()

all_elements = DT.normalize(X, nfunc)
@test all_elements == [
    0.875 0.0 0.625;
    0.25 0.5 0.75;
    0.375 1.0 0.125
]

grouby_cols = DT.normalize(X, nfunc; tabular=true)
@test grouby_cols == [
    1.0 0.0 0.8;
    0.0 0.5 1.0;
    0.2 1.0 0.0
]

grouby_rows = DT.normalize(X, nfunc; tabular=true, dim=:row)
@test isapprox(grouby_rows, [
    1.0 0.0 0.714285714;
    0.0 0.5 1.0;
    0.285714286 1.0 0.0
])

Xmatrix = [rand(1:100, 4, 2) for _ in 1:10, _ in 1:5]
nfunc = DT.zscore()

@test_nowarn DT.normalize(Xmatrix, nfunc)

@test_nowarn DT.normalize(Xmatrix, nfunc; tabular=true)

@test_nowarn DT.normalize(Xmatrix, nfunc; tabular=true, dim=:row)


X = rand(100, 12)
Xmatrix = fill(X, 50, 10)
nfunc = zscore()

@btime DT.normalize(Xmatrix, nfunc);
# 3.546 ms (1503 allocations: 4.62 MiB)

# ---------------------------------------------------------------------------- #
#                             tabular normalization                            #
# ---------------------------------------------------------------------------- #
a = [8 1 6; 3 5 7; 4 9 2]

# test values verified against MATLAB
zscore_norm = DT.normalize(a, DT.zscore(); tabular=true, dim=:col)
@test isapprox(zscore_norm, [1.13389 -1.0 0.377964; -0.755929 0.0 0.755929; -0.377964 1.0 -1.13389], atol=1e-5)

zscore_row = DT.normalize(a, DT.zscore(); tabular=true, dim=:row)
@test isapprox(zscore_row, [0.83205 -1.1094 0.27735; -1.0 0.0 1.0; -0.27735 1.1094 -0.83205], atol=1e-5)

zscore_robust = DT.normalize(a, DT.zscore(method=:robust); tabular=true, dim=:col)
@test zscore_robust == [4.0 -1.0 0.0; -1.0 0.0 1.0; 0.0 1.0 -4.0]

zscore_half = DT.normalize(a, DT.zscore(method=:half); tabular=true, dim=:col)

@test_throws ArgumentError DT.normalize(a, DT.zscore(); tabular=true, dim=:invalid)
@test_throws ArgumentError DT.normalize(a, DT.zscore(method=:invalid))

@test_nowarn DT.normalize(a, sigmoid(); tabular=true, dim=:col)

norm_norm = DT.normalize(a, pnorm(); tabular=true, dim=:col)
@test isapprox(norm_norm, [0.847998 0.0966736 0.635999; 0.317999 0.483368 0.741999; 0.423999 0.870063 0.212], atol=1e-6)

norm_norm = DT.normalize(a, pnorm(p=4); tabular=true, dim=:col)
@test isapprox(norm_norm, [0.980428 0.108608 0.768635; 0.36766 0.543042 0.896741; 0.490214 0.977475 0.256212], atol=1e-5)

norm_norm = DT.normalize(a, pnorm(p=Inf); tabular=true, dim=:col)
@test isapprox(norm_norm, [1.0 0.111111 0.857143; 0.375 0.555556 1.0; 0.5 1.0 0.285714], atol=1e-6)

scale_norm = DT.normalize(a, scale(factor=:std); tabular=true, dim=:col)
@test isapprox(scale_norm, [3.02372 0.25 2.26779; 1.13389 1.25 2.64575; 1.51186 2.25 0.755929], atol=1e-5)

scale_norm = DT.normalize(a, scale(factor=:mad); tabular=true, dim=:col)
@test scale_norm == [8.0 0.25 6.0; 3.0 1.25 7.0; 4.0 2.25 2.0]

scale_norm = DT.normalize(a, scale(factor=:first); tabular=true, dim=:col)
@test isapprox(scale_norm, [1.0 1.0 1.0; 0.375 5.0 1.16667; 0.5 9.0 0.333333], atol=1e-5)

scale_norm = DT.normalize(a, scale(factor=:iqr); tabular=true, dim=:col)

minmax_norm = DT.normalize(a, DT.minmax(); tabular=true, dim=:col)
@test minmax_norm == [1.0 0.0 0.8; 0.0 0.5 1.0; 0.2 1.0 0.0]

minmax_norm = DT.normalize(a, DT.minmax(lower=-2, upper=4); tabular=true, dim=:col)
@test minmax_norm == [4.0 -2.0 2.8; -2.0 1.0 4.0; -0.8 4.0 -2.0]

center_norm = DT.normalize(a, center(); tabular=true, dim=:col)
@test center_norm == [3.0 -4.0 1.0; -2.0 0.0 2.0; -1.0 4.0 -3.0]

center_norm = DT.normalize(a, center(method=:median); tabular=true, dim=:col)
@test center_norm == [4.0 -4.0 0.0; -1.0 0.0 1.0; 0.0 4.0 -4.0]

@test_nowarn DT.normalize(a, unitpower(); tabular=true, dim=:col)

@test_nowarn DT.normalize(a, outliersuppress(); tabular=true, dim=:col)
@test_nowarn DT.normalize(a, outliersuppress(thr=3); tabular=true, dim=:col)

# test against julia package Normalization
X = rand(200,100)

test = DT.normalize(X, DT.zscore(); tabular=true, dim=:col)
n = fit(ZScore, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, DT.zscore(method=:half); tabular=true, dim=:col)
n = fit(HalfZScore, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, sigmoid(); tabular=true, dim=:col)
n = fit(Sigmoid, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, pnorm(); tabular=true, dim=:col)
n = fit(UnitEnergy, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, DT.minmax(); tabular=true, dim=:col)
n = fit(MinMax, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, center(); tabular=true, dim=:col)
n = fit(Center, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, unitpower(); tabular=true, dim=:col)
n = fit(UnitPower, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, outliersuppress(;thr=5); tabular=true, dim=:col)
n = fit(OutlierSuppress, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

# ---------------------------------------------------------------------------- #
#                        single element normalization                          #
# ---------------------------------------------------------------------------- #
X = rand(100,75, 2)

@test_nowarn DT.normalize(X, DT.zscore())
@test_nowarn DT.normalize(X, sigmoid())
@test_nowarn DT.normalize(X, pnorm())
@test_nowarn DT.normalize(X, scale())
@test_nowarn DT.normalize(X, DT.minmax())
@test_nowarn DT.normalize(X, center())
@test_nowarn DT.normalize(X, unitpower())
@test_nowarn DT.normalize(X, outliersuppress())

# non-float convertion
@test_nowarn DT.normalize(a, DT.zscore())

# test against julia package Normalization
X = rand(200,100)

test = DT.normalize(X, DT.zscore())
n = fit(ZScore, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, DT.zscore(method=:half))
n = fit(HalfZScore, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, sigmoid())
n = fit(Sigmoid, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, pnorm())
n = fit(UnitEnergy, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, DT.minmax())
n = fit(MinMax, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, center())
n = fit(Center, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, unitpower())
n = fit(UnitPower, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = DT.normalize(X, outliersuppress(;thr=5))
n = fit(OutlierSuppress, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

# ---------------------------------------------------------------------------- #
#                    n-dimensional dataset normalization                       #
# ---------------------------------------------------------------------------- #
X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

@test_nowarn DT.normalize(X, DT.zscore())
@test_nowarn DT.normalize(X, sigmoid())
@test_nowarn DT.normalize(X, pnorm())
@test_nowarn DT.normalize(X, scale())
@test_nowarn DT.normalize(X, DT.minmax())
@test_nowarn DT.normalize(X, center())
@test_nowarn DT.normalize(X, unitpower())
@test_nowarn DT.normalize(X, outliersuppress())

function test_ds_norm(X, norm_func, NormType)
    test = DT.normalize(X, norm_func; tabular=true)
    # compute normalization the way ds_norm does (per column)
    col1_data = collect(Iterators.flatten(X[:, 1]))
    n = fit(NormType, reshape(col1_data, :, 1); dims=nothing)
    norm = Normalization.normalize(X[1,1], n)
    
    @test isapprox(test[1,1], norm)
end

# Run all tests
X = fill(rand(20, 10) .* 10, 10, 100)

test_ds_norm(X, DT.zscore(), ZScore)
test_ds_norm(X, DT.zscore(method=:half), HalfZScore)
test_ds_norm(X, sigmoid(), Sigmoid)
test_ds_norm(X, pnorm(), UnitEnergy)
test_ds_norm(X, DT.minmax(), MinMax)
test_ds_norm(X, center(), Center)
test_ds_norm(X, unitpower(), UnitPower)
test_ds_norm(X, outliersuppress(;thr=5), OutlierSuppress)

# non-float convertion
b = [rand(0:10, 20) for _ in 1:25, _ in 1:5]
@test_nowarn DT.normalize(b, DT.zscore())

# ---------------------------------------------------------------------------- #
#                            grouped normalization                             #
# ---------------------------------------------------------------------------- #
@testset "Basic grouped normalization" begin
    X = rand(100, 4)
    featvec = [mean, mean, std, std]
    
    # Test non-mutating version
    X_grouped = grouped_norm(X, DT.zscore(); featvec)
    @test size(X_grouped) == size(X)
    @test X_grouped isa Matrix{Float64}
    @test X != X_grouped  # Should be different array
    
    # Test mutating version
    X_mut = copy(X)
    result = grouped_norm!(X_mut, DT.zscore(); featvec)
    @test result === nothing
    @test X_mut ≈ X_grouped  # Should produce same result
end

@testset "Type conversion" begin
    # Test Integer to Float64 conversion
    X_int = rand(0:10, 50, 4)
    featvec = [mean, mean, maximum, minimum]
    
    X_norm = grouped_norm(X_int, DT.zscore(); featvec)
    @test eltype(X_norm) == Float64
    @test size(X_norm) == size(X_int)
    
    # Test with different normalization methods
    X_minmax = grouped_norm(X_int, DT.minmax(); featvec)
    @test eltype(X_minmax) == Float64
    
    X_center = grouped_norm(X_int, center(); featvec)
    @test eltype(X_center) == Float64
end

@testset "Grouping behavior" begin
    X = rand(100, 6)
    # Two groups: cols 1-3 (mean), cols 4-6 (std)
    featvec = [mean, mean, mean, std, std, std]
    
    X_grouped = grouped_norm(X, DT.zscore(); featvec)
    
    # Verify columns in same group share normalization
    # Collect all values from columns 1-3
    group1_orig = vcat(X[:, 1], X[:, 2], X[:, 3])
    group1_mean = Statistics.mean(group1_orig)
    group1_std = Statistics.std(group1_orig)
    
    # Check first group is normalized together
    group1_norm = vcat(X_grouped[:, 1], X_grouped[:, 2], X_grouped[:, 3])
    @test Statistics.mean(group1_norm) ≈ 0.0 atol=1e-10
    @test Statistics.std(group1_norm) ≈ 1.0 atol=1e-10
    
    # Collect all values from columns 4-6
    group2_norm = vcat(X_grouped[:, 4], X_grouped[:, 5], X_grouped[:, 6])
    @test Statistics.mean(group2_norm) ≈ 0.0 atol=1e-10
    @test Statistics.std(group2_norm) ≈ 1.0 atol=1e-10
end

@testset "Single feature per column" begin
    X = rand(50, 3)
    # Each column is its own group
    featvec = [mean, std, maximum]
    
    X_grouped = grouped_norm(X, DT.zscore(); featvec)
    X_tabular = tabular_norm(X, DT.zscore())
    
    # Should behave like tabular_norm when no grouping
    @test X_grouped ≈ X_tabular
end

@testset "All columns same feature" begin
    X = rand(100, 5)
    # All columns in same group
    featvec = fill(mean, 5)
    
    X_grouped = grouped_norm(X, DT.zscore(); featvec)
    X_element = element_norm(X, DT.zscore())
    
    # Should behave like element_norm when all grouped
    @test X_grouped ≈ X_element
end

@testset "Different normalization methods" begin
    X = rand(100, 4)
    featvec = [mean, mean, std, std]
    
    # DT.zscore
    X_z = grouped_norm(X, DT.zscore(); featvec)
    @test eltype(X_z) == Float64
    
    # minmax
    X_mm = grouped_norm(X, DT.minmax(); featvec)
    @test eltype(X_mm) == Float64
    group1 = vcat(X_mm[:, 1], X_mm[:, 2])
    @test minimum(group1) ≈ 0.0 atol=1e-10
    @test maximum(group1) ≈ 1.0 atol=1e-10
    
    # center
    X_c = grouped_norm(X, center(); featvec)
    group1_centered = vcat(X_c[:, 1], X_c[:, 2])
    @test Statistics.mean(group1_centered) ≈ 0.0 atol=1e-10
    
    # scale
    X_s = grouped_norm(X, scale(); featvec)
    @test eltype(X_s) == Float64
    
    # sigmoid
    X_sig = grouped_norm(X, sigmoid(); featvec)
    @test all(0 .< X_sig .< 1)
    
    # pnorm
    X_pn = grouped_norm(X, pnorm(); featvec)
    @test eltype(X_pn) == Float64
    
    # unitpower
    X_up = grouped_norm(X, unitpower(); featvec)
    @test eltype(X_up) == Float64
    
    # outliersuppress
    X_os = grouped_norm(X, outliersuppress(); featvec)
    @test eltype(X_os) == Float64
end

@testset "Edge cases" begin
    # Small dataset
    X = rand(5, 2)
    featvec = [mean, std]
    @test_nowarn grouped_norm(X, DT.zscore(); featvec)
    
    # Many groups
    X = rand(100, 10)
    featvec = [mean, std, minimum, maximum, median, 
                mean, std, minimum, maximum, median]
    X_grouped = grouped_norm(X, DT.zscore(); featvec)
    @test size(X_grouped) == size(X)
    
    # Large dataset (test threading)
    X = rand(1000, 20)
    featvec = repeat([mean, std, maximum, minimum], 5)
    X_grouped = grouped_norm(X, DT.zscore(); featvec)
    @test size(X_grouped) == (1000, 20)
end

@testset "In-place modification verification" begin
    X = rand(100, 4)
    featvec = [mean, mean, std, std]
    
    X_copy = copy(X)
    grouped_norm!(X_copy, DT.zscore(); featvec)
    
    # Verify it actually modified the array
    @test X != X_copy
    @test size(X) == size(X_copy)
    
    # Verify groups are normalized
    group1 = vcat(X_copy[:, 1], X_copy[:, 2])
    @test Statistics.mean(group1) ≈ 0.0 atol=1e-10
end

@testset "Consistent results between methods" begin
    X = rand(100, 6)
    featvec = [mean, mean, std, std, maximum, maximum]
    
    X1 = grouped_norm(X, DT.zscore(); featvec)
    
    X2 = copy(X)
    grouped_norm!(X2, DT.zscore(); featvec)
    
    @test X1 ≈ X2
end

@testset "grouped_norm Verification" begin
    @testset "Basic grouped normalization" begin
        # Create dataset where columns 1-2 are from 'mean' feature, 3-4 from 'std' feature
        X = [1.0  2.0  10.0  20.0;
             3.0  4.0  30.0  40.0;
             5.0  6.0  50.0  60.0]
        
        featvec = [mean, mean, std, std]
        
        X_grouped = grouped_norm(X, DT.zscore(); featvec=featvec)
        
        # Columns 1-2 should be normalized together using all values from both columns
        all_means = vec([X[:, 1]; X[:, 2]])  # [1,3,5,2,4,6]
        expected_mean_cols = DT.zscore()(all_means)
        
        # Check that columns 1-2 are normalized as a group
        μ_group1 = mean([X[:, 1]; X[:, 2]])
        σ_group1 = std([X[:, 1]; X[:, 2]])
        
        @test mean([X_grouped[:, 1]; X_grouped[:, 2]]) ≈ 0.0 atol=1e-10
        @test std([X_grouped[:, 1]; X_grouped[:, 2]]) ≈ 1.0 atol=1e-10
        
        # Check that columns 3-4 are normalized as a separate group
        μ_group2 = mean([X[:, 3]; X[:, 4]])
        σ_group2 = std([X[:, 3]; X[:, 4]])
        
        @test mean([X_grouped[:, 3]; X_grouped[:, 4]]) ≈ 0.0 atol=1e-10
        @test std([X_grouped[:, 3]; X_grouped[:, 4]]) ≈ 1.0 atol=1e-10
        
        # Verify specific values
        @test X_grouped[1, 1] ≈ (1.0 - μ_group1) / σ_group1
        @test X_grouped[1, 3] ≈ (10.0 - μ_group2) / σ_group2
    end
    
    @testset "Single group (all columns same feature)" begin
        X = [1.0  2.0  3.0;
             4.0  5.0  6.0;
             7.0  8.0  9.0]
        
        # All columns from same feature
        featvec = [mean, mean, mean]
        
        X_grouped = grouped_norm(X, DT.zscore(); featvec=featvec)
        
        # Should behave like element_norm since all columns are in same group
        X_element = element_norm(X, DT.zscore())
        
        @test X_grouped ≈ X_element atol=1e-10
    end
    
    @testset "Each column separate group" begin
        X = [1.0  10.0  100.0;
             2.0  20.0  200.0;
             3.0  30.0  300.0]
        
        # Each column is its own group (different features)
        featvec = [mean, maximum, minimum]
        
        X_grouped = grouped_norm(X, DT.zscore(); featvec=featvec)
        
        # Should behave like tabular_norm since each column is separate
        X_tabular = tabular_norm(X, DT.zscore())
        
        @test X_grouped ≈ X_tabular atol=1e-10
    end
    
    @testset "minmax with grouped features" begin
        X = [1.0  2.0  10.0  20.0;
             3.0  4.0  30.0  40.0;
             5.0  6.0  50.0  60.0]
        
        featvec = [mean, mean, std, std]
        
        X_grouped = grouped_norm(X, DT.minmax(); featvec=featvec)
        
        # Check group 1 (columns 1-2) normalized to [0, 1]
        @test minimum([X_grouped[:, 1]; X_grouped[:, 2]]) ≈ 0.0 atol=1e-10
        @test maximum([X_grouped[:, 1]; X_grouped[:, 2]]) ≈ 1.0 atol=1e-10
        
        # Check group 2 (columns 3-4) normalized to [0, 1]
        @test minimum([X_grouped[:, 3]; X_grouped[:, 4]]) ≈ 0.0 atol=1e-10
        @test maximum([X_grouped[:, 3]; X_grouped[:, 4]]) ≈ 1.0 atol=1e-10
    end
    
    @testset "In-place grouped_norm!" begin
        X_orig = [1.0  2.0  10.0  20.0;
                  3.0  4.0  30.0  40.0;
                  5.0  6.0  50.0  60.0]
        
        X_copy = copy(X_orig)
        featvec = [mean, mean, std, std]
        
        # Test in-place version
        grouped_norm!(X_copy, DT.zscore(); featvec=featvec)
        
        # Should give same result as non-mutating version
        X_grouped = grouped_norm(X_orig, DT.zscore(); featvec=featvec)
        
        @test X_copy ≈ X_grouped
    end
    
    @testset "Three feature groups" begin
        X = [1.0  2.0  10.0  20.0  100.0  200.0;
             3.0  4.0  30.0  40.0  300.0  400.0;
             5.0  6.0  50.0  60.0  500.0  600.0]
        
        # Three groups: cols 1-2 (mean), 3-4 (std), 5-6 (maximum)
        featvec = [mean, mean, std, std, maximum, maximum]
        
        X_grouped = grouped_norm(X, DT.zscore(); featvec=featvec)
        
        # Verify each group is normalized independently
        for (cols, name) in [([1,2], "group1"), ([3,4], "group2"), ([5,6], "group3")]
            group_data = vec(X_grouped[:, cols])
            @test mean(group_data) ≈ 0.0 atol=1e-10
            @test std(group_data) ≈ 1.0 atol=1e-10
        end
    end
    
    @testset "Real to Float64 conversion" begin
        X_int = [1  2  10  20;
                 3  4  30  40;
                 5  6  50  60]
        
        featvec = [mean, mean, std, std]
        
        X_grouped = grouped_norm(X_int, DT.zscore(); featvec=featvec)
        
        @test eltype(X_grouped) <: AbstractFloat
        @test mean([X_grouped[:, 1]; X_grouped[:, 2]]) ≈ 0.0 atol=1e-10
    end
end

@testset "different function calling method" begin
    X = rand(100, 50)

    X1 = element_norm(X, DT.zscore())
    X2 = element_norm(X, DT.zscore)
    @test X1 == X2

    X1 = element_norm(X, DT.zscore(method=:std)) # default
    X2 = element_norm(X, DT.zscore)
    @test X1 == X2
    
    X1 = element_norm(X, sigmoid())
    X2 = element_norm(X, sigmoid)
    @test X1 == X2

    X1 = element_norm(X, pnorm())
    X2 = element_norm(X, pnorm)
    @test X1 == X2

    X1 = element_norm(X, pnorm(p=2)) # default
    X2 = element_norm(X, pnorm)
    @test X1 == X2

    X1 = element_norm(X, scale())
    X2 = element_norm(X, scale)
    @test X1 == X2

    X1 = element_norm(X, scale(factor=:std)) # default
    X2 = element_norm(X, scale)
    @test X1 == X2

    X1 = element_norm(X, DataTreatments.minmax())
    X2 = element_norm(X, DataTreatments.minmax)
    @test X1 == X2

    X1 = element_norm(X, DataTreatments.minmax(lower=0.0, upper=1.0)) # default
    X2 = element_norm(X, DataTreatments.minmax)
    @test X1 == X2

    X1 = element_norm(X, center())
    X2 = element_norm(X, center)
    @test X1 == X2

    X1 = element_norm(X, center(method=:mean)) # default
    X2 = element_norm(X, center)
    @test X1 == X2

    X1 = element_norm(X, unitpower())
    X2 = element_norm(X, unitpower)
    @test X1 == X2

    X1 = element_norm(X, outliersuppress())
    X2 = element_norm(X, outliersuppress)
    @test X1 == X2

    X1 = element_norm(X, outliersuppress(thr=0.5)) # default
    X2 = element_norm(X, outliersuppress)
    @test X1 == X2
end
