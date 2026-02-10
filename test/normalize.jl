using Test
using DataTreatments
const DT = DataTreatments

# ---------------------------------------------------------------------------- #
#                                 normalization                                #
# ---------------------------------------------------------------------------- #
X = Float64.([8 1 6; 3 5 7; 4 9 2])

all_elements = normalize(X, MinMax)
@test all_elements == [
    0.875 0.0 0.625;
    0.25 0.5 0.75;
    0.375 1.0 0.125
]

grouby_cols = normalize(X, MinMax; dims=1)
@test grouby_cols == [
    1.0 0.0 0.8;
    0.0 0.5 1.0;
    0.2 1.0 0.0
]

grouby_rows = normalize(X, MinMax; dims=2)
@test isapprox(grouby_rows, [
    1.0 0.0 0.714285714;
    0.0 0.5 1.0;
    0.285714286 1.0 0.0
])

Xmatrix = [Float64.(rand(1:100, 4, 2)) for _ in 1:10, _ in 1:5]

@test_nowarn normalize(Xmatrix, ZScore)
@test_nowarn normalize(Xmatrix, ZScore; dims=1)
@test_nowarn normalize(Xmatrix, ZScore; dims=2)

# ---------------------------------------------------------------------------- #
#                            multi dim normalization                           #
# ---------------------------------------------------------------------------- #
m1 = [1.0 1.0 1.0; 1.0 2.5 1.0; 1.0 1.0 1.0]
m2 = [1.0 1.0 1.0; 1.0 7.5 1.0; 1.0 1.0 1.0]
m3 = [9.0 9.0 9.0; 9.0 2.5 9.0; 9.0 9.0 9.0]
m4 = [9.0 9.0 9.0; 9.0 7.5 9.0; 9.0 9.0 9.0]

M = reshape([m1, m2, m3, m4], 2, 2) # 2x2 matrix of matrices

multidim_norm = normalize(M, MinMax)

# all elements of the matrices were scaled by the same coefficient,
# computed using all values across the matrices.
@test multidim_norm[1,1] ==
    [0.0 0.0 0.0; 0.0 0.1875 0.0; 0.0 0.0 0.0]
@test multidim_norm[1,2] == 
    [1.0 1.0 1.0; 1.0 0.1875 1.0; 1.0 1.0 1.0]
@test multidim_norm[2,1] ==
    [0.0 0.0 0.0; 0.0 0.8125 0.0; 0.0 0.0 0.0]
@test multidim_norm[2,2] ==
    [1.0 1.0 1.0; 1.0 0.8125 1.0; 1.0 1.0 1.0]

# ---------------------------------------------------------------------------- #
#                             tabular normalization                            #
# ---------------------------------------------------------------------------- #
a = [8 1 6; 3 5 7; 4 9 2]

# test values verified against MATLAB
zscore_norm = normalize(a, ZScore; dims=2)
@test isapprox(zscore_norm, [1.13389 -1.0 0.377964; -0.755929 0.0 0.755929; -0.377964 1.0 -1.13389], atol=1e-5)

zscore_row = normalize(a, ZScore; dims=1)
@test isapprox(zscore_row, [0.83205 -1.1094 0.27735; -1.0 0.0 1.0; -0.27735 1.1094 -0.83205], atol=1e-5)

@test_nowarn normalize(a, sigmoid(); dims=2)

norm_norm = normalize(a, pnorm(); dims=2)
@test isapprox(norm_norm, [0.847998 0.0966736 0.635999; 0.317999 0.483368 0.741999; 0.423999 0.870063 0.212], atol=1e-6)

norm_norm = normalize(a, pnorm(p=4); dims=2)
@test isapprox(norm_norm, [0.980428 0.108608 0.768635; 0.36766 0.543042 0.896741; 0.490214 0.977475 0.256212], atol=1e-5)

norm_norm = normalize(a, pnorm(p=Inf); dims=2)
@test isapprox(norm_norm, [1.0 0.111111 0.857143; 0.375 0.555556 1.0; 0.5 1.0 0.285714], atol=1e-6)

scale_norm = normalize(a, scale(factor=:std); dims=2)
@test isapprox(scale_norm, [3.02372 0.25 2.26779; 1.13389 1.25 2.64575; 1.51186 2.25 0.755929], atol=1e-5)

scale_norm = normalize(a, scale(factor=:mad); dims=2)
@test scale_norm == [8.0 0.25 6.0; 3.0 1.25 7.0; 4.0 2.25 2.0]

scale_norm = normalize(a, scale(factor=:first); dims=2)
@test isapprox(scale_norm, [1.0 1.0 1.0; 0.375 5.0 1.16667; 0.5 9.0 0.333333], atol=1e-5)

scale_norm = normalize(a, scale(factor=:iqr); dims=2)

minmax_norm = normalize(a, minmax(); dims=2)
@test minmax_norm == [1.0 0.0 0.8; 0.0 0.5 1.0; 0.2 1.0 0.0]

minmax_norm = normalize(a, minmax(lower=-2, upper=4); dims=2)
@test minmax_norm == [4.0 -2.0 2.8; -2.0 1.0 4.0; -0.8 4.0 -2.0]

center_norm = normalize(a, center(); dims=2)
@test center_norm == [3.0 -4.0 1.0; -2.0 0.0 2.0; -1.0 4.0 -3.0]

center_norm = normalize(a, center(method=:median); dims=2)
@test center_norm == [4.0 -4.0 0.0; -1.0 0.0 1.0; 0.0 4.0 -4.0]

@test_nowarn normalize(a, unitpower(); dims=2)

@test_nowarn normalize(a, outliersuppress(); dims=2)
@test_nowarn normalize(a, outliersuppress(thr=3); dims=2)

# test against julia package Normalization
X = rand(200,100)

test = normalize(X, ZScore; dims=2)
n = fit(ZScore, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, zscore(method=:half); dims=2)
n = fit(HalfZScore, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, sigmoid(); dims=2)
n = fit(Sigmoid, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, pnorm(); dims=2)
n = fit(UnitEnergy, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, minmax(); dims=2)
n = fit(MinMax, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, center(); dims=2)
n = fit(Center, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, unitpower(); dims=2)
n = fit(UnitPower, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, outliersuppress(;thr=5); dims=2)
n = fit(OutlierSuppress, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

# ---------------------------------------------------------------------------- #
#                        single element normalization                          #
# ---------------------------------------------------------------------------- #
X = rand(100,75, 2)

@test_nowarn normalize(X, ZScore)
@test_nowarn normalize(X, sigmoid())
@test_nowarn normalize(X, pnorm())
@test_nowarn normalize(X, scale())
@test_nowarn normalize(X, minmax())
@test_nowarn normalize(X, center())
@test_nowarn normalize(X, unitpower())
@test_nowarn normalize(X, outliersuppress())

# non-float convertion
@test_nowarn normalize(a, ZScore)

# test against julia package Normalization
X = rand(200,100)

test = normalize(X, ZScore)
n = fit(ZScore, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, zscore(method=:half))
n = fit(HalfZScore, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, sigmoid())
n = fit(Sigmoid, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, pnorm())
n = fit(UnitEnergy, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, minmax())
n = fit(MinMax, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, center())
n = fit(Center, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, unitpower())
n = fit(UnitPower, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

test = normalize(X, outliersuppress(;thr=5))
n = fit(OutlierSuppress, X)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

# ---------------------------------------------------------------------------- #
#                    n-dimensional dataset normalization                       #
# ---------------------------------------------------------------------------- #
X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

@test_nowarn normalize(X, ZScore)
@test_nowarn normalize(X, sigmoid())
@test_nowarn normalize(X, pnorm())
@test_nowarn normalize(X, scale())
@test_nowarn normalize(X, minmax())
@test_nowarn normalize(X, center())
@test_nowarn normalize(X, unitpower())
@test_nowarn normalize(X, outliersuppress())

function test_ds_norm(X, norm_func, NormType)
    test = normalize(X, norm_func; dims=2)
    # compute normalization the way ds_norm does (per column)
    col1_data = collect(Iterators.flatten(X[:, 1]))
    n = fit(NormType, reshape(col1_data, :, 1); dims=nothing)
    norm = Normalization.normalize(X[1,1], n)
    
    @test isapprox(test[1,1], norm)
end

# Run all tests
X = fill(rand(20, 10) .* 10, 10, 100)

test_ds_norm(X, ZScore, ZScore)
test_ds_norm(X, zscore(method=:half), HalfZScore)
test_ds_norm(X, sigmoid(), Sigmoid)
test_ds_norm(X, pnorm(), UnitEnergy)
test_ds_norm(X, minmax(), MinMax)
test_ds_norm(X, center(), Center)
test_ds_norm(X, unitpower(), UnitPower)
test_ds_norm(X, outliersuppress(;thr=5), OutlierSuppress)

# non-float convertion
b = [rand(0:10, 20) for _ in 1:25, _ in 1:5]
@test_nowarn normalize(b, ZScore)

# ---------------------------------------------------------------------------- #
#                              benchmark test                                  #
# ---------------------------------------------------------------------------- #
# test against julia package Normalization
X = rand(2000,1000)

test = normalize(X, ZScore; dims=2)
n = fit(ZScore, X, dims=1)
norm = Normalization.normalize(X, n)
@test isapprox(test, norm)

@btime test = normalize(X, ZScore; dims=2)
# 13.703 ms (6006 allocations: 45.91 MiB)

@btime begin
    n = fit(ZScore, X, dims=1)
    norm = Normalization.normalize(X, n)   
end
# 2.245 ms (178 allocations: 15.29 MiB)