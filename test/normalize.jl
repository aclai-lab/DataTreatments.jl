using Test
using DataTreatments

using Normalization

a = [8 1 6; 3 5 7; 4 9 2]
b = reshape(1:18, 3, 3, 2)

zscore_norm = tabular_norm(a, zscore())
@test isapprox(zscore_norm, [1.13389 -1.0 0.377964; -0.755929 0.0 0.755929; -0.377964 1.0 -1.13389], atol=1e-5)

zscore_row = tabular_norm(a, zscore(); dim=row)
@test isapprox(zscore_row, [0.83205 -1.1094 0.27735; -1.0 0.0 1.0; -0.27735 1.1094 -0.83205], atol=1e-5)


### test
X = rand(1000,750)

@test_nowarn element_norm(X, zscore())
@test_nowarn element_norm(X, sigmoid())
@test_nowarn element_norm(X, rescale())
@test_nowarn element_norm(X, center())
@test_nowarn element_norm(X, unitenergy())
@test_nowarn element_norm(X, unitpower())
@test_nowarn element_norm(X, halfzscore())
@test_nowarn element_norm(X, outliersuppress())
@test_nowarn element_norm(X, minmaxclip())

X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

@test_nowarn ds_norm(X, zscore())
@test_nowarn ds_norm(X, sigmoid())
@test_nowarn ds_norm(X, rescale())
@test_nowarn ds_norm(X, center())
@test_nowarn ds_norm(X, unitenergy())
@test_nowarn ds_norm(X, unitpower())
@test_nowarn ds_norm(X, halfzscore())
@test_nowarn ds_norm(X, outliersuppress())
@test_nowarn ds_norm(X, minmaxclip())


### TEST all passed
X = rand(200,100)

test = element_norm(X, zscore())
n = fit(ZScore, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, sigmoid())
n = fit(Sigmoid, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, rescale())
n = fit(MinMax, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, center())
n = fit(Center, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, unitenergy())
n = fit(UnitEnergy, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, unitpower())
n = fit(UnitPower, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, halfzscore())
n = fit(HalfZScore, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, outliersuppress())
n = fit(OutlierSuppress, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, minmaxclip())
n = fit(MinMaxClip, X)
norm = normalize(X, n)
@test isapprox(test, norm)

### TEST all passed
X = rand(200,100)

test = tabular_norm(X, zscore())
n = fit(ZScore, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, sigmoid())
n = fit(Sigmoid, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, rescale())
n = fit(MinMax, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, center())
n = fit(Center, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, unitenergy())
n = fit(UnitEnergy, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, unitpower())
n = fit(UnitPower, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, halfzscore())
n = fit(HalfZScore, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, outliersuppress())
n = fit(OutlierSuppress, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, minmaxclip())
n = fit(MinMaxClip, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)


### TEST ##############################################################################

function test_ds_norm(X, norm_func, NormType)
    test = ds_norm(X, norm_func())
    # compute normalization the way ds_norm does (per column)
    col1_data = collect(Iterators.flatten(X[:, 1]))
    n = fit(NormType, reshape(col1_data, :, 1); dims=nothing)
    norm = normalize(X[1,1], n)
    
    @test isapprox(test[1,1], norm)
end

# Run all tests
X = fill(rand(20, 10) .* 10, 10, 100)

test_ds_norm(X, zscore, ZScore)
test_ds_norm(X, sigmoid, Sigmoid)
test_ds_norm(X, rescale, MinMax)
test_ds_norm(X, center, Center)
test_ds_norm(X, unitenergy, UnitEnergy)
test_ds_norm(X, unitpower, UnitPower)
test_ds_norm(X, halfzscore, HalfZScore)
test_ds_norm(X, outliersuppress, OutlierSuppress)
test_ds_norm(X, minmaxclip, MinMaxClip)

