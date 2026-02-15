using Test
using DataTreatments
const DT = DataTreatments

@test ZScore == ZScore
@test DT._nt(ZScore()) == (type = ZScore, dims=nothing)
@test DT._nt(ZScore(dims=2)) == (type=ZScore, dims=2)
@test DT._nt(ZScore(method=:std)) == (type=ZScore, dims=nothing)
@test DT._nt(ZScore(method=:robust)) == (type=DT.ZScoreRobust, dims=nothing)
@test DT._nt(ZScore(method=:half)) == (type=DT.HalfZScore, dims=nothing)
@test_throws ErrorException ZScore(dims=3)
@test_throws ErrorException ZScore(method=:invalid)

@test MinMax == MinMax
@test DT._nt(MinMax()) == (type=MinMax, dims=nothing)
@test DT._nt(MinMax(dims=2)) == (type=MinMax, dims=2)
@test_throws ErrorException MinMax(dims=3)

@test Scale == Scale
@test DT._nt(Scale()) == (type=Scale, dims=nothing)
@test DT._nt(Scale(dims=2)) == (type=Scale, dims=2)
@test DT._nt(Scale(method=:std)) == (type=Scale, dims=nothing)
@test DT._nt(Scale(method=:mad)) == (type=DT.ScaleMad, dims=nothing)
@test DT._nt(Scale(method=:first)) == (type=DT.ScaleFirst, dims=nothing)
@test DT._nt(Scale(method=:iqr)) == (type=DT.ScaleIqr, dims=nothing)
@test_throws ErrorException Scale(dims=3)
@test_throws ErrorException Scale(method=:invalid)

@test Sigmoid == Sigmoid
@test DT._nt(Sigmoid()) == (type=Sigmoid, dims=nothing)
@test DT._nt(Sigmoid(dims=2)) == (type=Sigmoid, dims=2)
@test_throws ErrorException Sigmoid(dims=3)

@test Center == Center
@test DT._nt(Center()) == (type=Center, dims=nothing)
@test DT._nt(Center(dims=2)) == (type=Center, dims=2)
@test DT._nt(Center(method=:mean)) == (type=Center, dims=nothing)
@test DT._nt(Center(method=:median)) == (type=DT.CenterMedian, dims=nothing)
@test_throws ErrorException Center(dims=3)
@test_throws ErrorException Center(method=:invalid)

@test UnitEnergy == UnitEnergy
@test DT._nt(UnitEnergy()) == (type=UnitEnergy, dims=nothing)
@test DT._nt(UnitEnergy(dims=2)) == (type=UnitEnergy, dims=2)
@test_throws ErrorException UnitEnergy(dims=3)

@test UnitPower == UnitPower
@test DT._nt(UnitPower()) == (type=UnitPower, dims=nothing)
@test DT._nt(UnitPower(dims=2)) == (type=UnitPower, dims=2)
@test_throws ErrorException UnitPower(dims=3)

@test PNorm == PNorm
@test DT._nt(PNorm()) == (type=DT.PNorm, dims=nothing)
@test DT._nt(PNorm(dims=2)) == (type=DT.PNorm, dims=2)
@test DT._nt(PNorm(p=1)) == (type=DT.PNorm1, dims=nothing)
@test DT._nt(PNorm(p=Inf)) == (type=DT.PNormInf, dims=nothing)
@test_throws ErrorException PNorm(dims=3)
@test_throws ErrorException PNorm(p=5)

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

grouby_cols = normalize(X, MinMax(dims=1))
@test grouby_cols == [
    1.0 0.0 0.8;
    0.0 0.5 1.0;
    0.2 1.0 0.0
]

grouby_rows = normalize(X, MinMax(dims=2))
@test isapprox(grouby_rows, [
    1.0 0.0 0.714285714;
    0.0 0.5 1.0;
    0.285714286 1.0 0.0
])

Xmatrix = [Float64.(rand(1:100, 4, 2)) for _ in 1:10, _ in 1:5]

@test_nowarn normalize(Xmatrix, ZScore)
@test_nowarn normalize(Xmatrix, ZScore(dims=1))
@test_nowarn normalize(Xmatrix, ZScore(dims=2))

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
X = Float64.([8 1 6; 3 5 7; 4 9 2])

# test values verified against MATLAB
zscore_norm = normalize(X, ZScore(dims=1))
@test isapprox(zscore_norm, [1.13389 -1.0 0.377964; -0.755929 0.0 0.755929; -0.377964 1.0 -1.13389], atol=1e-5)

zscore_row = normalize(X, ZScore(dims=2))
@test isapprox(zscore_row, [0.83205 -1.1094 0.27735; -1.0 0.0 1.0; -0.27735 1.1094 -0.83205], atol=1e-5)

zscore_robust = normalize(X, ZScore(method=:robust, dims=1))
@test zscore_robust == [4.0 -1.0 0.0; -1.0 0.0 1.0; 0.0 1.0 -4.0]

@test_nowarn zscore_half = normalize(X, ZScore(method=:half, dims=2))

@test_nowarn normalize(X, Sigmoid(dims=2))

scale_norm = normalize(X, Scale(dims=1))
@test isapprox(scale_norm, [3.02372 0.25 2.26779; 1.13389 1.25 2.64575; 1.51186 2.25 0.755929], atol=1e-5)

scale_norm = normalize(X, Scale(method=:mad, dims=1))
@test scale_norm == [8.0 0.25 6.0; 3.0 1.25 7.0; 4.0 2.25 2.0]

scale_norm = normalize(X, Scale(method=:first, dims=1))
@test isapprox(scale_norm, [1.0 1.0 1.0; 0.375 5.0 1.16667; 0.5 9.0 0.333333], atol=1e-5)

@test_nowarn scale_norm = normalize(X, Scale(method=:iqr, dims=1))

minmax_norm = normalize(X, MinMax(dims=1))
@test minmax_norm == [1.0 0.0 0.8; 0.0 0.5 1.0; 0.2 1.0 0.0]

center_norm = normalize(X, Center(dims=1))
@test center_norm == [3.0 -4.0 1.0; -2.0 0.0 2.0; -1.0 4.0 -3.0]

center_norm = normalize(X, Center(method=:median, dims=1))
@test center_norm == [4.0 -4.0 0.0; -1.0 0.0 1.0; 0.0 4.0 -4.0]

@test_nowarn normalize(X, UnitEnergy(dims=1))
@test_nowarn normalize(X, UnitPower(dims=1))

# assolutamente da verificare
# @test_nowarn normalize(X, OutlierSuppress; dims=1)

norm_pnorm = normalize(X, PNorm(dims=1, p=1))
@test isapprox(norm_pnorm, [0.533333 0.0666667 0.4; 0.2 0.333333 0.466667; 0.266667 0.6 0.133333], atol=1e-5)

norm_pnorm = normalize(X, PNorm(dims=1))
@test isapprox(norm_pnorm, [0.847998 0.0966736 0.635999; 0.317999 0.483368 0.741999; 0.423999 0.870063 0.212], atol=1e-6)

norm_pnorm = normalize(X, PNorm(dims=1, p=Inf))
@test isapprox(norm_pnorm, [1.0 0.111111 0.857143; 0.375 0.555556 1.0; 0.5 1.0 0.285714], atol=1e-6)

@testset "NormSpec show/convert/Tuple" begin
    ns1 = ZScore(dims=1)
    ns2 = MinMax()

    for ns in (ns1, ns2)
        nt = DT._nt(ns)

        @test nt isa NamedTuple
        @test haskey(nt, :type)
        @test haskey(nt, :dims)

        @test Tuple(ns) == (nt.type, nt.dims)

        # Base.show(io::IO, ns::NormSpec)
        @test sprint(show, ns) == sprint(show, nt)

        # Base.show(io::IO, ::MIME"text/plain", ns::NormSpec)
        @test sprint(show, MIME"text/plain"(), ns) ==
              sprint(show, MIME"text/plain"(), nt)
    end
end