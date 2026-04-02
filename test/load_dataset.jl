using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays
using Statistics
using Impute

# ---------------------------------------------------------------------------- #
#                                  test data                                   #
# ---------------------------------------------------------------------------- #
function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
    DataFrame(
        str_col  = [missing, "blue", "green", "red", "blue"],
        sym_col  = [:circle, :square, :triangle, :square, missing],
        img4     = [i == 3 ? missing : create_image(i + 30) for i in 1:5],
        int_col  = Int[10, 20, 30, 40, 50],
        V1       = [NaN, missing, 3.0, 4.0, 5.6],
        V2       = [2.5, missing, 4.5, 5.5, NaN],
        ts1      = [missing, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
        V4       = [4.1, NaN, NaN, 7.1, 5.5],
        V5       = [5.0, 6.0, 7.0, 8.0, 1.8],
        ts2      = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), missing],
        ts3      = [[1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2], missing, missing, missing, [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]],
        V3       = [3.2, 4.2, 5.2, missing, 2.4],
        ts4      = [
            [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8],
            missing,
            [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8],
            [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]
        ],
        img1     = [create_image(i) for i in 1:5],
        cat_col  = categorical(["small", "medium", missing, "small", "large"]),
        uint_col = UInt32[1, 2, 3, 4, 5],
        img2     = [i == 1 ? NaN : create_image(i + 10) for i in 1:5],
        img3     = [create_image(i + 20) for i in 1:5],
    )
end

df = build_test_df()
t_classif = ["classA", "classB", "classC", "classA", "classB"]
t_regress = [1.2, 3.4, 2.2, 4.8, 0.9]

# ---------------------------------------------------------------------------- #
#                           basic load_dataset tests                           #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — basic" begin
    @testset "no treatments, no target" begin
        dt = load_dataset(df)
        @test dt isa DataTreatment
        @test get_target(dt) isa AbstractVector
    end

    @testset "no treatments, classification target" begin
        dt = load_dataset(df, t_classif)
        @test dt isa DataTreatment
        @test length(get_target(dt)) == nrow(df)
    end

    @testset "no treatments, regression target" begin
        dt = load_dataset(df, t_regress)
        @test dt isa DataTreatment
        @test eltype(get_target(dt)) <: AbstractFloat
    end

    @testset "from Matrix" begin
        m = rand(5, 4)
        dt = load_dataset(m)
        @test dt isa DataTreatment
    end
end

# ---------------------------------------------------------------------------- #
#                        TreatmentGroup — column selection                     #
# ---------------------------------------------------------------------------- #
@testset "TreatmentGroup — column selection" begin
    @testset "dims=0 (scalar columns only)" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(dims=0))
        data, vnames = get_tabular(dt)
        @test !isempty(vnames)
        @test size(data, 1) == nrow(df)
    end

    @testset "dims=1 (1D array columns only)" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(dims=1))
        @test dt isa DataTreatment
    end

    @testset "dims=2 (2D array columns only)" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(dims=2))
        @test dt isa DataTreatment
    end

    @testset "datatype=:discrete" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(dims=0, datatype=:discrete))
        data, vnames = get_discrete(dt)
        @test !isempty(vnames)
    end

    @testset "datatype=:continuous" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(dims=0, datatype=:continuous))
        data, vnames = get_continuous(dt)
        @test !isempty(vnames)
        @test eltype(data) <: Union{Missing, AbstractFloat}
    end

    @testset "name_expr regex" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(name_expr=r"^V"))
        _, vnames = get_tabular(dt)
        @test all(startswith(n, "V") for n in vnames)
    end

    @testset "name_expr vector" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(name_expr=["V1", "V3"]))
        _, vnames = get_continuous(dt)
        @test "V1" in vnames
        @test "V3" in vnames
    end

    @testset "name_expr function" begin
        dt = load_dataset(df, t_classif, TreatmentGroup(name_expr=n -> endswith(n, "col")))
        _, vnames = get_tabular(dt)
        @test all(endswith(n, "col") for n in vnames)
    end
end

# ---------------------------------------------------------------------------- #
#                       TreatmentGroup — multiple groups                       #
# ---------------------------------------------------------------------------- #
@testset "TreatmentGroup — multiple groups" begin
    @testset "discrete + continuous split" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, datatype=:discrete),
            TreatmentGroup(dims=0, datatype=:continuous)
        )
        data_d, vnames_d = get_discrete(dt)
        data_c, vnames_c = get_continuous(dt)
        @test !isempty(vnames_d)
        @test !isempty(vnames_c)
        @test isempty(intersect(vnames_d, vnames_c))
    end

    @testset "1D + 2D split" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=1),
            TreatmentGroup(dims=2)
        )
        @test dt isa DataTreatment
    end

    @testset "leftover_ds includes unmatched columns" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(name_expr=r"^V"),
            leftover_ds=true
        )
        @test dt isa DataTreatment
    end

    @testset "leftover_ds=false excludes unmatched columns" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(name_expr=r"^V"),
            leftover_ds=false
        )
        _, vnames = get_tabular(dt)
        @test all(startswith(n, "V") for n in vnames)
    end
end

# ---------------------------------------------------------------------------- #
#                         aggregate processing mode                            #
# ---------------------------------------------------------------------------- #
@testset "aggregate — tabular feature extraction" begin
    @testset "1D columns with splitwindow" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=DT.aggregate(
                    features=(mean, maximum),
                    win=(splitwindow(nwindows=2),)
                )
            )
        )
        data, vnames = get_aggregated(dt)
        @test !isempty(vnames)
        @test size(data, 1) == nrow(df)
        @test size(data, 2) == length(vnames)
    end

    @testset "2D columns with splitwindow" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=2,
                aggrfunc=DT.aggregate(
                    features=(mean, maximum),
                    win=(splitwindow(nwindows=2),)
                )
            )
        )
        data, vnames = get_aggregated(dt)
        @test !isempty(vnames)
        @test size(data, 1) == nrow(df)
    end

    @testset "1D columns with adaptivewindow" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=DT.aggregate(
                    features=(mean, maximum, minimum),
                    win=(adaptivewindow(nwindows=3, overlap=0.2),)
                )
            )
        )
        data, vnames = get_aggregated(dt)
        @test !isempty(vnames)
    end

    @testset "get_tabular includes aggregated" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=DT.aggregate(
                    features=(mean, maximum),
                    win=(splitwindow(nwindows=2),)
                )
            )
        )
        data, vnames = get_tabular(dt)
        @test !isempty(vnames)
        @test size(data, 1) == nrow(df)
    end
end

# ---------------------------------------------------------------------------- #
#                         reducesize processing mode                           #
# ---------------------------------------------------------------------------- #
@testset "reducesize — dimensionality reduction" begin
    @testset "1D columns" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=3),)
                )
            )
        )
        data, vnames = get_reduced(dt)
        @test !isempty(vnames)
        @test size(data, 1) == nrow(df)
    end

    @testset "2D columns" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=2,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=2),)
                )
            )
        )
        data, vnames = get_reduced(dt)
        @test !isempty(vnames)
        @test size(data, 1) == nrow(df)
    end

    @testset "get_multidim returns reduced" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=3),)
                )
            )
        )
        data, vnames = get_multidim(dt)
        @test !isempty(vnames)
    end

    @testset "1D with adaptivewindow" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=reducesize(
                    reducefunc=median,
                    win=(adaptivewindow(nwindows=3, overlap=0.2),)
                )
            )
        )
        data, vnames = get_reduced(dt)
        @test !isempty(vnames)
    end
end

# ---------------------------------------------------------------------------- #
#                          imputation (Impute.jl)                              #
# ---------------------------------------------------------------------------- #
@testset "Imputation — scalar columns" begin
    @testset "LOCF on all dims=0" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, impute=(LOCF(),))
        )
        @test dt isa DataTreatment
    end

    @testset "LOCF + NOCB on discrete" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, impute=(LOCF(), NOCB()), datatype=:discrete)
        )
        data, _ = get_discrete(dt)
        @test !isempty(data)
    end

    @testset "SVD on continuous" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, impute=(SVD(),), datatype=:continuous)
        )
        data, _ = get_continuous(dt)
        @test !isempty(data)
    end

    @testset "Interpolate + LOCF + NOCB on continuous" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=0,
                impute=(Interpolate(), LOCF(), NOCB()),
                datatype=:continuous
            )
        )
        data, _ = get_continuous(dt)
        @test !isempty(data)
    end

    @testset "separate impute strategies per datatype" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, impute=(LOCF(), NOCB()), datatype=:discrete),
            TreatmentGroup(dims=0, impute=(SVD(),), datatype=:continuous)
        )
        data_d, _ = get_discrete(dt)
        data_c, _ = get_continuous(dt)
        @test !isempty(data_d)
        @test !isempty(data_c)
    end
end

@testset "Imputation — multidimensional columns" begin
    @testset "SVD on 1D" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=1, impute=(SVD(),))
        )
        @test dt isa DataTreatment
    end

    @testset "SVD on 2D" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=2, impute=(SVD(),))
        )
        @test dt isa DataTreatment
    end

    @testset "Substitute on 1D with reducesize" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=3),)
                ),
                impute=(DT.Substitute(statistic=mean),)
            )
        )
        @test dt isa DataTreatment
    end

    @testset "LOCF + NOCB on 2D with reducesize" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=2,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=2),)
                ),
                impute=(LOCF(), NOCB())
            )
        )
        @test dt isa DataTreatment
    end
end

# ---------------------------------------------------------------------------- #
#                        normalization (Normalization.jl)                      #
# ---------------------------------------------------------------------------- #
@testset "Normalization" begin
    @testset "MinMax on continuous" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, datatype=:continuous, norm=MinMax)
        )
        data, _ = get_continuous(dt)
        valid = filter(x -> !ismissing(x) && !isnan(x), data)
        @test all(0.0 .<= valid .<= 1.0)
    end

    @testset "MinMax on 1D reducesize" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=2),)
                ),
                norm=MinMax
            )
        )
        data, _ = get_multidim(dt)
        @test !isempty(data)
    end

    @testset "MinMax on 2D reducesize" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=2,
                aggrfunc=reducesize(
                    reducefunc=mean,
                    win=(splitwindow(nwindows=2),)
                ),
                norm=MinMax
            )
        )
        data, _ = get_multidim(dt)
        @test !isempty(data)
    end

    @testset "MinMax on 1D aggregate" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=1,
                aggrfunc=DT.aggregate(
                    features=(mean, maximum),
                    win=(splitwindow(nwindows=2),)
                ),
                norm=MinMax
            )
        )
        data, _ = get_tabular(dt)
        valid = filter(x -> !ismissing(x) && !isnan(x), data)
        @test all(0.0 .<= valid .<= 1.0)
    end

    @testset "MinMax on 2D aggregate" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(
                dims=2,
                aggrfunc=DT.aggregate(
                    features=(mean, maximum),
                    win=(splitwindow(nwindows=2),)
                ),
                norm=MinMax
            )
        )
        data, _ = get_tabular(dt)
        @test !isempty(data)
    end
end

# ---------------------------------------------------------------------------- #
#                             accessor methods                                 #
# ---------------------------------------------------------------------------- #
@testset "Accessor methods" begin
    dt = load_dataset(df, t_classif,
        TreatmentGroup(dims=0),
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean,), win=(splitwindow(nwindows=2),))),
        TreatmentGroup(dims=2, aggrfunc=reducesize(reducefunc=mean, win=(splitwindow(nwindows=2),)))
    )

    @testset "get_discrete" begin
        data, vnames = get_discrete(dt)
        @test vnames isa Vector{String}
        @test data isa AbstractMatrix
    end

    @testset "get_continuous" begin
        data, vnames = get_continuous(dt)
        @test vnames isa Vector{String}
        @test data isa AbstractMatrix
    end

    @testset "get_aggregated" begin
        data, vnames = get_aggregated(dt)
        @test vnames isa Vector{String}
        @test data isa AbstractMatrix
    end

    @testset "get_reduced" begin
        data, vnames = get_reduced(dt)
        @test vnames isa Vector{String}
        @test data isa AbstractArray
    end

    @testset "get_tabular merges discrete + continuous + aggregated" begin
        data, vnames = get_tabular(dt)
        @test vnames isa Vector{String}
        @test size(data, 1) == nrow(df)
        @test length(vnames) == size(data, 2)
    end

    @testset "get_multidim returns reduced arrays" begin
        data, vnames = get_multidim(dt)
        @test vnames isa Vector{String}
        @test data isa AbstractArray
    end

    @testset "get_target" begin
        target = get_target(dt)
        @test length(target) == nrow(df)
    end

    @testset "empty returns on missing category" begin
        dt_disc = load_dataset(df, t_classif, TreatmentGroup(dims=0, datatype=:discrete))
        data, vnames = get_continuous(dt_disc)
        @test isempty(vnames)

        dt_cont = load_dataset(df, t_classif, TreatmentGroup(dims=0, datatype=:continuous))
        data, vnames = get_discrete(dt_cont)
        @test isempty(vnames)
    end
end

# ---------------------------------------------------------------------------- #
#                            float_type parameter                              #
# ---------------------------------------------------------------------------- #
@testset "float_type parameter" begin
    @testset "Float32" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, datatype=:continuous);
            float_type=Float32
        )
        dt isa DataTreatment{Float32}
        data, _ = get_continuous(dt)
        @test eltype(skipmissing(data)) == Float32
    end

    @testset "Float64 (default)" begin
        dt = load_dataset(df, t_classif,
            TreatmentGroup(dims=0, datatype=:continuous)
        )
        @test dt isa DataTreatment{Float64}
        data, _ = get_continuous(dt)
        @test eltype(skipmissing(data)) == Float64
    end
end