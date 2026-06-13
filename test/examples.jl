using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays
using Statistics: mean

# ---------------------------------------------------------------------------- #
#                                 test data                                    #
# ---------------------------------------------------------------------------- #
function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
    DataFrame(
        str_col  = [missing, "blue", "green", "red", "blue"],
        sym_col  = [:circle, :square, :triangle, :square, missing],
        img4     = [
            i == 3 ? missing : create_image(i + 30) for i in 1:5
        ],
        int_col  = Int[10, 20, 30, 40, 50],
        V1       = [NaN, missing, 3.0, 4.0, 5.6],
        V2       = [2.5, missing, 4.5, 5.5, NaN],
        ts1      = [
            missing, collect(2.0:7.0),
            missing, collect(4.0:9.0),
            collect(5.0:10.0),
        ],
        V4       = [4.1, NaN, NaN, 7.1, 5.5],
        V5       = [5.0, 6.0, 7.0, 8.0, 1.8],
        ts2      = [
            collect(2.0:0.5:5.5), collect(1.0:0.5:4.5),
            collect(3.0:0.5:6.5), collect(4.0:0.5:7.5),
            missing,
        ],
        ts3      = [
            [1.0, 1.2, 1.2, 2.6, NaN,     4.0, 4.2],
            missing, missing, missing,
            [3.0, NaN, 4.4, missing, 5.8,  7.0, 7.2],
        ],
        V3       = [3.2, 4.2, 5.2, missing, 2.4],
        ts4      = [
            [6.0, 5.2, missing, 4.4, 1.2,     3.6, 2.8],
            missing,
            [5.0, 4.2, NaN,     3.4, missing,  2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN,      5.6, 4.8],
            [9.0, NaN, 8.2,     missing, 7.4,  6.6, 5.8],
        ],
        img1     = [create_image(i)      for i in 1:5],
        cat_col  = categorical(
            ["small", "medium", missing, "small", "large"]
        ),
        uint_col = UInt32[1, 2, 3, 4, 5],
        img2     = [
            i == 1 ? NaN : create_image(i + 10) for i in 1:5
        ],
        img3     = [create_image(i + 20) for i in 1:5],
    )
end

const DF         = build_test_df()
const T_CLASSIF  = ["classA", "classB", "classC", "classA", "classB"]
const T_REGRESS  = [1.2, 3.4, 2.2, 4.8, 0.9]

# ---------------------------------------------------------------------------- #
#                          helper: basic sanity checks                         #
# ---------------------------------------------------------------------------- #
function check_getters(dt)
    get_discrete(dt)
    get_continuous(dt)
    get_multidim(dt)
    get_tabular(dt)
end

# ---------------------------------------------------------------------------- #
#                         1. no treatment group                                #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — no treatment group" begin
    dt = load_dataset(DF)
    @test dt isa DataTreatment
    check_getters(dt)
end

# ---------------------------------------------------------------------------- #
#                    2. dims=0, all types                                      #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — dims=0 all types" begin
    dt = load_dataset(DF, T_CLASSIF, TreatmentGroup(dims=0))
    @test dt isa DataTreatment
    check_getters(dt)
    X_tab, _ = get_tabular(dt)
    @test size(X_tab, 1) == 5
end

# ---------------------------------------------------------------------------- #
#                    3. dims=0, separate discrete/continuous                   #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — dims=0 discrete + continuous" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(dims=0, datatype=:discrete),
        TreatmentGroup(dims=0, datatype=:continuous),
    )
    @test dt isa DataTreatment
    check_getters(dt)

    Xd, nd = get_discrete(dt)
    Xc, nc = get_continuous(dt)
    @test !isempty(nd)
    @test !isempty(nc)
end

# ---------------------------------------------------------------------------- #
#                    4. dims=0, continuous only                                #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — dims=0 continuous only" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(dims=0, datatype=:continuous),
    )
    @test dt isa DataTreatment
    Xd, nd = get_discrete(dt)
    @test isempty(nd)
    Xc, nc = get_continuous(dt)
    @test !isempty(nc)
end

# ---------------------------------------------------------------------------- #
#                    5. dims=1 and dims=2 (multidim)                           #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — dims=1 and dims=2" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(dims=1),
        TreatmentGroup(dims=2),
    )
    @test dt isa DataTreatment
    check_getters(dt)
end

# ---------------------------------------------------------------------------- #
#                    6. reducesize dims=1                                      #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — reducesize dims=1" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 1,
            aggrfunc = reducesize(
                reducefunc = mean,
                win        = (splitwindow(nwindows=3),),
            ),
        ),
    )
    @test dt isa DataTreatment
    Xm, nm = get_multidim(dt)
    @test !isempty(nm)
end

# ---------------------------------------------------------------------------- #
#                    7. reducesize dims=2                                      #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — reducesize dims=2" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 2,
            aggrfunc = reducesize(
                reducefunc = mean,
                win        = (splitwindow(nwindows=2),),
            ),
        ),
    )
    @test dt isa DataTreatment
    Xm, nm = get_multidim(dt)
    @test !isempty(nm)
end

# ---------------------------------------------------------------------------- #
#                    8. imputation — LOCF scalar                               #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — impute LOCF scalar" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(dims=0, impute=(LOCF(),)),
    )
    @test dt isa DataTreatment
    check_getters(dt)
end

# ---------------------------------------------------------------------------- #
#                    9. imputation — LOCF+NOCB discrete, SVD continuous        #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — impute LOCF+NOCB / SVD" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 0,
            impute   = (LOCF(), NOCB()),
            datatype = :discrete,
        ),
        TreatmentGroup(
            dims     = 0,
            impute   = (SVD(),),
            datatype = :continuous,
        ),
    )
    @test dt isa DataTreatment
    check_getters(dt)
end

# ---------------------------------------------------------------------------- #
#                    10. imputation — Interpolate+LOCF+NOCB continuous         #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — impute Interpolate+LOCF+NOCB" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 0,
            impute   = (Interpolate(), LOCF(), NOCB()),
            datatype = :continuous,
        ),
    )
    @test dt isa DataTreatment
    Xc, nc = get_continuous(dt)
    @test !isempty(nc)
end

# ---------------------------------------------------------------------------- #
#                    11. imputation — SVD multidim dims=1,2                    #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — impute SVD multidim" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(dims=1, impute=(SVD(),)),
        TreatmentGroup(dims=2, impute=(SVD(),)),
    )
    @test dt isa DataTreatment
    check_getters(dt)
end

# ---------------------------------------------------------------------------- #
#                    12. impute + reducesize dims=1                            #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — impute + reducesize dims=1" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 1,
            aggrfunc = reducesize(
                reducefunc = mean,
                win        = (splitwindow(nwindows=3),),
            ),
            impute   = (DT.Substitute(statistic=mean),),
        ),
    )
    @test dt isa DataTreatment
    Xm, nm = get_multidim(dt)
    @test !isempty(nm)
end

# ---------------------------------------------------------------------------- #
#                    13. impute + reducesize dims=2                            #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — impute + reducesize dims=2" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 2,
            aggrfunc = reducesize(
                reducefunc = mean,
                win        = (splitwindow(nwindows=2),),
            ),
            impute   = (LOCF(), NOCB()),
        ),
    )
    @test dt isa DataTreatment
    Xm, nm = get_multidim(dt)
    @test !isempty(nm)
end

# ---------------------------------------------------------------------------- #
#                    14. normalization — MinMax continuous                     #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — norm MinMax continuous" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 0,
            datatype = :continuous,
            norm     = DT.MinMax,
        ),
    )
    @test dt isa DataTreatment
    Xc, _ = get_continuous(dt)
    valid  = filter(!ismissing, Xc)
    @test all(x -> isnan(x) || -1e-6 <= x <= 1+1e-6, valid)
end

# ---------------------------------------------------------------------------- #
#                    15. normalization — MinMax reducesize dims=1              #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — norm MinMax reducesize dims=1" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 1,
            aggrfunc = reducesize(
                reducefunc = mean,
                win        = (splitwindow(nwindows=2),),
            ),
            norm     = DT.MinMax,
        ),
    )
    @test dt isa DataTreatment
    Xm, nm = get_multidim(dt)
    @test !isempty(nm)
end

# ---------------------------------------------------------------------------- #
#                    16. normalization — MinMax reducesize dims=2              #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — norm MinMax reducesize dims=2" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 2,
            aggrfunc = reducesize(
                reducefunc = mean,
                win        = (splitwindow(nwindows=2),),
            ),
            norm     = DT.MinMax,
        ),
    )
    @test dt isa DataTreatment
    Xm, nm = get_multidim(dt)
    @test !isempty(nm)
end

# ---------------------------------------------------------------------------- #
#                    17. aggregate dims=1 + MinMax                             #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — aggregate dims=1 + MinMax" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 1,
            aggrfunc = DT.aggregate(
                features = (mean, maximum),
                win      = (splitwindow(nwindows=2),),
            ),
            norm     = DT.MinMax,
        ),
    )
    @test dt isa DataTreatment
    Xt, nt = get_tabular(dt)
    @test !isempty(nt)
    valid = filter(!ismissing, Xt)
    @test all(x -> isnan(x) || -1e-6 <= x <= 1+1e-6, valid)
end

# ---------------------------------------------------------------------------- #
#                    18. aggregate dims=2 + MinMax                             #
# ---------------------------------------------------------------------------- #
@testset "load_dataset — aggregate dims=2 + MinMax" begin
    dt = load_dataset(
        DF, T_CLASSIF,
        TreatmentGroup(
            dims     = 2,
            aggrfunc = DT.aggregate(
                features = (mean, maximum),
                win      = (splitwindow(nwindows=2),),
            ),
            norm     = DT.MinMax,
        ),
    )
    @test dt isa DataTreatment
    Xt, nt = get_tabular(dt)
    @test !isempty(nt)
    valid = filter(!ismissing, Xt)
    @test all(x -> isnan(x) || -1e-6 <= x <= 1+1e-6, valid)
end