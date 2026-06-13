using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
    DataFrame(
        str_col=[missing, "blue", "green", "red", "blue"],
        sym_col=[:circle, :square, :triangle, :square, missing],
        img4=[i == 3 ? missing : create_image(i + 30) for i in 1:5],
        int_col=Int[10, 20, 30, 40, 50],
        V1=[NaN, missing, 3.0, 4.0, 5.6],
        V2=[2.5, missing, 4.5, 5.5, NaN],
        ts1=[
            NaN, collect(2.0:7.0),
            missing, collect(4.0:9.0),
            collect(5.0:10.0)
        ],
        V4=[4.1, NaN, NaN, 7.1, 5.5],
        V5=[5.0, 6.0, 7.0, 8.0, 1.8],
        ts2=[
            collect(2.0:0.5:5.5),
            collect(1.0:0.5:4.5),
            collect(3.0:0.5:6.5),
            collect(4.0:0.5:7.5),
            NaN
        ],
        ts3=[
            [1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2],
            NaN, NaN, missing,
            [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]
        ],
        V3=[3.2, 4.2, 5.2, missing, 2.4],
        ts4=[
            [6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8],
            missing,
            [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8],
            [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8],
            [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]
        ],
        img1=[create_image(i) for i in 1:5],
        cat_col=categorical(["small", "medium", missing, "small", "large"]),
        uint_col=UInt32[1, 2, 3, 4, 5],
        img2=[i == 1 ? NaN : create_image(i + 10) for i in 1:5],
        img3=[create_image(i + 20) for i in 1:5],
    )
end

df = build_test_df()

t_classif = ["classA", "classB", "classC", "classA", "classB"]
t_regress = [1.2, 3.4, 2.2, 4.8, 0.9]

dt = DT.load_dataset(df)

@test_nowarn DT.get_tabular(dt)
@test_nowarn DT.get_multidim(dt)

dt = DT.load_dataset(df, t_classif)

dt = DT.load_dataset(df, t_regress)

dt = DT.load_dataset(
    df,
    TreatmentGroup(
        dims=1,
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        ),
        groupby=:feat
    )
)
@test get_tabular(dt)[1] isa Matrix
@test get_tabular(dt)[2] isa Vector{String}

dt = DT.load_dataset(df)
@test get_tabular(dt)[1] isa Matrix
@test get_tabular(dt)[2] isa Vector{String}
@test DT.is_tabular(dt) == true
@test DT.is_multidim(dt) == false

dt =DT.load_dataset(
    df,
    TreatmentGroup(
        dims=1,
        aggrfunc=reducesize(
            reducefunc=mean,
            win=(splitwindow(nwindows=5),)
        )
    );
)
multidim = get_multidim(dt)
@test multidim[1] isa Matrix{Union{Missing, Array{Float64}}}
@test multidim[2] isa Vector
@test DT.is_tabular(dt) == false
@test DT.is_multidim(dt) == true

@testset "DataTreatment API" begin
    dt = DT.load_dataset(df, t_classif)
    @test get_target(dt) == ["classA", "classB", "classC", "classA", "classB"]
    @test isa(get_discrete(dt), Tuple{AbstractMatrix, AbstractVector})
    @test isa(get_continuous(dt), Tuple{AbstractMatrix, AbstractVector})
    @test isa(get_tabular(dt), Tuple{AbstractMatrix, AbstractVector})
    @test size(get_tabular(dt)[1], 1) == nrow(df)

    dt = DT.load_dataset(df, t_regress)
    @test get_target(dt) == t_regress

    dt2 = DT.load_dataset(
        df,
        TreatmentGroup(
            dims=1,
            aggrfunc=DT.aggregate(
                features=(mean, maximum),
                win=(adaptivewindow(nwindows=5, overlap=0.4),)
            ),
            groupby=:feat
        )
    )
    agg = get_aggregated(dt2)
    @test length(dt2.treats) == 1
    @test dt2.treats[1].ids == [7, 10, 11, 13]
    @test dt2.treats[1].vnames == ["ts1", "ts2", "ts3", "ts4"]
    @test size(dt2.data[1].data, 2) == 40
    @test length(dt2.data[1].groups) == 2
    @test isa(agg, Tuple{AbstractMatrix, AbstractVector})
    @test size(agg[1], 1) == nrow(df)

    dt3 = DT.load_dataset(
        df,
        TreatmentGroup(
            dims=1,
            aggrfunc=reducesize(
                reducefunc=mean,
                win=(splitwindow(nwindows=5),)
            )
        )
    )
    red = get_multidim(dt3)
    @test isa(red, Tuple{AbstractMatrix, AbstractVector})
end

@testset "DT.load_dataset(dt::DataTreatment) round-trip" begin
    df = build_test_df()
    t_classif = ["classA", "classB", "classC", "classA", "classB"]

    # Basic round-trip: no treatments
    dt_orig = DT.load_dataset(df, t_classif)
    dt_rt = DT.load_dataset(dt_orig)

    @test isa(dt_rt, DT.DataTreatment)
    @test get_target(dt_rt) == get_target(dt_orig)
    @test DT.nrows(dt_rt) == DT.nrows(dt_orig)

    # Column count should be preserved
    orig_tab = get_tabular(dt_orig)
    rt_tab   = get_tabular(dt_rt)
    @test size(orig_tab[1]) == size(rt_tab[1])
    @test orig_tab[2] == rt_tab[2]

    # Round-trip preserves float_type kwarg
    dt_f32 = DT.load_dataset(dt_orig; float_type=Float32)
    @test dt_f32 isa DT.DataTreatment{Float32}

    # Round-trip with explicit DefaultTreatmentGroup
    dt_explicit = DT.load_dataset(dt_orig, DT.DefaultTreatmentGroup)
    @test isa(dt_explicit, DT.DataTreatment)
    @test DT.nrows(dt_explicit) == DT.nrows(dt_orig)

    # Round-trip with a custom treatment
    dt_custom = DT.load_dataset(
        df,
        TreatmentGroup(
            dims=1,
            aggrfunc=DT.aggregate(
                features=(mean, maximum),
                win=(adaptivewindow(nwindows=5, overlap=0.4),)
            ),
            groupby=:feat
        )
    )
    dt_rt_custom = DT.load_dataset(dt_custom)
    @test isa(dt_rt_custom, DT.DataTreatment)
    @test DT.nrows(dt_rt_custom) == DT.nrows(dt_custom)
end

n_rows = 100
n_cols = 10

# helper to sprinkle missing/NaN at given fraction
function make_col(n, frac)
    col = Vector{Union{Missing,Float64}}(rand(n))
    n_bad = round(Int, frac * n)
    idxs = randperm(n)[1:n_bad]
    for i in idxs
        col[i] = iseven(i) ? missing : NaN
    end
    return col
end

# build columns: 8 clean-ish (< 40%), 1 at ~40%, 1 at ~90%
cols = [make_col(n_rows, 0.05) for _ in 1:8]
push!(cols, make_col(n_rows, 0.40))   # col 9  → 40% bad
push!(cols, make_col(n_rows, 0.90))   # col 10 → 90% bad

data = Matrix(hcat(cols...))          # n_rows × n_cols

# build rows: make row 5 → ~90% bad, row 20 → ~40% bad
for j in 1:round(Int, 0.90 * n_cols)
    data[5,  j] = iseven(j) ? missing : NaN
end
for j in 1:round(Int, 0.40 * n_cols)
    data[20, j] = iseven(j) ? missing : NaN
end

vnames = ["V$i" for i in 1:n_cols]
target = categorical(rand(["A","B"], n_rows))

@testset "missing and nan filter" begin
    dt = DT.load_dataset(data, vnames, target)

    dt_filt = filter_missing(dt, 0.7, include_nans=true, dims=2)
    dt_filt = filter_missing(dt_filt, 0.7, include_nans=true, dims=1)

    @test size(dt_filt.data[1].data) == (99, 9)

    dt_filt = filter_missing(dt_filt, 0.3, include_nans=true, dims=2)
    dt_filt = filter_missing(dt_filt, 0.45, include_nans=true, dims=1)

    @test size(dt_filt.data[1].data) == (98, 8)
end
