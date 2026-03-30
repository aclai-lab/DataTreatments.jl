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

dt = load_dataset(df)

# using Impute

# data = dt.data[2].data

# Impute.declaremissings(data; values=(NaN, "NULL"))

# # Impute.interp(data)

# Impute.interp(data) |> Impute.locf() |> Impute.nocb()

# using Impute: Interpolate, impute!, impute

# d = dt.data[2].data

# impute!(d, Interpolate(); dims=2)

# impute(data, Interpolate(); dims=2)
# impute(Impute.declaremissings(data; values=(NaN, "NULL")), Impute.KNN(); dims=2)
# impute(data, Impute.LOCF(); dims=2)
# impute(data, Impute.NOCB(); dims=2)
# impute(data, Impute.Substitute(statistic=Impute.defaultstats); dims=2)
# impute(data, Impute.Substitute(statistic=mean); dims=2)
# impute(data, Impute.SVD(); dims=2)


dt = load_dataset(
    df,
    TreatmentGroup(
        dims=0,
        impute=(LOCF(),)
    )
)

dt = load_dataset(
    df,
    TreatmentGroup(
        dims=0,
        impute=(LOCF(), NOCB()),
        datatype=:discrete
    ),
    TreatmentGroup(
        dims=0,
        impute=(SVD(),),
        datatype=:continuous
    )
)

dt = load_dataset(
    df,
    TreatmentGroup(
        dims=0,
        impute=(Interpolate(), LOCF(), NOCB())
    )
)

dt = load_dataset(
    df,
    TreatmentGroup(
        dims=1,
        impute=(SVD(),)
    ),
    TreatmentGroup(
        dims=2,
        impute=(SVD(),)
    )
)

dt = load_dataset(
    df,
    TreatmentGroup(
        dims=1,
        impute=(Interpolate(), LOCF(), NOCB())
    ),
    TreatmentGroup(
        dims=2,
        impute=(Interpolate(), LOCF(), NOCB())
    )
)

# using Impute

# data = dt.data[1].data

# Impute.declaremissings(data; values=(NaN, "NULL"))

# # Impute.interp(data)

# Impute.interp(data) |> Impute.locf() |> Impute.nocb()

# using Impute: Interpolate, impute!, impute

# d = dt.data[1].data

# impute!(d, Interpolate(); dims=2)

# impute(data, Interpolate(r=RoundNearest); dims=2)
# impute(Impute.declaremissings(data; values=(NaN, "NULL")), Impute.KNN(); dims=2)
# impute(data, Impute.LOCF(); dims=2)
# impute(data, Impute.NOCB(); dims=2)
# impute(data, Impute.Substitute(statistic=Impute.defaultstats); dims=2)
# impute(data, Impute.Substitute(statistic=mean, ); dims=2)
# impute(data, Impute.SVD(); dims=2)

