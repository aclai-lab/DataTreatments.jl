using DataTreatments
using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

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
    ts3 = [[1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2], NaN, NaN, missing, [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]],
    ts4 = [[6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8], missing, [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8], [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8], [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

dt = DataTreatment(df)

test1 = get_dataset(dt)

test2 = get_dataset(dt, matrix=true)
test3 = get_dataset(dt, dataframe=true)

test4 = get_dataset(
    dt,
    TreatmentGroup(
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
            )),
    dataframe=true
)

test5 = get_dataset(
    dt,
    TreatmentGroup(
        dims=1,
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    dataframe=true
)

test6 = get_dataset(
    dt,
    TreatmentGroup(
        dims=1,
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    TreatmentGroup(
        dims=2,
        aggrfunc=DT.aggregate(
            features=(minimum,),
            win=(DT.splitwindow(nwindows=3,),)
        )),
    dataframe=true
)

test7 = get_dataset(
    dt,
    TreatmentGroup(
        dims=1,
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    TreatmentGroup(
        dims=2,
        aggrfunc=DT.reducesize(
            reducefunc=minimum,
            win=(DT.splitwindow(nwindows=3,),)
        )),
    dataframe=true
)

test8 = get_dataset(
    dt,
    TreatmentGroup(
        name_expr=r"^(V|i)"
    ),
    dataframe=true
)

test9 = get_dataset(
    dt,
    TreatmentGroup(
        dims=1,
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
        ),
        groupby=(:vname, :feat),
    ),
    groupby_split=true,
    dataframe=true
)