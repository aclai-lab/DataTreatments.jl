using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using Statistics

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    V1 = [1.0, 2.0, 3.0, 4.0, 5.0],
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [create_image(i+10) for i in 1:5],
)

dt = DataTreatment(df)

@testset "_split_md_by_dims" begin
    mds = filter(d -> d isa MultidimDataset, get_dataset(dt))
    @test length(mds) >= 2
    for md in mds
        @test length(unique(get_dims(md))) == 1
    end
end

@testset "groupby :vname" begin
    ds = get_dataset(dt,
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean,)), groupby=:vname),
        groupby_split=true, leftover_ds=false, output_type=dataframe)
    @test length(ds) == 2  # ts1, ts2
    @test all(d -> nrow(d) == 5, ds)
end

@testset "groupby :feat" begin
    ds = get_dataset(dt,
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean, maximum)), groupby=:feat),
        groupby_split=true, leftover_ds=false, output_type=dataframe)
    @test length(ds) == 2  # mean, maximum
    @test all(d -> nrow(d) == 5, ds)
end

@testset "groupby (:vname, :feat)" begin
    ds = get_dataset(dt,
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean, maximum)), groupby=(:vname, :feat)),
        groupby_split=true, leftover_ds=false, output_type=dataframe)
    @test length(ds) == 4  # 2 vnames × 2 feats
    @test all(d -> nrow(d) == 5 && ncol(d) >= 1, ds)
end

@testset "groupby_split=false preserves column count" begin
    treat = TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean,)), groupby=:vname)
    ds_split = get_dataset(dt, treat, groupby_split=true, leftover_ds=false, output_type=matrix)
    ds_flat  = get_dataset(dt, treat, leftover_ds=false)
    mds = filter(d -> d isa MultidimDataset, ds_flat)
    @test sum(size(m, 2) for m in ds_split) == sum(length(md) for md in mds)
end