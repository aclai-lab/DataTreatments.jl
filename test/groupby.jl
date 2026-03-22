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
    ts1 = [
        collect(1.0:6.0),
        collect(2.0:7.0),
        collect(3.0:8.0),
        collect(4.0:9.0),
        collect(5.0:10.0)
    ],
    ts2 = [
        collect(2.0:0.5:5.5),
        collect(1.0:0.5:4.5),
        collect(3.0:0.5:6.5),
        collect(4.0:0.5:7.5),
        collect(5.0:0.5:8.5)
    ],
    img1 = [create_image(i) for i in 1:5],
    img2 = [create_image(i+10) for i in 1:5],
)

dt = DataTreatment(df)

@testset "_split_md_by_dims" begin
    mds = filter(d -> d isa MultidimDataset, first(get_dataset(dt)))
    @test length(mds) >= 2
    for md in mds
        @test length(unique(get_dims(md))) == 1
    end
end

@testset "groupby :vname" begin
    ds, _ = get_dataset(dt,
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean,)), groupby=:vname),
        leftover_ds=false)
    @test get_ncols(ds[1]) == 2  # ts1, ts2
    @test get_nrows(ds[1]) == 5
end

@testset "groupby :feat" begin
    ds, _ = get_dataset(dt,
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean, maximum)), groupby=:feat),
        leftover_ds=false)
    @test get_ncols(ds[1]) == 4  # mean, maximum * ts1, ts2
    @test get_nrows(ds[1]) == 5
end

@testset "groupby (:vname, :feat)" begin
    ds, _ = get_dataset(dt,
        TreatmentGroup(dims=1, aggrfunc=DT.aggregate(features=(mean, maximum)), groupby=(:vname, :feat)),
        leftover_ds=false)
    groups = get_groups(ds[1])
    @test length(groups) == 4  # 2 vnames × 2 feats
end
