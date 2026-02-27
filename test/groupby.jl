using Test
using DataTreatments
const DT = DataTreatments

using MLJ, DataFrames
using SoleData: Artifacts

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                             DataFrame groupby                                #
# ---------------------------------------------------------------------------- #
@testset "groupby single column" begin
    fields1 = [:sepal_length]
    groups1 = DT.groupby(Xc, fields1)

    @test length(groups1) == 2
    @test propertynames(groups1[2]) == [:sepal_width, :petal_length, :petal_width]

    fields2 = :sepal_length
    groups2 = DT.groupby(Xc, fields2)

    @test length(groups2) == 2
    @test propertynames(groups2[2]) == [:sepal_width, :petal_length, :petal_width]

    @test groups1 == groups2
end

@testset "groupby adds leftover columns" begin
    fields = [[:sepal_length, :petal_length], [:sepal_width]]
    groups = DT.groupby(Xc, fields)
    @test length(groups) == 3
    @test propertynames(groups[3]) == [:petal_width]
end

# ---------------------------------------------------------------------------- #
#                       DataTreatment external groupby                         #
# ---------------------------------------------------------------------------- #
tab_no_grp = DataTreatment(Xc, yc)

win = adaptivewindow(nwindows=3, overlap=0.2)
features = (mean, maximum)
rs_no_grp = DataTreatment(Xts, yts; aggrtype=:reducesize, win)
ag_no_grp = DataTreatment(Xts, yts; aggrtype=:aggregate, win, features)

@testset "groupby on tabular DataTreatment" begin
    mask = BitVector([1, 0, 0, 1])
    groups_mask = DT.groupby(tab_no_grp, mask)
    @test length(collect(groups_mask)) == 2

    groups_single = DT.groupby(tab_no_grp, [[:sepal_length]])
    @test length(collect(groups_single)) == 2

    groups_multi = DT.groupby(tab_no_grp, [[:sepal_length, :petal_length], [:petal_width]])
    @test length(collect(groups_multi)) == 3
end

@testset "groupby on rs_no_grp by :vname" begin
    rs_featureids = get_datafeature(rs_no_grp)
    rs_vnames = unique(get_vname.(rs_featureids))

    groups = DT.groupby(rs_no_grp, :vname)
    collected = collect(groups)

    @test length(collected) == length(rs_vnames)

    for (i, vn) in enumerate(rs_vnames)
        expected = findall(fid -> get_vname(fid) == vn, rs_featureids)
        @test collected[i] == expected
    end
end

@testset "groupby on ag_no_grp by [:vname, :feat]" begin
    ag_featureids = get_datafeature(ag_no_grp)
    ag_vnames = unique(get_vname.(ag_featureids))

    groups = DT.groupby(ag_no_grp, [:vname, :feat])
    collected = reduce(vcat, collect.(collect(groups)))

    # manually build expected groups
    expected = Vector{Vector{Int}}()
    for vn in ag_vnames
        vn_idxs = findall(fid -> get_vname(fid) == vn, ag_featureids)
        feats = unique(get_feat.(ag_featureids[vn_idxs]))
        for f in feats
            push!(expected, filter(i -> get_feat(ag_featureids[i]) == f, vn_idxs))
        end
    end

    @test length(collected) == length(expected)
    @test all(collected .== expected)
end
# ---------------------------------------------------------------------------- #
#                       DataTreatment internal groupby                         #
# ---------------------------------------------------------------------------- #
@testset "internal groupby on tabular DataTreatment" begin
    tab_mask   = DataTreatment(Xc, yc; groups=BitVector([1, 0, 0, 1]))
    tab_single = DataTreatment(Xc, yc; groups=[[:sepal_length]])
    tab_multi  = DataTreatment(Xc, yc; groups=[[:sepal_length, :petal_length], [:petal_width]])

    @test length(get_groups(tab_mask))   == 2
    @test length(get_groups(tab_single)) == 2
    @test length(get_groups(tab_multi))  == 3
end

@testset "internal groupby on rs DataTreatment" begin
    win = adaptivewindow(nwindows=3, overlap=0.2)

    # baseline: raw datafeature vector, no grouping applied
    rs_feats = get_datafeature(DataTreatment(Xts, yts; aggrtype=:reducesize, win))

    rs_type  = DataTreatment(Xts, yts; groups=:type,          aggrtype=:reducesize, win)
    rs_vname = DataTreatment(Xts, yts; groups=:vname,         aggrtype=:reducesize, win)
    rs_rfunc = DataTreatment(Xts, yts; groups=:reducefunc,    aggrtype=:reducesize, win)
    rs_multi = DataTreatment(Xts, yts; groups=[:vname, :type], aggrtype=:reducesize, win)

    # :type grouping
    groups_type = get_groups(rs_type)
    @test length(groups_type) == length(unique(get_type.(rs_feats)))
    for (i, t) in enumerate(unique(get_type.(rs_feats)))
        @test groups_type[i] == findall(f -> get_type(f) == t, rs_feats)
    end

    # :vname grouping
    groups_vname = get_groups(rs_vname)
    @test length(groups_vname) == length(unique(get_vname.(rs_feats)))
    for (i, vn) in enumerate(unique(get_vname.(rs_feats)))
        @test groups_vname[i] == findall(f -> get_vname(f) == vn, rs_feats)
    end

    # :reducefunc grouping
    groups_rfunc = get_groups(rs_rfunc)
    @test length(groups_rfunc) == length(unique(get_reducefunc.(rs_feats)))
    for (i, rf) in enumerate(unique(get_reducefunc.(rs_feats)))
        @test groups_rfunc[i] == findall(f -> get_reducefunc(f) == rf, rs_feats)
    end

    # [:vname, :type] multi-level grouping
    groups_multi = get_groups(rs_multi)
    expected_multi = Vector{Vector{Int}}()
    for vn in unique(get_vname.(rs_feats))
        vn_idxs = findall(f -> get_vname(f) == vn, rs_feats)
        for t in unique(get_type.(rs_feats[vn_idxs]))
            push!(expected_multi, filter(i -> get_type(rs_feats[i]) == t, vn_idxs))
        end
    end
    @test length(groups_multi) == length(expected_multi)
    @test all(groups_multi .== expected_multi)
end

@testset "internal groupby on ag DataTreatment" begin
    win = adaptivewindow(nwindows=3, overlap=0.2)
    features = (mean, maximum)

    # baseline: raw datafeature vector, no grouping applied
    ag_feats = get_datafeature(DataTreatment(Xts, yts; aggrtype=:aggregate, win, features))

    ag_type  = DataTreatment(Xts, yts; groups=:type,  aggrtype=:aggregate, win, features)
    ag_vname = DataTreatment(Xts, yts; groups=:vname, aggrtype=:aggregate, win, features)
    ag_nwin  = DataTreatment(Xts, yts; groups=:nwin,  aggrtype=:aggregate, win, features)
    ag_feat  = DataTreatment(Xts, yts; groups=:feat,  aggrtype=:aggregate, win, features)
    ag_multi = DataTreatment(Xts, yts; groups=[:vname, :nwin], aggrtype=:aggregate, win, features)

    # :type grouping
    groups_type = get_groups(ag_type)
    @test length(groups_type) == length(unique(get_type.(ag_feats)))
    for (i, t) in enumerate(unique(get_type.(ag_feats)))
        @test groups_type[i] == findall(f -> get_type(f) == t, ag_feats)
    end

    # :vname grouping
    groups_vname = get_groups(ag_vname)
    @test length(groups_vname) == length(unique(get_vname.(ag_feats)))
    for (i, vn) in enumerate(unique(get_vname.(ag_feats)))
        @test groups_vname[i] == findall(f -> get_vname(f) == vn, ag_feats)
    end

    # :nwin grouping
    groups_nwin = get_groups(ag_nwin)
    @test length(groups_nwin) == length(unique(get_nwin.(ag_feats)))
    for (i, nw) in enumerate(unique(get_nwin.(ag_feats)))
        @test groups_nwin[i] == findall(f -> get_nwin(f) == nw, ag_feats)
    end

    # :feat grouping
    groups_feat = get_groups(ag_feat)
    @test length(groups_feat) == length(unique(get_feat.(ag_feats)))
    for (i, ft) in enumerate(unique(get_feat.(ag_feats)))
        @test groups_feat[i] == findall(f -> get_feat(f) == ft, ag_feats)
    end

    # [:vname, :nwin] multi-level grouping
    groups_multi = get_groups(ag_multi)
    expected_multi = Vector{Vector{Int}}()
    for vn in unique(get_vname.(ag_feats))
        vn_idxs = findall(f -> get_vname(f) == vn, ag_feats)
        for nw in unique(get_nwin.(ag_feats[vn_idxs]))
            push!(expected_multi, filter(i -> get_nwin(ag_feats[i]) == nw, vn_idxs))
        end
    end
    @test length(groups_multi) == length(expected_multi)
    @test all(groups_multi .== expected_multi)
end