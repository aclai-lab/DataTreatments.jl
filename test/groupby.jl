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
#                           windowing and treatment                            #
# ---------------------------------------------------------------------------- #
win = adaptivewindow(nwindows=3, overlap=0.2)
features = (mean, maximum)

rs_no_grp = DataTreatment(Xts, :reducesize; win)
ag_no_grp = DataTreatment(Xts, :aggregate; win, features)

# ---------------------------------------------------------------------------- #
#                             DataFrame groupby                                #
# ---------------------------------------------------------------------------- #
@testset "groupby single column" begin
    fields = [:sepal_length]

    groups = DT.groupby(Xc, fields)
    @test length(groups) == 2
    @test propertynames(groups[2]) == [:sepal_width, :petal_length, :petal_width]
end

@testset "groupby adds leftover columns" begin
    fields = [[:sepal_length, :petal_length], [:sepal_width]]

    groups = DT.groupby(Xc, fields)
    @test length(groups) == 3
    @test propertynames(groups[3]) == [:petal_width]
end

# ---------------------------------------------------------------------------- #
#                           DataTreatment groupby                              #
# ---------------------------------------------------------------------------- #
rs_grp = DataTreatment(Xts, :reducesize; win, groups=(:vname,))
ag_grp = DataTreatment(Xts, :aggregate; win, features, groups=(:vname, :feat))

@testset "Manual groupby vs built-in groupby" begin
    # manual grouping for rs_no_grp by :vname
    rs_featureids = get_featureid(rs_no_grp)
    rs_vnames = unique(get_vname.(rs_featureids))
    rs_manual_groups = [findall(fid -> get_vname(fid) == vn, rs_featureids) for vn in rs_vnames]
    
    @test length(rs_manual_groups) == length(get_groups(rs_grp))
    @test all(rs_manual_groups .== [gr.group for gr in get_groups(rs_grp)])
    
    # manual grouping for ag_no_grp by :vname then :feat
    ag_featureids = get_featureid(ag_no_grp)
    ag_vnames = unique(get_vname.(ag_featureids))
    
    # first level: group by vname
    ag_manual_groups_l1 = [findall(fid -> get_vname(fid) == vn, ag_featureids) for vn in ag_vnames]
    
    # second level: within each vname group, split by feat
    ag_manual_groups = Vector{Vector{Int64}}()
    for group_l1 in ag_manual_groups_l1
        feats_in_group = unique(get_feat.(ag_featureids[group_l1]))
        for f in feats_in_group
            mask = findall(i -> get_feat(ag_featureids[i]) == f, group_l1)
            push!(ag_manual_groups, group_l1[mask])
        end
    end
    
    @test length(ag_manual_groups) == length(get_groups(ag_grp))
    @test all(ag_manual_groups .== [gr.group for gr in get_groups(ag_grp)])
end

@testset "DataTreatments groupby vs built-in groupby" begin
    # test groupby function on rs_no_grp by :vname
    rs_groupby_groups, _ = DT.groupby(rs_no_grp, :vname)
    rs_builtin_groups = [gr.group for gr in get_groups(rs_grp)]
    
    @test length(rs_groupby_groups) == length(rs_builtin_groups)
    @test all(rs_groupby_groups .== rs_builtin_groups)
    
    # test groupby function on ag_no_grp by :vname then :feat
    ag_groupby_groups, _ = DT.groupby(ag_no_grp, :vname, :feat)
    ag_builtin_groups = [gr.group for gr in get_groups(ag_grp)]
    
    @test length(ag_groupby_groups) == length(ag_builtin_groups)
    @test all(ag_groupby_groups .== ag_builtin_groups)
end