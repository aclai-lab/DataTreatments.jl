using Test
using DataTreatments

using SoleData: Artifacts
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                           windowing and treatment                            #
# ---------------------------------------------------------------------------- #
win = adaptivewindow(nwindows=3, overlap=0.2)
features = (mean, maximum)

rs = DataTreatment(Xts, :reducesize; win)
ag = DataTreatment(Xts, :aggregate; win, features)

# macro groupby(x, idlabels...)
#     esc_idlabels = map(esc, idlabels)
#     quote
#         _x = $(esc(x))
#         _l = ($(esc_idlabels...),)
#         ds = _x.dataset
#         nid = length(_l)
        
#     #     # apply each window function to corresponding dimension
#     #     # if more dims than functions, reuse the last function
#     #     tuple((
#     #         let idx = min(i, length(_w))
#     #             _w[idx](dims[i])
#     #         end
#     #         for i in 1:length(dims)
#     #     )...)
#     end
# end

function groupby(d::DataTreatment, l::Symbol)

end

"""
    groupby(dt::DataTreatment, field::Symbol) -> Dict

Group feature indices by values in the specified `FeatureId` field.

# Arguments
- `dt::DataTreatment`: The data treatment object containing feature metadata
- `field::Symbol`: Field to group by (`:vname`, `:feat`, or `:nwin`)

# Returns
A dictionary mapping field values to vectors of feature indices.

# Examples
```julia
# Group by variable name
vname_groups = groupby(dt, :vname)
# Returns: Dict(:channel1 => [1,2,3,...], :channel2 => [10,11,12,...], ...)

# Group by feature function
feat_groups = groupby(dt, :feat)
# Returns: Dict(mean => [1,4,7,...], std => [2,5,8,...], ...)

# Group by window number
win_groups = groupby(dt, :nwin)
# Returns: Dict(1 => [1,2,3,...], 2 => [4,5,6,...], ...)
```
"""
function groupby(d::DataTreatment, field::Symbol)
    field in fieldnames(FeatureId) || 
        throw(ArgumentError("field must be one of $(fieldnames(FeatureId))"))

    getter = @eval $(Symbol(:get_, field))  
    featureids = get_featureid(d)
    feats = unique(getter.(featureids))

    groups = Dict{eltype(feats), Vector{Int}}()

    for f in feats
        groups[f] = findall(fid -> getter(fid) == f, featureids)
    end

    return groups, feats
end
