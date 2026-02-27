a=[:a,:b,:c,:a,:a,:c,:b]

unique_vals = unique(a)

@btime groups = (findall(==(v), a) for v in unique_vals)

@btime groups = [findall(==(v), a) for v in unique_vals]

@btime groups = map(v -> findall(==(v), a), unique_vals)

groups = (findall(==(v), a) for v in unique_vals)

using DataFrames
using DataTreatments
using SoleData: Artifacts
natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)
X = Xts[1:10, 1:5]
a = DataTreatment(X, win=splitwindow(nwindows=2), groups=:all)
AbstractDataFeature = DataTreatments.AbstractDataFeature
datafeats = a.datafeature
field = :vname

# function groupby(::Matrix{T}, f::Vector{<:AbstractDataFeature}, args...) where T
#     groupby((1:length(f)), f, args...)
# end

# idxs = (1:length(f))
# idxs = eachindex(f)

# const FIELD_GETTERS = Dict{Symbol, Function}(
#     :type  => get_type,
#     :vname => get_vname,
#     :nwin  => get_nwin,
#     :feat  => get_feat,
# )

# @inline field_getter(field::Symbol) =
#     haskey(FIELD_GETTERS, field) ? FIELD_GETTERS[field] : throw(ArgumentError("Unknown field: $field"))

@inline field_getter(field::Symbol) =
    field == :type ? get_type :
    field == :vname ? get_vname :
    field == :nwin ? get_nwin :
    field == :feat ? get_feat :
    throw(ArgumentError("Unknown field: $field"))

function groupby(
    datafeats::Vector{<:AbstractDataFeature},
    field::Symbol
)
    getter = field_getter(field)
    vals = getter.(datafeats)
    unique_vals = unique(vals)
    idxs = (findall(==(v), vals) for v in unique_vals)
    return (get_id.(@view(datafeats[i])) for i in idxs)
end

@btime test = groupby(datafeats, :vname)
# 1.034 μs (18 allocations: 1.06 KiB)

test = groupby(datafeats, :vname)

q=[groupby(datafeats, :vname),groupby(datafeats, :nwin)]

# for i in test
#     @show datafeats[i]
# end

function groupby(
    datafeats::Vector{<:AbstractDataFeature},
    fields::Vector{Symbol}
)
    # this function performs multi-level grouping (recursive).
    # - idxs: current groups of column indices
    # - datafeats: current groups of FeatureId metadata (aligned with idxs)
    # - fields: remaining fields to group by (e.g., [:feat, :vname, :nwin])

    # split by the first field
    sub_idxs = groupby(datafeats, first(fields))

    isempty(fields[2:end]) && return sub_idxs

    # recursively group each sub-group by remaining fields
    all_groups = Vector{Base.Generator}()

    for i in sub_idxs
        groups = groupby(datafeats[i], fields[2:end])
        push!(all_groups, groups)
    end

    return all_groups
end

@btime groupby(datafeats, [:vname, :nwin])
# 3.682 μs (102 allocations: 5.03 KiB)

groupidxs = groupby(datafeats, [:vname, :nwin])

r = DataTreatments.groupby(datafeats, [:vname, :nwin])

using Normalization

a = DataTreatment(Xts, groups=[:vname, :nwin])
X = a.X
groupidxs = a.metadata.groups
norm = MinMax

@btime begin
    if !isnothing($norm)
        Threads.@threads for g in groupidxs
            normalize!(@view(X[:, reduce(vcat, g)]), norm)
        end
        # flat_groups = [reduce(vcat, g) for g in groupidxs]
        # Threads.@threads for g in flat_groups
        #     normalize!(@view(X[:, g]), norm)
        # end
    end
end
# 154.860 μs (1968 allocations: 475.88 KiB)
# 41.894 μs (1984 allocations: 477.19 KiB)
# 53.484 μs (2272 allocations: 289.50 KiB)
# 46.363 μs (1912 allocations: 76.31 KiB)


Threads.@threads for g in groupidxs
    normalize!(@view(X[:, collect(g)]), norm)
end

foreach(g -> normalize!(@view(X[:, g...]), norm), groupidxs)



DataTreatment(X, norm=Robust{ZScore})

DataTreatment(X, norm=Robust{Scale})

DataTreatment(X, norm=PNorm1)

DataTreatment(X, norm=ScaleMad)


