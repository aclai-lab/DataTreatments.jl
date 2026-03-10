# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
_get_features(a::Base.Callable) = a.features
_get_reducefunc(r::Base.Callable) = r.reducefunc

"""
    discrete_encode(X::Matrix) -> (codes, levels)

Encode each column of `X` as a categorical variable.

`missing` values are **not** categorized: they are preserved as
`missing` in the output `codes` and are excluded from the level labels.

# Arguments
- `X::Matrix`: a matrix whose columns contain discrete values of any type.

# Returns
- `codes`: a vector of `Vector{Union{Missing,Int}}`, where `codes[i]` contains
  the integer level codes for column `i`. `missing` and entries in the
  original column are mapped to `missing` (not assigned a level code).
- `levels`: a vector of `Vector{String}`, where `levels[i]` contains the sorted
  unique string labels for column `i`, such that `levels[i][codes[i][j]]`
  reconstructs the original value of `X[j, i]` for non-missing entries.
"""
function discrete_encode(X::Matrix)
    to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
    cats = [categorical(to_str.(col)) for col in eachcol(X)]
    return [levelcode.(cat) for cat in cats], levels.(cats)
end

# ---------------------------------------------------------------------------- #
#                               dataset structs                                #
# ---------------------------------------------------------------------------- #
struct DiscreteDataset <: AbstractDataset
    dataset::Matrix
    info::Vector{DiscreteFeat}

    DiscreteDataset(dataset::Matrix, info::Vector{DiscreteFeat}) = new(dataset, info)
    
    function DiscreteDataset(
        id::Vector,
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int}
    )
        T = get_datatype(ds_struct, cols)
        vnames = get_vnames(ds_struct, cols)
        idx = get_valididxs(ds_struct, cols)
        miss = get_missingidxs(ds_struct, cols)
        codes, levels = discrete_encode(dataset[:, cols])

        return DiscreteDataset(
            stack(codes),
            [DiscreteFeat{T[i]}(push!(id, i), vnames[i], levels[i], idx[i], miss[i])
                for i in eachindex(vnames)]
        )
    end
end

struct ContinuousDataset{T} <: AbstractDataset
    dataset::Matrix
    info::Vector{ContinuousFeat}

    ContinuousDataset(dataset::Matrix, info::Vector{ContinuousFeat{T}}) where T =
        new{T}(dataset, info)

    function ContinuousDataset(
        id::Vector,
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int},
        float_type::Type
    )
        vnames = get_vnames(ds_struct, cols)
        idx = get_valididxs(ds_struct, cols)
        miss = get_missingidxs(ds_struct, cols)
        nan = get_nanidxs(ds_struct, cols)

        return ContinuousDataset(
            reduce(hcat, [map(x -> ismissing(x) ? missing : float_type(x), @view dataset[:, col])
                for col in cols]),
            [ContinuousFeat{float_type}(push!(id, i), vnames[i], idx[i], miss[i], nan[i])
                for i in eachindex(vnames)]
        )
    end
end

struct MultidimDataset{T} <: AbstractDataset
    dataset::Matrix
    info::Vector{Union{AggregateFeat,ReduceFeat}}

    MultidimDataset(dataset::Matrix, info::Vector{AggregateFeat{T}}) where T =
        new{T}(dataset, info)
    MultidimDataset(dataset::Matrix, info::Vector{ReduceFeat{T}}) where T =
        new{T}(dataset, info)

    function MultidimDataset(
        id::Vector,
        dataset::Matrix, 
        ds_struct::DatasetStructure,
        cols::Vector{Int},
        aggrfunc::Base.Callable,
        float_type::Type
    )
        dataset = @view dataset[:, cols]
        vnames = get_vnames(ds_struct, cols)
        idx = get_valididxs(ds_struct, cols)
        miss = get_missingidxs(ds_struct, cols)
        nan = get_nanidxs(ds_struct, cols)
        hasmiss = get_hasmissing(ds_struct, cols)
        hasnan = get_hasnans(ds_struct, cols)

        md, nwindows = aggrfunc(dataset, idx, float_type)

        md_feats = if hasfield(typeof(aggrfunc), :features)
            vec([AggregateFeat{float_type}(
                push!(id, i),
                vnames[c],
                f,
                nwindows[c],
                idx[c],
                miss[c],
                nan[c],
                hasmiss[c],
                hasnan[c])
                for (i, (f, c)) in enumerate(Iterators.product(_get_features(aggrfunc), axes(dataset,2)))])
        else
            [ReduceFeat{AbstractArray{float_type}}(
                push!(id, i),
                vnames[c],
                _get_reducefunc(aggrfunc),
                idx[c],
                miss[c],
                nan[c],
                hasmiss[c],hasnan[c])
                for (i, c) in enumerate(axes(dataset,2))]
        end

        return MultidimDataset(md, md_feats)
    end
end