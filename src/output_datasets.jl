# ---------------------------------------------------------------------------- #
#                              metadata structs                                #
# ---------------------------------------------------------------------------- #
"""
    DiscreteFeat{T} <: AbstractDataFeature

Metadata for a discrete (categorical) feature in a dataset.

# Fields
- `id::Int`: Unique identifier for the feature (column index in the source data).
- `vname::String`: Original column name.
- `levels::CategoricalArrays.CategoricalVector`: Categorical vector of levels.
- `valididxs::Vector{Int}`: Indices of valid (non-missing) entries.
- `missingidxs::Vector{Int}`: Indices of missing entries.
- `datatype`: Original column datatype
"""
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    datatype::Type
end

"""
    ContinuousFeat{T} <: AbstractDataFeature

Metadata for a continuous (numeric scalar) feature in a dataset.

# Fields
- `id::Int`: Unique identifier for the feature (column index in the source data).
- `vname::String`: Original column name.
- `valididxs::Vector{Int}`: Indices of valid (non-missing, non-NaN) entries.
- `missingidxs::Vector{Int}`: Indices of missing entries.
- `nanidxs::Vector{Int}`: Indices of NaN entries.
"""
struct ContinuousFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
end

"""
    AggregateFeat{T} <: AbstractDataFeature

Metadata for an aggregated feature derived from a multidimensional column.

# Fields
- `id::Int`: Source column index.
- `subid::Int`: Sub-feature index (for window/feature combinations).
- `vname::String`: Original column name.
- `dims::Int`: Dimensionality of the source data.
- `feat::Base.Callable`: Aggregation function used.
- `nwin::Int`: Window index.
- `valididxs::Vector{Int}`: Indices of valid entries.
- `missingidxs::Vector{Int}`: Indices of missing entries.
- `nanidxs::Vector{Int}`: Indices of NaN entries.
- `hasmissing::Vector{Int}`: Indices where the array contains internal `missing`.
- `hasnans::Vector{Int}`: Indices where the array contains internal `NaN`.
"""
struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    subid::Int
    vname::String
    dims::Int
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}
end

"""
    ReduceFeat{T} <: AbstractDataFeature

Metadata for a reduced-size feature derived from a multidimensional column.

# Fields
- `id::Int`: Source column index.
- `vname::String`: Original column name.
- `dims::Int`: Dimensionality of the source data.
- `reducefunc::Base.Callable`: Reduction function used.
- `valididxs::Vector{Int}`: Indices of valid entries.
- `missingidxs::Vector{Int}`: Indices of missing entries.
- `nanidxs::Vector{Int}`: Indices of NaN entries.
- `hasmissing::Vector{Int}`: Indices where the array contains internal `missing`.
- `hasnans::Vector{Int}`: Indices where the array contains internal `NaN`.
"""
struct ReduceFeat{T} <: AbstractDataFeature
    id::Int
    vname::String
    dims::Int
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    hasmissing::Vector{Int}
    hasnans::Vector{Int}
end

# Base.eltype(::ContinuousFeat{T}) where T = T
Base.eltype(::AggregateFeat{T}) where T = T
Base.eltype(::ReduceFeat{T}) where T = T

get_subid(f::AggregateFeat) = f.subid
get_dims(f::Union{AggregateFeat,ReduceFeat}) = f.dims

get_vnames(d::AbstractDataFeature) = d.vname

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
_get_features(a::Base.Callable) = a.features
_get_reducefunc(r::Base.Callable) = r.reducefunc

"""
    _reindex_groups(groups, idxs) -> Union{Nothing, Vector{Vector{Int}}}

Re-map group indices from the original column space to the new subset `idxs`.
Only groups that have at least one member in `idxs` are kept.
"""
function _reindex_groups(groups::Vector{Vector{Int}}, idxs::AbstractVector{Int})
    idx_set = Set(idxs)
    # Build reverse mapping: old index -> new index
    old_to_new = Dict(old => new for (new, old) in enumerate(idxs))
    
    new_groups = Vector{Vector{Int}}()

    for grp in groups
        new_grp = [old_to_new[i] for i in grp if i in idx_set]
        if !isempty(new_grp)
            push!(new_groups, new_grp)
        end
    end
    
    return new_groups
end

_reindex_groups(
    groups::Nothing,
    idxs::AbstractVector{Int}
) = Vector{Vector{Int}}()

function _impute(data::Matrix, impute::Tuple{Vararg{<:Impute.Imputor}})
    Impute.declaremissings(data; values=(NaN, "NULL"))
    for im in impute
        Impute.impute!(data, im; dims=2)
    end
    any(ismissing.(data)) || (data = disallowmissing(data))

    return data
end

# ---------------------------------------------------------------------------- #
#                           output dataset structs                             #
# ---------------------------------------------------------------------------- #
"""
    DiscreteDataset <: AbstractDataset

Output dataset for **discrete (categorical)** columns 
collected by `DataTreatment`.

# Fields
- `data::AbstractMatrix`: Encoded integer-coded matrix 
  (one column per discrete feature).
  Each value is either an `Int` level code or `missing`.
- `info::Vector{<:DiscreteFeat}`: Per-column metadata, 
  including the original column name, categorical levels, validity indices, 
  and an `id` vector tracing the column 
  back to the source dataset and `TreatmentGroup`.

# Constructors

- `DiscreteDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple
    )`  
  Internal constructor: selects columns `ids` from `data`, 
  encodes them categorically, and builds the corresponding 
  [`DiscreteFeat`](@ref) metadata from `datastruct`.

## Arguments for internal constructor
- `ids::Vector{Int}`: Column indices to include in this dataset.
- `data::AbstractMatrix`: The full raw dataset matrix.
- `vnames::Vector{String}`: Names of all columns.
- `datastruct::NamedTuple`: Pre-computed dataset metadata.

See also: [`ContinuousDataset`](@ref), [`MultidimDataset`](@ref), 
[`DiscreteFeat`](@ref)
"""
mutable struct DiscreteDataset{T} <: AbstractDataset
    data::AbstractMatrix
    info::Vector{<:DiscreteFeat}

    DiscreteDataset(
        data::AbstractMatrix,
        info::Vector{<:DiscreteFeat{T}}
    ) where T = new{T}(data, info)
    
    function DiscreteDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}}
    )
        codes = _discrete_encode(@views(data[:, ids]))
        vnames = vnames[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        datatype = datastruct.datatype[ids]

        data = isempty(codes) ?
            Matrix{eltype(codes)}(undef, 0, 0) :
            stack(codes)

        isnothing(impute) || (data = _impute(data, impute))

        return new{eltype(codes)}(
            data,
            [DiscreteFeat{eltype(codes)}(
                ids[i],
                vnames[i],
                valid[i],
                miss[i],
                datatype[i]
            )
                for i in eachindex(ids)]
        )
    end
end

"""
    ContinuousDataset{T} <: AbstractDataset

Output dataset for **continuous (numeric scalar)** columns produced by `DataTreatment`.

Each selected column is cast to the target `float_type` (e.g., `Float64`), with
`missing` values preserved. The resulting `data` matrix is a numeric matrix
ready for downstream ML pipelines, and each column is described by a
[`ContinuousFeat`](@ref) metadata entry in `info`.

# Type Parameter
- `T`: The floating-point type used for numeric conversion (e.g., `Float64`, `Float32`).

# Fields
- `data::AbstractMatrix`: Numeric matrix (one column per continuous feature), with
  elements of type `Union{Missing, T}`.
- `info::Vector{<:ContinuousFeat}`: Per-column metadata, including the original
  column name, validity indices, missing indices, NaN indices, and an `id` vector
  tracing the column back to the source dataset and `TreatmentGroup`.

# Constructors

- `ContinuousDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        float_type::Type
    )`  
  Internal constructor: selects columns `ids` from `data`, converts each element to `float_type`
  (preserving `missing`), and builds the corresponding [`ContinuousFeat`](@ref) metadata from `datastruct`.

## Arguments for internal constructor
- `ids::Vector{Int}`: Column indices to include in this dataset.
- `data::AbstractMatrix`: The full raw dataset matrix.
- `vnames::Vector{String}`: Names of all columns.
- `datastruct::NamedTuple`: Pre-computed dataset metadata.
- `float_type::Type`: Target floating-point type for numeric conversion.

See also: [`DiscreteDataset`](@ref), [`MultidimDataset`](@ref), [`ContinuousFeat`](@ref)
"""
mutable struct ContinuousDataset{T} <: AbstractDataset
    data::AbstractMatrix
    info::Vector{<:ContinuousFeat}

    ContinuousDataset(
        data::AbstractMatrix,
        info::Vector{<:ContinuousFeat{T}}
    ) where T = new{T}(data, info)

    function ContinuousDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}},
        float_type::Type
    )
        vnames = vnames[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        nan = datastruct.nanidxs[ids]

        data = if isempty(ids)
            Matrix{float_type}(undef, 0, 0)
        else
            reduce(hcat, [map(x -> ismissing(x) ?
                missing :
                float_type(x), @view data[:, id])
                for id in ids]
            )
        end

        isnothing(impute) || (data = _impute(data, impute))

        return new{float_type}(
            data,
            [ContinuousFeat{float_type}(
                ids[i],
                vnames[i],
                valid[i],
                miss[i],
                nan[i]
            )
                for i in eachindex(ids)]
        )
    end
end

"""
    MultidimDataset{T} <: AbstractDataset

Output dataset for **multidimensional** columns produced by `DataTreatment`.

Handles columns whose elements are arrays (e.g., time series, spectrograms).
The output format depends on the aggregation strategy chosen in the `TreatmentGroup`:

- **`aggregate`**: Each multidimensional element is flattened into multiple scalar
  columns — one per (window, feature) combination. The resulting `data` is a
  **tabular matrix** (not multidimensional), ready to be used alongside
  [`DiscreteDataset`](@ref) and [`ContinuousDataset`](@ref) in standard ML
  pipelines. Each column is described by an [`AggregateFeat`](@ref) entry.

- **`reducesize`**: Each element preserves its original dimensionality but is
  reduced in size (e.g., downsampling from 10 000 points to 256). The resulting
  `data` remains a matrix of arrays. Each column is described by a
  [`ReduceFeat`](@ref) entry.

Every column carries an `id` vector that traces its provenance back to the
original dataset column and `TreatmentGroup`, which is useful for groupby
operations or auditing the source of each derived feature.

# Type Parameter
- `T`: The element type of the inner arrays (e.g., `Float64`), or the scalar
  type when aggregation flattens the data.

# Fields
- `data::AbstractArray`: The processed matrix. When using `aggregate`, this is a
  scalar tabular matrix with one column per (feature × window × original column)
  combination. When using `reducesize`, this is a matrix of reduced-size arrays.
- `info::Vector{<:Union{AggregateFeat,ReduceFeat}}`: Per-column metadata. Contains
  [`AggregateFeat`](@ref) entries when using `aggregate`, or [`ReduceFeat`](@ref)
  entries when using `reducesize`. Each entry stores the original column name,
  source dimensionality (`dims`), validity/missing/NaN indices, internal corruption
  indices (`hasmissing`, `hasnans`), and the applied feature function or reduction
  function.
- `groups::Union{Nothing,Vector{Vector{Int}}}`: Optional column groupings for
  grouped operations. Each inner vector contains column indices belonging to the
  same group. `nothing` when no grouping is applied.

# Constructors

- `MultidimDataset(
        ids::Vector{Int},
        data::AbstractMatrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        aggrfunc::Base.Callable,
        float_type::Type,
        groups::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}
    )`  
  Internal constructor: selects columns `ids` from `data`, applies `aggrfunc` 
  to transform the multidimensional elements,
  and builds the corresponding metadata from `datastruct`.

## Arguments for internal constructor
- `ids::Vector{Int}`: Column indices to include in this dataset.
- `data::AbstractMatrix`: The full raw dataset matrix.
- `vnames::Vector{String}`: Names of all columns.
- `datastruct::NamedTuple`: Pre-computed dataset metadata.
- `aggrfunc::Base.Callable`: The aggregation or reduction strategy.
- `float_type::Type`: Target floating-point type for numeric conversion.
- `groups::Union{Nothing,Symbol,Tuple{Vararg{Symbol}}}`: Grouping specification.
  `nothing` disables grouping; a `Symbol` or tuple of `Symbol`s enables it.

See also: [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`AggregateFeat`](@ref), [`ReduceFeat`](@ref), [`aggregate`](@ref), [`reducesize`](@ref)
"""
mutable struct MultidimDataset{T,S} <: AbstractDataset
    data::AbstractArray
    info::Vector{<:Union{AggregateFeat,ReduceFeat}}
    groups::Union{Nothing,Vector{Vector{Int}}}

    MultidimDataset(
        data::AbstractArray,
        info::Vector{<:AggregateFeat{T}},
        groups::Union{Nothing,Vector{Vector{Int}}}
    ) where T = new{T,AggregateFeat}(data, info, groups)

    MultidimDataset(
        data::AbstractArray,
        info::Vector{<:ReduceFeat{T}},
        groups::Union{Nothing,Vector{Vector{Int}}}=nothing
    ) where T = new{T,ReduceFeat}(data, info, groups)

    function MultidimDataset(
        ids::Vector{Int},
        data::Matrix,
        vnames::Vector{String},
        datastruct::NamedTuple,
        aggrfunc::F,
        impute::Union{Nothing,Tuple{Vararg{<:Impute.Imputor}}},
        float_type::Type{T},
        groups::Union{Nothing, Tuple{Vararg{Symbol}}}
    ) where {T<:Float, F<:Base.Callable}
        data = @view data[:, ids]
        vnames = vnames[ids]
        dims = datastruct.dims[ids]
        valid = datastruct.valididxs[ids]
        miss = datastruct.missingidxs[ids]
        nan = datastruct.nanidxs[ids]
        hasmiss = datastruct.hasmissing[ids]
        hasnan = datastruct.hasnans[ids]

        md, nwindows = aggrfunc(data, valid, float_type)

        md_feats = if hasfield(typeof(aggrfunc), :features)
            tuples = Iterators.flatten((
                ((c, f, n) for f in _get_features(aggrfunc) for n in 1:nwindows[c])
                for c in eachindex(ids)
            ))

            [AggregateFeat{float_type}(
                ids[c],
                j,
                vnames[c],
                dims[c],
                f,
                n,
                valid[c],
                miss[c],
                nan[c],
                hasmiss[c],
                hasnan[c]
            ) for (j, (c, f, n)) in enumerate(tuples)]
        else
            [ReduceFeat{AbstractArray{float_type}}(
                ids[i],
                vnames[c],
                dims[c],
                _get_reducefunc(aggrfunc),
                valid[c],
                miss[c],
                nan[c],
                hasmiss[c],hasnan[c])
                for (i, c) in enumerate(axes(md,2))]
        end

        grouped = isnothing(groups) ? nothing : _groupby(md_feats, groups)

        new{float_type,eltype(md_feats)}(md, md_feats, grouped)
    end
end

# ---------------------------------------------------------------------------- #
#                                  methods                                     #
# ---------------------------------------------------------------------------- #
Base.eltype(::DiscreteDataset{T}) where T = T
Base.eltype(::ContinuousDataset{T}) where T = T
Base.eltype(::MultidimDataset{T})where T = T

Base.getindex(ds::MultidimDataset, idxs::AbstractVector{Int}) =
    MultidimDataset(
        @view(ds.data[:, idxs]),
        ds.info[idxs],
        _reindex_groups(ds.groups, idxs)
    )

Base.isempty(ds::AbstractDataset) = isempty(ds.data)

get_dims(d::MultidimDataset) = [get_dims(f) for f in d.info]

get_data(d::Vector{<:AbstractDataset}) = reduce(hcat, get_data.(d))
get_data(d::DiscreteDataset)= d.data
get_data(d::ContinuousDataset)= d.data
get_data(d::MultidimDataset)= d.data

get_info(d::Vector{<:AbstractDataset}) = reduce(vcat, get_info.(d))
get_info(d::AbstractDataset) = d.info

is_tabular(d::AbstractDataset) = d isa Union{
    ContinuousDataset,
    DiscreteDataset,
    MultidimDataset{<:Any, AggregateFeat}
}
is_multidim(d::AbstractDataset) = isa(d, MultidimDataset{<:Any, ReduceFeat})

get_vnames(d::AbstractDataset)::Vector{String} = get_vnames.(d.info)

function get_vnames(
    ds::MultidimDataset{<:Any,<:AggregateFeat};
    groupby_split::Bool=false
)
    names =
        ["$(f.vname),$(f.feat),win:$(f.nwin)" for f in ds.info]
    # groupby_split && has_groups(ds) ?
    #     [names[g] for g in get_groups(ds)] :
    #     names
end

