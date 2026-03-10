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
"""
    DiscreteDataset <: AbstractDataset

Output dataset for **discrete (categorical)** columns produced by `DataTreatment`.

Each column of the original dataset that belongs to a discrete `TreatmentGroup`
is encoded as an integer-coded categorical variable via [`discrete_encode`](@ref).
The resulting `dataset` matrix contains integer level codes (with `missing`
preserved), and each column is described by a [`DiscreteFeat`](@ref) metadata
entry in `info`.

# Fields
- `dataset::Matrix`: encoded integer-coded matrix (one column per discrete feature).
  Each value is either an `Int` level code or `missing`.
- `info::Vector{DiscreteFeat}`: per-column metadata, including the original column
  name, categorical levels, validity indices, and an `id` vector tracing the
  column back to the source dataset and `TreatmentGroup`.

# Constructors

    DiscreteDataset(dataset::Matrix, info::Vector{DiscreteFeat})

Direct constructor from a pre-built matrix and metadata vector.

    DiscreteDataset(id::Vector, dataset::Matrix, ds_struct::DatasetStructure, cols::Vector{Int})

Lazy constructor called internally by `DataTreatment`. Selects columns `cols`
from `dataset`, encodes them categorically, and builds the corresponding
[`DiscreteFeat`](@ref) metadata from `ds_struct`.

## Arguments
- `id::Vector`: base identifier vector; each feature appends its local index to
  trace provenance back to the source `TreatmentGroup`.
- `dataset::Matrix`: the full raw dataset matrix.
- `ds_struct::DatasetStructure`: pre-computed dataset metadata.
- `cols::Vector{Int}`: column indices to include in this dataset.

See also: [`ContinuousDataset`](@ref), [`MultidimDataset`](@ref), [`DiscreteFeat`](@ref)
"""
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

"""
    ContinuousDataset{T} <: AbstractDataset

Output dataset for **continuous (numeric scalar)** columns produced by `DataTreatment`.

Each selected column is cast to the target `float_type` (e.g., `Float64`), with
`missing` values preserved. The resulting `dataset` matrix is a numeric matrix
ready for downstream ML pipelines, and each column is described by a
[`ContinuousFeat`](@ref) metadata entry in `info`.

# Type Parameter
- `T`: the floating-point type used for numeric conversion (e.g., `Float64`, `Float32`).

# Fields
- `dataset::Matrix`: numeric matrix (one column per continuous feature), with
  elements of type `Union{Missing, T}`.
- `info::Vector{ContinuousFeat}`: per-column metadata, including the original
  column name, validity indices, missing indices, NaN indices, and an `id` vector
  tracing the column back to the source dataset and `TreatmentGroup`.

# Constructors

    ContinuousDataset(dataset::Matrix, info::Vector{ContinuousFeat{T}})

Direct constructor from a pre-built matrix and metadata vector.

    ContinuousDataset(id::Vector, dataset::Matrix, ds_struct::DatasetStructure, cols::Vector{Int}, float_type::Type)

Lazy constructor called internally by `DataTreatment`. Selects columns `cols`
from `dataset`, converts each element to `float_type` (preserving `missing`),
and builds the corresponding [`ContinuousFeat`](@ref) metadata from `ds_struct`.

## Arguments
- `id::Vector`: base identifier vector; each feature appends its local index.
- `dataset::Matrix`: the full raw dataset matrix.
- `ds_struct::DatasetStructure`: pre-computed dataset metadata.
- `cols::Vector{Int}`: column indices to include in this dataset.
- `float_type::Type`: target floating-point type for numeric conversion.

See also: [`DiscreteDataset`](@ref), [`MultidimDataset`](@ref), [`ContinuousFeat`](@ref)
"""
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

"""
    MultidimDataset{T} <: AbstractDataset

Output dataset for **multidimensional** columns produced by `DataTreatment`.

Handles columns whose elements are arrays (e.g., time series, spectrograms).
The output format depends on the aggregation strategy chosen in the `TreatmentGroup`:

- **`aggregate`**: each multidimensional element is flattened into multiple scalar
  columns — one per (window, feature) combination. The resulting `dataset` is a
  **tabular matrix** (not multidimensional), ready to be used alongside
  [`DiscreteDataset`](@ref) and [`ContinuousDataset`](@ref) in standard ML
  pipelines. Each column is described by an [`AggregateFeat`](@ref) entry.

- **`reducesize`**: each element preserves its original dimensionality but is
  reduced in size (e.g., downsampling from 10 000 points to 256). The resulting
  `dataset` remains a matrix of arrays. Each column is described by a
  [`ReduceFeat`](@ref) entry.

Every column carries an `id` vector that traces its provenance back to the
original dataset column and `TreatmentGroup`, which is useful for groupby
operations or auditing the source of each derived feature.

# Type Parameter
- `T`: the element type of the inner arrays (e.g., `Float64`), or the scalar
  type when aggregation flattens the data.

# Fields
- `dataset::Matrix`: the processed matrix. When using `aggregate`, this is a
  scalar tabular matrix with one column per (feature × window × original column)
  combination. When using `reducesize`, this is a matrix of reduced-size arrays.
- `info::Vector{Union{AggregateFeat,ReduceFeat}}`: per-column metadata. Contains
  [`AggregateFeat`](@ref) entries when using `aggregate`, or [`ReduceFeat`](@ref)
  entries when using `reducesize`. Each entry stores the original column name,
  validity/missing/NaN indices, internal corruption indices (`hasmissing`,
  `hasnans`), and the applied feature function or reduction function.

# Constructors

    MultidimDataset(dataset::Matrix, info::Vector{AggregateFeat{T}})
    MultidimDataset(dataset::Matrix, info::Vector{ReduceFeat{T}})

Direct constructors from a pre-built matrix and metadata vector.

    MultidimDataset(id::Vector, dataset::Matrix, ds_struct::DatasetStructure, cols::Vector{Int}, aggrfunc::Base.Callable, float_type::Type)

Lazy constructor called internally by `DataTreatment`. Selects columns `cols`
from `dataset`, applies `aggrfunc` to transform the multidimensional elements,
and builds the corresponding metadata from `ds_struct`.

The constructor inspects `aggrfunc` to decide the output format:
- If `aggrfunc` has a `features` field (i.e., it is an `aggregate` callable),
  the data is flattened and [`AggregateFeat`](@ref) metadata is produced.
- Otherwise (i.e., it is a `reducesize` callable), the data is reduced in place
  and [`ReduceFeat`](@ref) metadata is produced.

## Arguments
- `id::Vector`: base identifier vector; each feature appends its local index.
- `dataset::Matrix`: the full raw dataset matrix.
- `ds_struct::DatasetStructure`: pre-computed dataset metadata.
- `cols::Vector{Int}`: column indices to include in this dataset.
- `aggrfunc::Base.Callable`: the aggregation or reduction strategy.
- `float_type::Type`: target floating-point type for numeric conversion.

See also: [`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`AggregateFeat`](@ref), [`ReduceFeat`](@ref), [`aggregate`](@ref), [`reducesize`](@ref)
"""
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

# ---------------------------------------------------------------------------- #
#                               Base methods                                   #
# ---------------------------------------------------------------------------- #
Base.size(ds::AbstractDataset) = size(ds.dataset)
Base.size(ds::AbstractDataset, d::Int) = size(ds.dataset, d)

Base.length(ds::AbstractDataset) = length(ds.info)
Base.ndims(ds::AbstractDataset) = ndims(ds.dataset)
Base.eachindex(ds::AbstractDataset) = Base.OneTo(length(ds))
Base.iterate(ds::AbstractDataset, state=1) =
    state > length(ds) ? nothing : (ds.info[state], state + 1)

Base.getindex(ds::AbstractDataset, i::Int) = ds.info[i]
Base.getindex(ds::AbstractDataset, idxs::Vector{Int}) = ds.info[idxs]

Base.eltype(::DiscreteDataset) = DiscreteFeat
Base.eltype(::ContinuousDataset{T}) where T = ContinuousFeat{T}
Base.eltype(::MultidimDataset{T}) where T = Union{AggregateFeat{T},ReduceFeat{T}}

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_dataset(ds::AbstractDataset) -> Matrix

Returns the underlying dataset matrix.
"""
get_dataset(ds::AbstractDataset) = ds.dataset

"""
    get_dataset(ds::AbstractDataset, i::Int) -> Vector
    get_dataset(ds::AbstractDataset, idxs::Vector{Int}) -> Matrix

Returns column `i` or columns `idxs` of the underlying dataset matrix.
"""
get_dataset(ds::AbstractDataset, i::Int) = @view ds.dataset[:, i]
get_dataset(ds::AbstractDataset, idxs::Vector{Int}) = @view ds.dataset[:, idxs]

"""
    get_info(ds::AbstractDataset) -> Vector{<:AbstractDataFeature}

Returns the full vector of feature metadata entries.
"""
get_info(ds::AbstractDataset) = ds.info

"""
    get_info(ds::AbstractDataset, i::Int) -> AbstractDataFeature
    get_info(ds::AbstractDataset, idxs::Vector{Int}) -> Vector{<:AbstractDataFeature}

Returns the `i`-th feature metadata entry, or a subset by indices.
"""
get_info(ds::AbstractDataset, i::Int) = ds.info[i]
get_info(ds::AbstractDataset, idxs::Vector{Int}) = @views ds.info[idxs]

"""
    get_nrows(ds::AbstractDataset) -> Int

Returns the number of rows (observations) in the dataset.
"""
get_nrows(ds::AbstractDataset) = size(ds.dataset, 1)

"""
    get_ncols(ds::AbstractDataset) -> Int

Returns the number of columns (features) in the dataset.
"""
get_ncols(ds::AbstractDataset) = size(ds.dataset, 2)

"""
    get_vnames(ds::AbstractDataset) -> Vector{String}

Returns the variable names of all features in the dataset.
"""
get_vnames(ds::AbstractDataset) = [get_vname(f) for f in ds.info]

"""
    get_vnames(ds::AbstractDataset, i::Int) -> String
    get_vnames(ds::AbstractDataset, idxs::Vector{Int}) -> Vector{String}

Returns the variable name(s) for the specified feature index/indices.
"""
get_vnames(ds::AbstractDataset, i::Int) = get_vname(ds.info[i])
get_vnames(ds::AbstractDataset, idxs::Vector{Int}) = [get_vname(ds.info[i]) for i in idxs]

"""
    get_ids(ds::AbstractDataset) -> Vector{Vector}

Returns the id vectors for all features in the dataset.
"""
get_idxs(ds::AbstractDataset) = [get_id(f) for f in ds.info]

"""
    get_idxs(ds::AbstractDataset, i::Int) -> Vector
    get_idxs(ds::AbstractDataset, idxs::Vector{Int}) -> Vector{Vector}

Returns the id vector(s) for the specified feature index/indices.
"""
get_idxs(ds::AbstractDataset, i::Int) = get_id(ds.info[i])
get_idxs(ds::AbstractDataset, idxs::Vector{Int}) = [get_id(ds.info[i]) for i in idxs]

# ---------------------------------------------------------------------------- #
#                               show methods                                   #
# ---------------------------------------------------------------------------- #
# one-line
function Base.show(io::IO, ds::DiscreteDataset)
    nrows, ncols = size(ds)
    print(io, "DiscreteDataset($(nrows)×$(ncols))")
end

function Base.show(io::IO, ds::ContinuousDataset{T}) where T
    nrows, ncols = size(ds)
    print(io, "ContinuousDataset{$T}($(nrows)×$(ncols))")
end

function Base.show(io::IO, ds::MultidimDataset{T}) where T
    nrows, ncols = size(ds)
    mode = all(f -> f isa AggregateFeat, ds.info) ? "aggregate" : "reducesize"
    print(io, "MultidimDataset{$T}($(nrows)×$(ncols), $mode)")
end

# multi-line
function Base.show(io::IO, ::MIME"text/plain", ds::DiscreteDataset)
    nrows, ncols = size(ds)
    println(io, "DiscreteDataset($(nrows) rows × $(ncols) columns)")
    println(io, "├─ vnames: $(get_vnames(ds))")

    n_miss = count(f -> !isempty(get_missingidxs(f)), ds.info)
    if n_miss > 0
        println(io, "├─ columns with missing: $n_miss")
    end

    print(io, "└─ levels per column: $(join([string(length(get_levels(f))) for f in ds.info], ", "))")
end

function Base.show(io::IO, ::MIME"text/plain", ds::ContinuousDataset{T}) where T
    nrows, ncols = size(ds)
    println(io, "ContinuousDataset{$T}($(nrows) rows × $(ncols) columns)")
    println(io, "├─ vnames: $(get_vnames(ds))")

    n_miss = count(f -> !isempty(get_missingidxs(f)), ds.info)
    n_nan = count(f -> !isempty(get_nanidxs(f)), ds.info)

    if n_miss > 0
        println(io, "├─ columns with missing: $n_miss")
    end
    if n_nan > 0
        println(io, "├─ columns with NaN: $n_nan")
    end

    print(io, "└─ float type: $T")
end

function Base.show(io::IO, ::MIME"text/plain", ds::MultidimDataset{T}) where T
    nrows, ncols = size(ds)
    mode = all(f -> f isa AggregateFeat, ds.info) ? "aggregate" : "reducesize"

    println(io, "MultidimDataset{$T}($(nrows) rows × $(ncols) columns)")
    println(io, "├─ mode: $mode")
    println(io, "├─ vnames: $(get_vnames(ds))")

    n_miss = count(f -> !isempty(get_missingidxs(f)), ds.info)
    n_nan = count(f -> !isempty(get_nanidxs(f)), ds.info)
    n_hmiss = count(f -> !isempty(get_hasmissing(f)), ds.info)
    n_hnan = count(f -> !isempty(get_hasnans(f)), ds.info)

    if n_miss > 0
        println(io, "├─ columns with missing: $n_miss")
    end
    if n_nan > 0
        println(io, "├─ columns with NaN: $n_nan")
    end
    if n_hmiss > 0
        println(io, "├─ columns with internal missing: $n_hmiss")
    end
    if n_hnan > 0
        println(io, "├─ columns with internal NaN: $n_hnan")
    end

    if mode == "aggregate"
        unique_feats = unique([string(nameof(get_feat(f))) for f in ds.info])
        println(io, "├─ features: $(join(unique_feats, ", "))")
        unique_nwins = unique([get_nwin(f) for f in ds.info])
        print(io, "└─ windows: $(join(string.(unique_nwins), ", "))")
    else
        print(io, "└─ reduce function: $(nameof(get_reducefunc(ds.info[1])))")
    end
end