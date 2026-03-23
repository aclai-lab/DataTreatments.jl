# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
_isnanval(v) = v isa AbstractFloat && isnan(v)
_isarray(v) = v isa AbstractArray

# ---------------------------------------------------------------------------- #
#                              dataset inspecting                              #
# ---------------------------------------------------------------------------- #
"""
    _inspecting(data::Matrix) -> NamedTuple

Inspects a dataset provided as a column-wise matrix and extracts metadata for each column.

# Output Fields

For each column:

- `id::Vector{Int}`: Unique ID.
- `datatype::Vector{Type}`: Data type.
- `dims::Vector{Int}`: Dimensionality of array elements, scalar values will have `dims = 0`.
- `valididxs::Vector{Vector{Int}}`: Indices of valid (non-`missing`, non-`NaN`) values for each column.
- `missingidxs::Vector{Vector{Int}}`: Indices of `missing` values.
- `nanidxs::Vector{Vector{Int}}`: Indices of `NaN` values.
- `hasmissing::Vector{Vector{Int}}`: Indices of array elements that internally contain `missing` values.
- `hasnans::Vector{Vector{Int}}`: Indices of array elements that internally contain `NaN` values.
"""
function _inspecting(data::Matrix)
        ncols = size(data, 2)
        id = collect(1:ncols)

        datatype = Vector{Type}(undef, ncols)
        dims = Vector{Int}(undef, ncols)
        valididxs = Vector{Vector{Int}}(undef, ncols)
        missingidxs = Vector{Vector{Int}}(undef, ncols)
        nanidxs = Vector{Vector{Int}}(undef, ncols)
        hasmissing = Vector{Vector{Int}}(undef, ncols)
        hasnans = Vector{Vector{Int}}(undef, ncols)

        Threads.@threads for i in 1:ncols
            col = @view(data[:, i])

            _valid = Int[]
            _missing = Int[]
            _nan = Int[]
            _hasmissing = Int[]
            _hasnans = Int[]

            dt = Any
            maxdims = 0

            @inbounds for j in eachindex(col)
                v = col[j]
                if ismissing(v)
                    push!(_missing, j)
                elseif _isnanval(v)
                    push!(_nan, j)
                else
                    push!(_valid, j)
                    dt = dt === Any ? typeof(v) : typejoin(dt, typeof(v))
                    if _isarray(v)
                        d = ndims(v)
                        d > maxdims && (maxdims = d)
                        any(ismissing, v) && push!(_hasmissing, j)
                        any(_isnanval, v) && push!(_hasnans, j)
                    end
                end
            end

            datatype[i] = dt
            dims[i] = maxdims
            valididxs[i] = _valid
            missingidxs[i] = _missing
            nanidxs[i] = _nan
            hasmissing[i] = _hasmissing
            hasnans[i] = _hasnans
        end

        return (
            id=id,
            datatype=datatype,
            dims=dims,
            valididxs=valididxs,
            missingidxs=missingidxs,
            nanidxs=nanidxs,
            hasmissing=hasmissing,
            hasnans=hasnans
        )
end