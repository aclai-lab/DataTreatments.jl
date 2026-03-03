# ---------------------------------------------------------------------------- #
#                           nan/missing handle utils                           #
# ---------------------------------------------------------------------------- #
"""
    base_eltype(col::AbstractVector) -> (valtype, hasmissing, hasnan)

Inspect a column vector and infer its base element type along with the presence
of `missing` and `NaN` values.

# Arguments
- `col::AbstractVector`: a column vector of any element type.

# Returns
- `valtype::Union{Type, Nothing}`: the inferred base type of the non-missing elements.
  Returns `Float64` for scalar floats, the concrete type for array-valued elements,
  or the concrete type for any other value. Returns `nothing` if the column is empty
  or contains only `missing` values.
- `hasmissing::Bool`: `true` if any element is `missing`.
- `hasnan::Bool`: `true` if any element is `NaN` (scalar or inside a vector).
"""
# function base_eltype(col::AbstractVector)
#     valtype, hasmissing, hasnan = nothing, false, false
#     for val in col
#         if ismissing(val)
#             hasmissing = true
#         elseif val isa AbstractFloat
#             isnan(val) && (hasnan = true)
#             isnothing(valtype) && (valtype = Float64)
#         elseif val isa AbstractVector{<:AbstractFloat}
#             if any(isnan, val)
#                 hasnan = true
#             end
#             isnothing(valtype) && (valtype = typeof(val))
#         else
#             isnothing(valtype) && (valtype = typeof(val))
#         end
#         !isnothing(valtype) && hasmissing && hasnan && break
#     end
#     return valtype, hasmissing, hasnan
# end
function base_eltype(col::AbstractVector)
    valtype, hasmissing, hasnan = nothing, false, false
    for val in col
        if ismissing(val)
            hasmissing = true
        elseif val isa AbstractFloat
            isnan(val) && (hasnan = true)
            isnothing(valtype) && (valtype = Float64)
        elseif val isa AbstractVector{<:AbstractFloat}
            if any(isnan, val)
                hasnan = true
            end
            # Override scalar Float64 with actual array type
            valtype = typeof(val)
        else
            isnothing(valtype) && (valtype = typeof(val))
        end
        !isnothing(valtype) && hasmissing && hasnan && break
    end
    return valtype, hasmissing, hasnan
end

"""
    check_integrity(X::Matrix) -> (valtype, hasmissing, hasnan)

Check the integrity of each column of a matrix, inferring element types and
detecting `missing` and `NaN` values.

# Arguments
- `X::Matrix`: input matrix of any element type.

# Returns
- `valtype::Vector{Type}`: inferred base type for each column (see [`base_eltype`](@ref)).
- `hasmissing::Vector{Bool}`: `true` for each column that contains `missing` values.
- `hasnan::Vector{Bool}`: `true` for each column that contains `NaN` values.
"""
function check_integrity(X::Matrix)
    valtype = Vector{Type}(undef, size(X, 2))
    hasmissing = Vector{Bool}(undef, size(X, 2))
    hasnan = Vector{Bool}(undef, size(X, 2))

    Threads.@threads for i in axes(X, 2)
        valtype[i], hasmissing[i], hasnan[i] = base_eltype(@view(X[:, i]))
    end
    return valtype, hasmissing, hasnan
end

# ---------------------------------------------------------------------------- #
#                               discrete utils                                 #
# ---------------------------------------------------------------------------- #
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
#                               dataset builder                                #
# ---------------------------------------------------------------------------- #
function build_datasets(
    X::Matrix;
    aggrtype::Symbol=:aggregate,
    vnames::Union{Vector{Symbol},Nothing}=[Symbol("V$i") for i in 1:size(X, 2)],
    win::Union{Base.Callable,Tuple{Vararg{Base.Callable}}}=wholewindow(),
    features::Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
    reducefunc::Base.Callable=mean,
    float_type::Type=Float64,
    kwargs...
)
    Xtd = Xtc = Xmd = td_feats = tc_feats = md_feats = nothing
    valtype, hasmissing, hasnan = check_integrity(X)

    td_cols = findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    # discrete
    if !isempty(td_cols)
        vnames_td = @views vnames[td_cols]
        miss_td = hasmissing[td_cols]
        codes, levels = discrete_encode(X[:, td_cols])

        Xtd = stack(codes)
        td_feats = [DiscreteFeat(i, vnames_td[i], levels[i], miss_td[i]) for i in eachindex(vnames_td)]
    end

    # scalar
    if !isempty(tc_cols)
        vnames_tc = @views vnames[tc_cols]
        miss_tc, nan_tc = hasmissing[tc_cols], hasnan[tc_cols]

        Xtc = reduce(hcat, [map(x -> ismissing(x) ? missing : float_type(x), @view X[:, col]) for col in tc_cols])
        tc_feats = [ScalarFeat{float_type}(i, vnames_tc[i], miss_tc[i], nan_tc[i]) for i in eachindex(vnames_tc)]
    end

    # multivariate
    if !isempty(md_cols)
        vnames_md = @views vnames[md_cols]
        miss_md, nan_md = hasmissing[md_cols], hasnan[md_cols]

        _X = @view X[:, md_cols]
        uniform = has_uniform_element_size(_X)
        win isa Base.Callable && (win = (win,))
        intervals = @evalwindow first(_X) win...
        nwindows = prod(length.(intervals))

        if aggrtype == :aggregate
            Xmd = DataTreatments.aggregate(_X, intervals; features, win, uniform, float_type)

            md_feats = if nwindows == 1
                # single window: apply to whole time series
                vec([AggregateFeat{float_type}(i, vnames_md[c], f, 1, miss_md[c], nan_md[c])
                    for (i, (f, c)) in enumerate(Iterators.product(features, axes(_X,2)))])
            else
                # multiple windows: apply to each interval
                vec([AggregateFeat{float_type}(i, vnames_md[c], f, n, miss_md[c], nan_md[c])
                    for (i, (n, f, c)) in enumerate(Iterators.product(1:nwindows, features, axes(_X,2)))])
            end

        elseif aggrtype == :reducesize
            Xmd = DataTreatments.reducesize(_X, intervals; reducefunc, win, uniform, float_type)

            md_feats = [ReduceFeat{AbstractArray{float_type}}(i, vnames_md[c], reducefunc, miss_md[c], nan_md[c])
                for (i, c) in enumerate(axes(_X,2))]

        else
            error("Unknown treatment type: $treat")
        end
    end

    return Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats
end
