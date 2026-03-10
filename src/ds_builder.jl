# # ---------------------------------------------------------------------------- #
# #                           dataset structure utils                            #
# # ---------------------------------------------------------------------------- #
# """
#     check_column_structure(col::AbstractArray) -> (valtype, idx, hasmissing, hasnan)

# Analyze the structure of a single column to extract preliminary information.

# DataTreatments is designed to import heterogeneous datasets composed of columns 
# with discrete values, continuous values, and multivariate values. Additionally, 
# the presence of possible `NaN` or `missing` values is expected. There are no 
# starting assumptions: we know nothing about how the dataset is structured.

# This internal function is used by DataTreatments to scan the column and output 
# preliminary information.

# # Arguments
# - `col::AbstractArray`: a single column to analyze

# # Returns
# - `valtype::Type`: the predominant type of values in the column
# - `idx::Vector{Int}`: indices of valid (non-missing, non-NaN) elements
# - `hasmissing::Bool`: whether the column contains `missing` values
# - `hasnan::Bool`: whether the column contains `NaN` values
# """
# function check_column_structure(col::AbstractArray)
#     valtype, idx, hasmissing, hasnan = nothing, Int[], false, false

#     for i in eachindex(col)
#         val = col[i]
#         if ismissing(val)
#             hasmissing = true
#         elseif val isa AbstractFloat && isnan(val)
#             hasnan = true
#         elseif val isa AbstractVector{<:AbstractFloat} || val isa AbstractArray{<:AbstractFloat}
#             if any(isnan, val)
#                 hasnan = true
#             end
#             valtype = typeof(val)
#             push!(idx, i)
#         elseif !(val isa AbstractFloat) || !isnan(val)
#             if isnothing(valtype) || !(valtype <: AbstractVector)
#                 valtype = typeof(val)
#                 push!(idx, i)
#             end
#         end
#     end

#     return valtype, idx, hasmissing, hasnan
# end

# """
#     check_dataset_structure(X::Matrix) -> (valtype, idx, hasmissing, hasnan)

# Analyze the structure of an entire dataset to extract preliminary information.

# DataTreatments is designed to import heterogeneous datasets composed of columns 
# with discrete values, continuous values, and multivariate values. Additionally, 
# the presence of possible `NaN` or `missing` values is expected. There are no 
# starting assumptions: we know nothing about how the dataset is structured.

# This internal function is used by DataTreatments to scan the dataset and output 
# preliminary information for each column, processed in parallel.

# # Arguments
# - `X::Matrix`: the dataset matrix to analyze

# # Returns
# - `valtype::Vector{Type}`: vector where `valtype[i]` is the predominant type of 
#   column `i`, distinguishing between discrete, continuous, and multivariate columns
# - `idx::Vector{Vector{Int}}`: vector of vectors where `idx[i]` contains the indices 
#   of valid (non-missing, non-NaN) elements in column `i`
# - `hasmissing::Vector{Bool}`: boolean vector indicating whether column `i` contains 
#   `missing` values (useful for future developments)
# - `hasnan::Vector{Bool}`: boolean vector indicating whether column `i` contains 
#   `NaN` values (useful for future developments)
# """
# function check_dataset_structure(X::Matrix)
#     dim = size(X, 2)
#     valtype = Vector{Type}(undef, dim)
#     idx = Vector{Vector{Int}}(undef, dim)
#     hasmissing = Vector{Bool}(undef, dim)
#     hasnan = Vector{Bool}(undef, dim)

#     Threads.@threads for i in axes(X, 2)
#         valtype[i], idx[i], hasmissing[i], hasnan[i] = check_column_structure(@view(X[:, i]))
#     end
#     return valtype, idx, hasmissing, hasnan
# end

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
    valtype, idx, hasmissing, hasnan = check_dataset_structure(X)

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
        X = @view X[:, md_cols]
        vnames_md = @views vnames[md_cols]
        idx_md = @views idx[md_cols]
        miss, nan = hasmissing[md_cols], hasnan[md_cols]
        win isa Base.Callable && (win = (win,))

        if aggrtype == :aggregate
            Xmd, nwindows = DataTreatments.aggregate(X, idx_md, float_type; win, features)
            md_feats = vec([AggregateFeat{float_type}(i, vnames_md[c], f, nwindows[c], miss[c], nan[c])
                    for (i, (f, c)) in enumerate(Iterators.product(features, axes(X,2)))])

        elseif aggrtype == :reducesize
            Xmd = DataTreatments.reducesize(X, idx_md, float_type; win, reducefunc)
            md_feats = [ReduceFeat{AbstractArray{float_type}}(i, vnames_md[c], reducefunc, miss[c], nan[c])
                for (i, c) in enumerate(axes(X,2))]

        else
            error("Unknown treatment type: $treat")
        end
    end

    return Xtd, Xtc, Xmd, td_feats, tc_feats, md_feats
end

#
