using DataFrames
using DataTreatments

# X = DataFrame(
#     vec_col  = [rand(Float64, 5), NaN, rand(Float64, 5), rand(Float64, 5), missing],
#     str_col  = ["hello", "world", missing, "foo", "bar"],
#     ve1_col  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5)],
#     int_col  = [1, NaN, 3, 4, missing],
#     float_col = [1.1, 2.2, missing, 4.4, NaN],
#     ve2_col  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5)]
# )

X = DataFrame(
    v1  = [rand(Float64, 5), NaN, rand(Float64, 5), rand(Float64, 5), missing],
    v2  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), missing, rand(Float64, 5)],
    v3  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5)],
    v4  = [rand(Float64, 5), rand(Float64, 5), NaN, rand(Float64, 5), rand(Float64, 5)],
)

test = DataTreatment(X, aggrtype=:aggregate)


allequal(eltype.(eachcol(X)))

@btime col1 = Tables.columntable(X)
# 3.681 μs (32 allocations: 1.36 KiB)
@btime col2 = collect(eachcol(X))
# 36.583 ns (3 allocations: 128 bytes)
@btime Tables.columns(X)
# 13.324 ns (1 allocation: 16 bytes)

col1 = Tables.columntable(X)

col2 = collect(eachcol(X))

col3 = Tables.columns(X)

col4 = eachcol(X)

sch = Tables.schema(X)
col, st = iterate(col3)

names = Tables.columnnames(col)

using DataTreatments
using MLJ, DataFrames
using SoleData: Artifacts

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

X = DataFrame(
    v1  = [rand(Float64, 5), NaN, rand(Float64, 5), rand(Float64, 5), missing],
    v2  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), missing, rand(Float64, 5)],
    v3  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5)],
    v4  = [rand(Float64, 5), rand(Float64, 5), NaN, rand(Float64, 5), rand(Float64, 5)],
)

DataTreatment(X)

filter(!=(Any), unique(typeof.(X[!, 1])))

X = DataFrame(
    v1 = [1.1, NaN, 3.3, missing],          # one NaN + one missing
    v2 = [1.1, 2.2, NaN, 4.4],              # one NaN
    v3 = [1.1, missing, 3.3, 4.4],          # one missing
    v4 = [1.1, 2.2, 3.3, 4.4],              # clean
    v5 = [1.1, 2.2, 3.3, 4.4],              # clean
)

DataTreatment(X)

function nansafe_col(col)
    return map(col) do val
        # NaN scalar → treat as missing
        if val isa AbstractFloat && isnan(val)
            return missing
        # Vector with NaN elements → replace with missing
        elseif val isa AbstractVector
            return any(isnan, val) ? missing : val
        else
            return val
        end
    end
end

function nansafe_df(X::AbstractDataFrame)
    return DataFrame(
        [name => nansafe_col(col) for (name, col) in pairs(eachcol(X))]
    )
end

function dropmissing_modal(X::AbstractDataFrame)
    # find rows where ANY column has missing
    valid = map(1:nrow(X)) do i
        all(col -> !ismissing(col[i]), eachcol(X))
    end
    return X[valid, :]
end

function prepare_df(X::AbstractDataFrame)
    X = nansafe_df(X)       # NaN → missing
    X = dropmissing(X)      # drop rows with missing
    return X
end

function base_eltype(col::AbstractVector)
    valtype, hasmissing, hasnan = nothing, false, false
    for val in col
        val isa AbstractVector ? begin
            all(ismissing.(val)) && (hasmissing = true; continue)
            all(isnan.(val)) && (hasnan = true; continue)
        end : begin
            ismissing(val) && (hasmissing = true; continue)
            isnan(val) && (hasnan = true; continue)
        end
        isnothing(valtype) && (valtype = typeof(val))
        !isnothing(valtype) && hasmissing && hasnan && break
    end
    return valtype, hasmissing, hasnan
end

function check_integrity(X::Matrix{T}) where T
    results = Vector{Tuple{Type,Bool,Bool}}(undef, size(X, 2))
    Threads.@threads for i in axes(X, 2)
        results[i] = base_eltype(@view(X[:, i]))
    end
    return results
end

# 242.642 μs (25979 allocations: 879.60 KiB)
# 95.924 μs (26034 allocations: 883.62 KiB)

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
            isnothing(valtype) && (valtype = typeof(val))
        else
            isnothing(valtype) && (valtype = typeof(val))
        end
        !isnothing(valtype) && hasmissing && hasnan && break
    end
    return valtype, hasmissing, hasnan
end

function check_integrity(X::Matrix{T}) where T
    results = Vector{Tuple{Type,Bool,Bool}}(undef, size(X, 2))
    Threads.@threads for i in axes(X, 2)
        results[i] = base_eltype(@view(X[:, i]))
    end
    return results
end