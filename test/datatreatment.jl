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

Tables.schema(X)
col, st = iterate(col3)

names = Tables.columnnames(col)



