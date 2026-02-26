using DataFrames
using DataTreatments

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
has_missing(T::Type) = T >: Missing
has_missing(x) = has_missing(eltype(x))

# ---------------------------------------------------------------------------- #
#                                  MetaData                                    #
# # ---------------------------------------------------------------------------- #
# struct MetaData <: AbstractMetaData
#     group::Vector{Int64}
#     method::Vector{Symbol}
# end

# various types dataset loading

# 1 - tabular
using MLJ

Xc, yc = @load_iris
Xc = DataFrame(Xc)

a = DataTreatment(Xc)
fields = [[:sepal_length, :petal_length], [:sepal_width]]
DataTreatments._groupby(a.X, a.datafeature, fields)
b = BitVector([1, 0, 1, 0])
DataTreatments._groupby(a.X, a.datafeature, b)

DataTreatment(Xc, groupby=fields)
DataTreatment(Xc, groupby=b)

# 2 - multidim
using SoleData: Artifacts

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

# groupby test
a = DataTreatment(Xts, win=splitwindow(nwindows=3))
field = :vname
DataTreatments._groupby(a.X, a.datafeature, field)
fields = [:vname, :nwin]
DataTreatments._groupby(a.X, a.datafeature, fields)

DataTreatment(Xts, win=splitwindow(nwindows=3), groupby=field)
DataTreatment(Xts, win=splitwindow(nwindows=3), groupby=fields)

# 3 - tabular with one missing
Xm = DataFrame(
    a = [1.0, 2.0, NaN, 4.0, 5.0],
    b = [1.1, 2.2, 3.3, 4.4, 5.5],
    c = [1.2, 2.3, 3.4, 4.5, 5.6],
    d = [1.3, 2.4, 3.5, 4.6, 5.7],
    e = [1.4, 2.5, 3.6, missing, 5.8]
)

a = DataTreatment(Xm, norm=MinMax)

# 4 - multitype with missing
dfmtm = DataFrame(
    vec_col  = [rand(Float64, 5), missing, rand(Float64, 5), rand(Float64, 5), missing],
    str_col  = ["hello", "world", missing, "foo", "bar"],
    int_col  = [1, missing, 3, 4, missing],
    float_col = [1.1, 2.2, missing, 4.4, missing]
)

dfmt = DataFrame(
    vec_col  = [rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5), rand(Float64, 5)],
    str_col  = ["hello", "world", "paso", "foo", "bar"],
    int_col  = [1, 7, 3, 4, 6],
    float_col = [1.1, 2.2, 3.5, 4.4, 5.3]
)
a = DataTreatment(dfmt)

df_str = DataFrame(
    a = ["foo", "bar", "baz", "qux"],
    b = ["hello", "world", "paso", "test"],
    c = ["alpha", "beta", "gamma", "delta"],
    d = ["one", "two", "three", "four"],
    e = ["cat", "dog", "bird", "fish"]
)

function code_dataset(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            # handle mixed types by converting to string first
            eltype(col) == AbstractString || (col = string.(coalesce.(col, "missing")))
            X[!, name] = CategoricalArrays.levelcode.(categorical(col))
        end
    end
    
    return X
end




# DataTreatment(Xc)
# DataTreatment(Xc, yc; groups=[[:sepal_length, :petal_length], [:sepal_width]])

# DataTreatment(Xts)
# DataTreatment(Xts; aggrtype=:aggregate, groups=:feat)

# @test_throws ArgumentError DataTreatment(Xts; aggrtype=:aggregate, groups=:invalid)

# DataTreatment(Xts; aggrtype=:reducesize)
# DataTreatment(Xts; aggrtype=:aggregate, features=(std, median))
# DataTreatment(Xts; aggrtype=:reducesize, features=(std, median))
# DataTreatment(Xts; aggrtype=:aggregate, win=splitwindow(nwindows=3))
# DataTreatment(Xts; aggrtype=:reducesize, win=splitwindow(nwindows=3))
# DataTreatment(Xts, yts)

# DataTreatment(Matrix(Xc))
# DataTreatment(Matrix(Xc), yc)

# DataTreatment(Matrix(Xts))
# DataTreatment(Matrix(Xts), yts)



# missing ref:
# https://it.mathworks.com/help/matlab/ref/fillmissing.html
# https://it.mathworks.com/help/matlab/ref/fillmissing2.html
# https://it.mathworks.com/help/matlab/ref/rmmissing.html
# https://it.mathworks.com/help/matlab/ref/ismissing.html

using DataFrames
using DataTreatments

X = DataFrame([Symbol("col_$i") => rand(1000) for i in 1:2000]...)

@btime DataTreatment(X)
# 1.588 ms (74 allocations: 15.44 MiB)
@btime DataTreatment(X, norm=MinMax);
# 69.963 ms (4194469 allocations: 176.17 MiB)
# 62.531 ms (4133659 allocations: 173.23 MiB)
# 5.299 ms (115562 allocations: 50.23 MiB)
# 5.276 ms (114152 allocations: 50.19 MiB)
# 5.620 ms (98151 allocations: 19.84 MiB)
