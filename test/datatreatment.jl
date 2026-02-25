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

a = DataTreatment(Xc, norm=MinMax)
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
a = NamedTuple{Tuple(propertynames(dfmt))}(Tuple(eachcol(dfmt)))
b = Tuple(eachcol(dfmt))

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


X = DataFrame([Symbol("col_$i") => rand(1000) for i in 1:2000]...)

@btime NamedTuple{Tuple(propertynames(X))}(Tuple(eachcol(X)));
# 202.404 μs (15 allocations: 110.09 KiB)

@btime Tuple(eachcol(X));
# 60.914 μs (6 allocations: 47.15 KiB)

@btime Matrix(X);
# 3.286 ms (2005 allocations: 15.30 MiB)

@btime ComponentArray(; (name => col for (name, col) in pairs(eachcol(X)))...);
# 16.115 μs (70 allocations: 28.56 KiB)

@btime(DataTreatment(X, norm=MinMax));
# 191.981 ms (4151584 allocations: 204.24 MiB)
# 125.769 ms (4238135 allocations: 148.51 MiB)

@btime(DataTreatment(X))

# a.vec_col
# a[:vec_col]

# # by index
# a[1]

# # iterate
# for (k, v) in pairs(a)
#     println(k, " => ", eltype(v))
# end

# # destructure
# vec_col, str_col, int_col, float_col = a

test = NamedTuple{Tuple(propertynames(X))}(Tuple(eachcol(X)));

@btime begin
    for t in test
    end
end
# 78.858 μs (3490 allocations: 85.78 KiB)

@btime begin
    for (name, value) in pairs(test)
    end
end
# 8.187 ms (9491 allocations: 288.97 KiB)

groupidxs_t = Tuple(g for g in groupidxs)

@generated function iterate_nt(t, ::Val{groupidxs}, ::Val{norm}) where {groupidxs, norm}
    quote
        $(Expr(:block, [:(normalize(reduce(hcat, t[collect($g)]), $norm)) for g in groupidxs]...))
    end
end

@btime begin
    iterate_nt(test)
end
# 103.784 ns (2 allocations: 64 bytes)
