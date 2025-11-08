# this code is for development use only

using BenchmarkTools
using DataTreatments
using SoleXplorer
using DataFrames

A = rand(10000)

# ---------------------------------------------------------------------------- #
@show Threads.nthreads()
# Threads.nthreads() = 1

# DataTreatments
wfunc = DataTreatments.movingwindow(winsize=10, winstep=5)
@btime DataTreatments.@evalwindow A wfunc
# 5.464 μs (13 allocations: 47.22 KiB)

wfunc = DataTreatments.wholewindow()
@btime DataTreatments.@evalwindow A wfunc
# 438.090 ns (10 allocations: 288 bytes)

wfunc = DataTreatments.splitwindow(nwindows=5)
@btime DataTreatments.@evalwindow A wfunc
# 549.941 ns (12 allocations: 528 bytes)

wfunc = DataTreatments.adaptivewindow(nwindows=5, overlap=0.2)
@btime DataTreatments.@evalwindow A wfunc
# 557.296 ns (12 allocations: 544 bytes)

# SoleXplorer
wfunc = SoleXplorer.MovingWindow(window_size=10, window_step=5)
@btime wfunc(length(A))
# 4.105 μs (5 allocations: 31.35 KiB)

wfunc = SoleXplorer.WholeWindow()
@btime wfunc(length(A))
# 914.432 ns (16 allocations: 464 bytes)

wfunc = SoleXplorer.SplitWindow(nwindows=5)
@btime wfunc(length(A))
# 2.824 μs (60 allocations: 1.72 KiB)

wfunc = SoleXplorer.AdaptiveWindow(nwindows=5, relative_overlap=0.2)
@btime wfunc(length(A))
#  2.863 μs (60 allocations: 1.72 KiB)

# ---------------------------------------------------------------------------- #
@show Threads.nthreads()
# Threads.nthreads() = 8

# DataTreatments
wfunc = DataTreatments.movingwindow(winsize=10, winstep=5)
@btime DataTreatments.@evalwindow A wfunc
# 5.224 μs (13 allocations: 47.22 KiB)

wfunc = DataTreatments.wholewindow()
@btime DataTreatments.@evalwindow A wfunc
# 434.864 ns (10 allocations: 288 bytes)

wfunc = DataTreatments.splitwindow(nwindows=5)
@btime DataTreatments.@evalwindow A wfunc
# 531.969 ns (12 allocations: 528 bytes)

wfunc = DataTreatments.adaptivewindow(nwindows=5, overlap=0.2)
@btime DataTreatments.@evalwindow A wfunc
# 521.203 ns (12 allocations: 544 bytes)

# SoleXplorer
wfunc = SoleXplorer.MovingWindow(window_size=10, window_step=5)
@btime wfunc(length(A))
# 3.975 μs (5 allocations: 31.35 KiB)

wfunc = SoleXplorer.WholeWindow()
@btime wfunc(length(A))
# 913.968 ns (16 allocations: 464 bytes)

wfunc = SoleXplorer.SplitWindow(nwindows=5)
@btime wfunc(length(A))
# 2.789 μs (60 allocations: 1.72 KiB)

wfunc = SoleXplorer.AdaptiveWindow(nwindows=5, relative_overlap=0.2)
@btime wfunc(length(A))
#  2.749 μs (60 allocations: 1.72 KiB)

# ---------------------------------------------------------------------------- #
Xmatrix = fill(rand(2000), 1000, 100)
Xdf = DataFrame(Xmatrix, :auto)
features = (mean, minimum, maximum)

# ---------------------------------------------------------------------------- #
@show Threads.nthreads()
# Threads.nthreads() = 1

# DataTreatments
wfunc = DataTreatments.adaptivewindow(nwindows=5, overlap=0.2)
@btime DataTreatments.DataTreatment(Xmatrix, :aggregate; vnames=Symbol.("var", 1:100),win=(wfunc,), features)
# 185.137 ms (11228 allocations: 11.93 MiB)

wfunc = DataTreatments.adaptivewindow(nwindows=5, overlap=0.2)
@btime DataTreatments.DataTreatment(Xmatrix, :reducesize; vnames=Symbol.("var", 1:100),win=(wfunc,))
# 19.397 ms (200831 allocations: 9.95 MiB)

# SoleXplorer
wfunc = SoleXplorer.AdaptiveWindow(nwindows=5, relative_overlap=0.2)
@btime SoleXplorer.treatment(Xdf, :aggregate; win=wfunc, features)
# 191.738 ms (21411 allocations: 12.32 MiB)

wfunc = SoleXplorer.AdaptiveWindow(nwindows=5, relative_overlap=0.2)
@btime SoleXplorer.treatment(Xdf, :reducesize; win=wfunc, features)
# 19.173 ms (200488 allocations: 9.94 MiB)

# ---------------------------------------------------------------------------- #
@show Threads.nthreads()
# Threads.nthreads() = 8

# DataTreatments
wfunc = DataTreatments.adaptivewindow(nwindows=5, overlap=0.2)
@btime DataTreatments.DataTreatment(Xmatrix, :aggregate; vnames=Symbol.("var", 1:100),win=(wfunc,), features)
# 48.314 ms (11263 allocations: 11.93 MiB)

wfunc = DataTreatments.adaptivewindow(nwindows=5, overlap=0.2)
@btime DataTreatments.DataTreatment(Xmatrix, :reducesize; vnames=Symbol.("var", 1:100),win=(wfunc,))
# 4.511 ms (200882 allocations: 9.95 MiB)

# SoleXplorer
wfunc = SoleXplorer.AdaptiveWindow(nwindows=5, relative_overlap=0.2)
@btime SoleXplorer.treatment(Xdf, :aggregate; win=wfunc, features)
# 210.443 ms (2321411 allocations: 62.67 MiB)

wfunc = SoleXplorer.AdaptiveWindow(nwindows=5, relative_overlap=0.2)
@btime SoleXplorer.treatment(Xdf, :reducesize; win=wfunc, features)
# 18.763 ms (200488 allocations: 9.94 MiB)
