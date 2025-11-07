using Test
using DataTreatments

using Statistics

X = rand(100)
wfunc = splitwindow(nwindows=10)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals)

X = rand(100, 120)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals; reducefunc=maximum)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = reducesize(Xmatrix, intervals; reducefunc=std)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
features = (mean, maximum)
result = aggregate(Xmatrix, intervals; features)


########################################################
using SoleData.Artifacts
# fill your Artifacts.toml file;
@test_nowarn Artifacts.fillartifacts()

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

win = DataTreatments.adaptivewindow(nwindows=6, overlap=0.2)
features = (maximum, minimum, mean, std, var)

Xreduced = DataTreatment(Xts, :reducesize; win, features)
Xaggregated = DataTreatment(Xts, :aggregate; win, features)