module DataTreatments

using Statistics
using DataFrames

export movingwindow, wholewindow, splitwindow, adaptivewindow
export @evalwindow
include("slidingwindow.jl")

export applyfeat, aggregate, reducesize
include("treatment.jl")

end
