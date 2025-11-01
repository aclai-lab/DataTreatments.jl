# https://ceur-ws.org/Vol-3010/PAPER_06.pdf
# https://medium.com/@whyamit404/understanding-sliding-window-in-numpy-50c86cac822b

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractWinFunction end
abstract type AbstractWinParams   end

# ---------------------------------------------------------------------------- #
#                                  windowing                                   #
# ---------------------------------------------------------------------------- #
"""
    WinFunction <: AbstractWinFunction

Callable wrapper for windowing algorithms with parameters.

# Fields
- `func::Function`: The windowing implementation function
- `params::AbstractWinParams`: Algorithm-specific parameters
"""
struct WinFunction <: AbstractWinFunction
    func   :: Function
    params :: AbstractWinParams
end

# function slidingwindow(
#     npoints  :: T;
#     nwindows :: Union{T, Tuple{}}=(),
#     winsize  :: Union{T, Tuple{}}=(),
#     winstep  :: Union{T, Tuple{}}=()
# )::Vector{Vector{UnitRange{Int64}}} where {T<:Tuple{Vararg{Int64}}}
#     results = Vector{Vector{UnitRange{Int64}}}(undef, length(npoints))

#     # collect windows indices dim by dim
#     Threads.@threads for i in eachindex(npoints)
#         results[i]   = slidingwindow(
#             npoints[i];
#             nwindows = isempty(nwindows) ? 0 : nwindows[i],
#             winsize  = isempty(winsize)  ? 0 : winsize[i],
#             winstep  = isempty(winstep)  ? 0 : winstep[i],
#         )
#     end

#     return results
# end

function slidingwindow(
    npoints :: T;
    winsize :: T=0,
    winstep :: T=0
)::Vector{UnitRange{Int64}} where {T<:Int64}
    starts  = collect(range(1, npoints, step=winstep))
    indices = Vector{UnitRange{Int64}}(undef, length(starts))

    @inbounds @simd for i in eachindex(starts)
        r = starts[i]
        indices[i] = r:r + winsize - 1
    end

    # force_coverage
    filter!((w) -> w.start in 1:npoints && w.stop in 1:npoints, indices)
end

function adaptedwindow(
    npoints  :: T;
    nwindows :: T,
    overlap  :: Float64=0.0
)::Vector{UnitRange{Int64}} where {T<:Int64}
    r_overlap = round(Int64, npoints / nwindows * overlap)
    starts = round.(Int64, collect(range(1, npoints+1, step=npoints/nwindows)))
    indices = Vector{UnitRange{Int64}}(undef, nwindows)

    @inbounds @simd for i in 1:nwindows
        if i == 1
            indices[i] = starts[i] : starts[i+1] -1 + r_overlap
        elseif i == nwindows
            indices[i] = starts[i] - r_overlap : starts[i+1] - 1
        else
            indices[i] = starts[i] - r_overlap : starts[i+1] - 1 + r_overlap
        end
    end

    return indices
end

movingwindow(;winsize::Int64, winstep::Int64=0) = npoints -> slidingwindow(npoints; winsize, winstep)
wholewindow() = npoints -> slidingwindow(npoints; winsize=npoints, winstep=npoints)
splitwindow(;nwindows::Int64) = npoints -> adaptedwindow(npoints; nwindows)
adaptivewindow(;nwindows::Int64, overlap::Float64=0.0) = npoints -> adaptedwindow(npoints; nwindows, overlap)

macro evalwindow(x, winfunc)
    quote
        _x = $(esc(x))
        _x isa Int64 ?
            $(esc(winfunc))(_x) :
            $(esc(winfunc))(length(_x))
    end
end

using SoleXplorer

A = rand(1150, 1150);
B = rand(2500);

points = length(B)
SX_w = MovingWindow(window_size=10, window_step=5)
SX_movingwindow = SX_w(points)
DT_w = movingwindow(winsize=10, winstep=5)
DT_movingwindow = @evalwindow points DT_w
SX_movingwindow == DT_movingwindow

@btime begin
    points = length(B)
    SX_w = MovingWindow(window_size=10, window_step=5)
    SX_movingwindow = SX_w(points)
end
# 1.344 μs (5 allocations: 7.91 KiB)

@btime begin
    DT_w = movingwindow(winsize=10, winstep=5)
    DT_movingwindow = @evalwindow B DT_w
end
# 1.273 μs (7 allocations: 11.91 KiB)

SX_w = WholeWindow()
SX_wholewindow = SX_w(points)
DT_w = wholewindow()
DT_wholewindow = @evalwindow points DT_w
SX_wholewindow == DT_wholewindow

@btime begin
    points = length(B)
    SX_w = WholeWindow()
    SX_wholewindow = SX_w(points)
end
# 849.787 ns (15 allocations: 448 bytes)

@btime begin
    DT_w = wholewindow()
    DT_wholewindow = @evalwindow B DT_w
end
# 58.831 ns (7 allocations: 192 bytes)

SX_w = SplitWindow(nwindows=11)
SX_splitwindow = SX_w(points)
DT_w = splitwindow(nwindows=11)
DT_splitwindow = @evalwindow points DT_w
length(DT_splitwindow) == 11

@btime begin
    points = length(B)
    SX_w = SplitWindow(nwindows=11)
    SX_splitwindow = SX_w(points)
end
# 4.385 μs (102 allocations: 2.89 KiB)

@btime begin
    DT_w = splitwindow(nwindows=11)
    DT_splitwindow = @evalwindow B DT_w
end
# 169.756 ns (6 allocations: 544 bytes)

SX_w = AdaptiveWindow(nwindows=11, relative_overlap=0.1)
SX_adaptivewindow = SX_w(points)
DT_w = adaptivewindow(nwindows=11, overlap=0.1)
DT_adaptivewindow = @evalwindow points DT_w
length(DT_splitwindow) == 11

@btime begin
        points = length(B)
    SX_w = AdaptiveWindow(nwindows=11, relative_overlap=0.1)
    SX_adaptivewindow = SX_w(points)
end
# 4.342 μs (102 allocations: 2.89 KiB)

@btime begin
    DT_w = adaptivewindow(nwindows=11, overlap=0.1)
    DT_adaptivewindow = @evalwindow B DT_w
end
# 167.503 ns (6 allocations: 544 bytes)
