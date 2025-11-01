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
# make it callable - npoints is passed at execution time
(w::WinFunction)(npoints::Int; kwargs...) = w.func(npoints; w.params..., kwargs...)

"""
    MovingWindow(; window_size::Int, window_step::Int) -> WinFunction

Create a moving window that slides across the time series.

# Parameters
- `window_size`: Number of time points in each window
- `window_step`: Step size between consecutive windows

# Example
```
win = MovingWindow(window_size=10, window_step=5)
intervals = win(100)  # For 100-point time series
```
"""
function MovingWindow(;
    window_size::Int,
    window_step::Int,
)
    WinFunction(movingwindow, (;window_size, window_step))
end

"""
    WholeWindow() -> WinFunction

Create a single window encompassing the entire time series.
Useful for global feature extraction without temporal partitioning.

# Example
```
win = WholeWindow()
intervals = win(100)  # Returns [1:100]
```
"""
WholeWindow(;) = WinFunction(wholewindow, (;))

"""
    SplitWindow(; nwindows::Int) -> WinFunction

Divide the time series into equal non-overlapping segments.

# Parameters
- `nwindows`: Number of equal-sized windows to create

# Example
```
win = SplitWindow(nwindows=4)
intervals = win(100)  # Four 25-point windows
```
"""
SplitWindow(;nwindows::Int) = WinFunction(splitwindow, (; nwindows))

"""
    AdaptiveWindow(; nwindows::Int, relative_overlap::AbstractFloat) -> WinFunction

Create overlapping windows with adaptive sizing based on series length.

# Parameters
- `nwindows`: Target number of windows
- `relative_overlap`: Fraction of overlap between adjacent windows (0.0-1.0)

# Example
```
win = AdaptiveWindow(nwindows=3, relative_overlap=0.1)
intervals = win(100)  # Three adaptive windows with 10% overlap
```
"""
function AdaptiveWindow(;
    nwindows::Int,
    relative_overlap::AbstractFloat,
)
    WinFunction(adaptivewindow, (; nwindows, relative_overlap))
end





"""
    sliding_window_view(A::AbstractArray, window_shape::NTuple; axes=nothing)

Return an array of views of all sliding windows of size `window_shape` over array `A`.
If `axes` is provided, windows are taken along those axes.
"""
function sliding_window_view(A::AbstractArray, window_shape::NTuple; axes=nothing)
    nd = ndims(A)
    sz = size(A)
    if axes === nothing
        axes = 1:nd
        length(window_shape) == nd || throw(ArgumentError("window_shape must match array dimensions"))
    else
        axes = Tuple(axes)
        length(window_shape) == length(axes) || throw(ArgumentError("window_shape and axes must have the same length"))
    end

    # Check window sizes
    for (ax, win) in zip(axes, window_shape)
        sz[ax] >= win || throw(ArgumentError("window_shape cannot be larger than input array shape"))
    end

    # Compute output shape
    out_shape = Tuple(sz[i] - (window_shape[j]-1)*(i in axes ? 1 : 0) for (i,j) in zip(1:nd, 1:nd))
    out_shape = ntuple(i -> sz[i] - (i in axes ? window_shape[findfirst(==(i), axes)] - 1 : 0), nd)
    # The shape of the array of windows
    win_shape = window_shape
    # The final output shape is out_shape..., win_shape...
    # We'll return an array of views

    # Generate all starting indices for the sliding windows
    idx_ranges = [1:(out_shape[d]) for d in 1:nd]
    inds = Iterators.product(idx_ranges...)

    # For each starting index, create a view
    windows = [@view A[ntuple(d -> i[d]:(i[d]+(d in axes ? window_shape[findfirst(==(d), axes)]-1 : 0)), nd)...] for i in inds]
    # Reshape to the output shape
    reshape(windows, out_shape...)
end





adaptivewindow(nwindows::Int64) = x -> slidingwindow(x; nwindows=nwindows)

macro evalwindow(x, winfunc)
    quote
        winfunc = $(esc(winfunc))
        winfunc($(esc(x)))
    end
end

function slidingwindow(
    npoints  :: T;
    nwindows :: Union{T, Tuple{}}=(),
    winsize  :: Union{T, Tuple{}}=(),
    winstep  :: Union{T, Tuple{}}=(),
    overlap  :: Union{T, Tuple{}}=(),
)::Vector{Vector{UnitRange{Int64}}} where {T<:Tuple{Vararg{Int64}}}
    results = Vector{Vector{UnitRange{Int64}}}(undef, length(npoints))

    # collect windows indices dim by dim
    Threads.@threads for i in eachindex(npoints)
        results[i]   = slidingwindow(
            npoints[i];
            nwindows = isempty(nwindows) ? 0 : nwindows[i],
            winsize  = isempty(winsize)  ? 0 : winsize[i],
            winstep  = isempty(winstep)  ? 0 : winstep[i],
            overlap  = isempty(overlap)  ? 0 : overlap[i]
        )
    end

    return results
end

function slidingwindow(
    npoints  :: T;
    nwindows :: T=0,
    winsize  :: T=0,
    winstep  :: T=0,
    overlap  :: T=0,
)::Vector{UnitRange{Int64}} where {T<:Int64}
    # winstep and overlap cannot both be nonzero
    if winstep != 0 && overlap != 0
        throw(ArgumentError("Cannot use winstep and overlap simultaneously: choose one of them."))
    end
    
    # set winstep using overlap
    overlap != 0 && (winstep = winsize - overlap)

    starts  = collect(range(1, npoints, step=winstep))
    indices = Vector{UnitRange{Int64}}(undef, length(starts))

    @inbounds @simd for i in eachindex(starts)
        r = starts[i]
        indices[i] = r:r+winsize-1
    end

    # force_coverage
    filter!((w) -> w.start in 1:npoints && w.stop in 1:npoints, indices)
end

A = rand(1150, 1150);
B = rand(2500);

# Example: 1D sliding windows over B (length 200)
@btime windows_1d = slidingwindow(length(B); winsize=10, winstep=5)
# 6.178 μs (93 allocations: 17.50 KiB)
# 1.150 μs (7 allocations: 11.91 KiB)
windows_1d_tuple = slidingwindow((length(B),); winsize=(10,), winstep=(5,))
@btime windows_2d = slidingwindow(size(A); winsize=(20,20), winstep=(10,10))
# 523.568 ns (11 allocations: 6.17 KiB)
# 530.319 ns (10 allocations: 6.12 KiB)
# 4.865 μs (98 allocations: 12.92 KiB)