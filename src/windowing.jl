# ---------------------------------------------------------------------------- #
#                               core functions                                 #
# ---------------------------------------------------------------------------- #
function _slidingwindow(
    npoints :: T;
    winsize :: T,
    winstep :: T
)::Vector{UnitRange{Int}} where {T<:Int}
    winstep == 0 && (winstep = winsize)
    starts  = collect(range(1, npoints, step=winstep))
    indices = Vector{UnitRange{Int}}(undef, length(starts))

    @inbounds @simd for i in eachindex(starts)
        r = starts[i]
        indices[i] = r:r + winsize - 1
    end

    # force_coverage
    filter!((w) -> w.start in 1:npoints && w.stop in 1:npoints, indices)
end

function _fixedwindow(
    npoints  :: T;
    nwindows :: T,
    overlap  :: Float64=0.0
)::Vector{UnitRange{Int}} where {T<:Int}
    r_overlap = round(Int, npoints / nwindows * overlap)
    starts = round.(Int, collect(range(1, npoints+1, step=npoints/nwindows)))
    indices = Vector{UnitRange{Int}}(undef, nwindows)

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

# ---------------------------------------------------------------------------- #
#                        one-dimensional windowing funcs                       #
# ---------------------------------------------------------------------------- #
# These functions set the parameters for the various windowing algorithms.
# They will then be passed to the @evalwindow macro.
# They can also be used standalone, but only on vectors.
# For windowing on multidimensional data, it is recommended to use @evalwindow.
"""
    movingwindow(; winsize::Int[, winstep::Int]) -> Function

Creates a moving (sliding) window function with fixed window size and step.

# Keyword Arguments
- `winsize::Int`: Size of each window
- `winstep::Int`: Step between consecutive windows (defaults to `winsize` if 0)

# Returns
- `Function`: A function that takes `npoints::Int` and returns `Vector{UnitRange{Int}}`

# Example
```julia
wfunc = movingwindow(winsize=10, winstep=5)
windows = wfunc(100)  # apply to sequence of length 100
```

# Use with macro
```julia
A = rand(200)
windows = @evalwindow A movingwindow(winsize=10, winstep=5)
```
"""
movingwindow(;winsize::Int, winstep::Int=0)::Function = 
    npoints -> _slidingwindow(npoints; winsize, winstep)

"""
    wholewindow() -> Function

Creates a window function that returns a single window covering the entire sequence.

# Returns
- `Function`: A function that takes `npoints::Int` and returns a single window `[1:npoints]`

# Example
```julia
wfunc = wholewindow()
windows = wfunc(100)  # returns [1:100]
```

# Use with macro
```julia
A = rand(200)
windows = @evalwindow A wholewindow()
```
"""
wholewindow()::Function =
    npoints -> _slidingwindow(npoints; winsize=npoints, winstep=npoints)

"""
    splitwindow(; nwindows::Int) -> Function

Creates a window function that splits the sequence into a fixed number of non-overlapping windows.

# Keyword Arguments
- `nwindows::Int`: Number of windows to create

# Returns
- `Function`: A function that takes `npoints::Int` and returns `Vector{UnitRange{Int}}`

# Example
```julia
wfunc = splitwindow(nwindows=5)
windows = wfunc(100)  # splits into 5 equal windows
```

# Use with macro
```julia
A = rand(200)
windows = @evalwindow A splitwindow(nwindows=5)
```
"""
splitwindow(;nwindows::Int)::Function =
    npoints -> _fixedwindow(npoints; nwindows)

"""
    adaptivewindow(; nwindows::Int[, overlap::Float64=0.0]) -> Function

Creates a window function that adaptively divides the sequence into windows with optional overlap.

# Keyword Arguments
- `nwindows::Int`: Number of windows to create
- `overlap::Float64`: Relative overlap between windows (0.0 to 1.0), defaults 0.0.

# Returns
- `Function`: A function that takes `npoints::Int` and returns `Vector{UnitRange{Int}}`

# Example
```julia
wfunc = adaptivewindow(nwindows=5, overlap=0.2)
windows = wfunc(100)  # 5 windows with 20% overlap
```

# Use with macro
```julia
A = rand(200)
windows = @evalwindow A adaptivewindow(nwindows=5, overlap=0.2)
```
"""
adaptivewindow(;nwindows::Int, overlap::Float64=0.0)::Function =
    npoints -> _fixedwindow(npoints; nwindows, overlap)

# ---------------------------------------------------------------------------- #
#                       multi-dimensional windowing macro                      #
# ---------------------------------------------------------------------------- #
"""
    @evalwindow(X, winfuncs...) -> Tuple{Vararg{Vector{UnitRange{Int}}}}

Apply window functions to each dimension of an array.

This macro evaluates window functions for each dimension of an array, automatically 
determining the appropriate number of points from the array's size. If fewer window 
functions are provided than dimensions, the last function is reused for remaining dimensions.

# Arguments
- `X`: Input array whose dimensions determine window parameters
- `winfuncs...`: One or more window functions (e.g., `movingwindow()`, `splitwindow()`, `adaptivewindow()`)

# Returns
- `Tuple{Vararg{Vector{UnitRange{Int}}}}`: Tuple of vectors containing window ranges for each dimension

# Behavior
- Each window function is applied to the corresponding dimension size
- If `length(winfuncs) < ndims(X)`, the last function is repeated for remaining dimensions
- Window functions must be created using `movingwindow`, `splitwindow`, `adaptivewindow`, or `wholewindow`

# Examples

## Single dimension
```julia
X = rand(200)
windows = @evalwindow X movingwindow(winsize=50, winstep=25)
```

## Two dimensions with different windows
```julia
X = rand(200, 120)
windows = @evalwindow X splitwindow(nwindows=4) adaptivewindow(nwindows=3, overlap=0.1)
# First dimension: 4 non-overlapping windows
```

## Reusing window function for multiple dimensions
```julia
X = rand(100, 120, 80)
windows = @evalwindow X splitwindow(nwindows=5)
```

# See Also
- [`movingwindow`](@ref): Fixed-size sliding windows
- [`splitwindow`](@ref): Equal non-overlapping windows
- [`adaptivewindow`](@ref): Windows with overlap
- [`wholewindow`](@ref): Single window covering entire dimension
"""
macro evalwindow(x, winfuncs...)
    esc_winfuncs = map(esc, winfuncs)
    quote
        _x = $(esc(x))
        _w = ($(esc_winfuncs...),)
        dims = size(_x)
        
        # apply each window function to corresponding dimension
        # if more dims than functions, reuse the last function
        tuple((
            let idx = min(i, length(_w))
                _w[idx](dims[i])
            end
            for i in 1:length(dims)
        )...)
    end
end
