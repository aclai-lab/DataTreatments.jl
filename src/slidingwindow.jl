# ---------------------------------------------------------------------------- #
#                               core functions                                 #
# ---------------------------------------------------------------------------- #
function _slidingwindow(
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

function _adaptedwindow(
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

# ---------------------------------------------------------------------------- #
#                         1 dimensional windowing funcs                        #
# ---------------------------------------------------------------------------- #
movingwindow(;winsize::Int64, winstep::Int64=0) = npoints -> _slidingwindow(npoints; winsize, winstep)
wholewindow() = npoints -> _slidingwindow(npoints; winsize=npoints, winstep=npoints)
splitwindow(;nwindows::Int64) = npoints -> _adaptedwindow(npoints; nwindows)
adaptivewindow(;nwindows::Int64, overlap::Float64=0.0) = npoints -> _adaptedwindow(npoints; nwindows, overlap)

# ---------------------------------------------------------------------------- #
#                       multi dimensional windowing macro                      #
# ---------------------------------------------------------------------------- #
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
