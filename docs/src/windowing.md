```@meta
CurrentModule = DataTreatments
```

# [Windowing](@id windowing)

Windowing functions partition a sequence (or each dimension of a multidimensional
array) into contiguous sub-ranges. These ranges are used internally by
[`aggregate`](@ref) and [`reducesize`](@ref) to extract features from
sub-segments of time series, spectrograms, and other array-valued columns.

All windowing functions return a **closure** that accepts `npoints::Int` (the
length of the dimension) and produces a `Vector{UnitRange{Int}}` of window
indices.

---

## Window Functions

### movingwindow

```@docs
movingwindow
```

### wholewindow

```@docs
wholewindow
```

### splitwindow

```@docs
splitwindow
```

### adaptivewindow

```@docs
adaptivewindow
```

---

## Multidimensional Windowing

### @evalwindow

```@docs
@evalwindow
```

---

## Internal Functions

The public window functions are thin wrappers around two internal routines.
These are not exported but are documented here for reference.

### \_slidingwindow

```julia
_slidingwindow(npoints::Int; winsize::Int, winstep::Int) -> Vector{UnitRange{Int}}
```

Core sliding-window implementation used by [`movingwindow`](@ref) and
[`wholewindow`](@ref).

Generates windows of size `winsize` starting every `winstep` points.
If `winstep == 0`, it defaults to `winsize` (non-overlapping).
A final **force-coverage** filter removes any window whose `start` or `stop`
falls outside `1:npoints`, so all returned ranges are guaranteed to be
within bounds.

### \_fixedwindow

```julia
_fixedwindow(npoints::Int; nwindows::Int, overlap::Float64=0.0) -> Vector{UnitRange{Int}}
```

Core fixed-partition implementation used by [`splitwindow`](@ref) and
[`adaptivewindow`](@ref).

Divides `npoints` into `nwindows` segments. Start positions are calculated
with `range(1, npoints+1, step=npoints/nwindows)` and rounded to the nearest
integer. When `overlap > 0`, each window is extended by
`round(Int, npoints / nwindows * overlap)` points:

- The **first** window extends only on the right.
- The **last** window extends only on the left.
- **Interior** windows extend on both sides.

---

## See Also

- [`aggregate`](@ref) — uses window functions to extract features from multidimensional columns.
- [`reducesize`](@ref) — uses window functions to reduce the size of multidimensional columns.
- [`TreatmentGroup`](@ref) — configuration object where `aggrfunc` (which wraps window functions) is specified.