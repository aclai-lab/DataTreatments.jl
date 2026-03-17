```@meta
CurrentModule = DataTreatments
```

# [Treatment Functions](@id treatment)

Treatment functions implement the core data transformations applied to
multidimensional columns (time series, spectrograms, images, etc.) during
[`DataTreatment`](@ref) processing. Two strategies are available:

| Strategy | Function | Output |
|----------|----------|--------|
| **Tabularization** | [`aggregate`](@ref) | Flat scalar matrix — one column per (feature × window × source column) |
| **Dimensionality reduction** | [`reducesize`](@ref) | Matrix of reduced-size arrays — same structure, fewer points |

Both strategies rely on [windowing functions](@ref windowing) to partition each
element into sub-ranges, and on [`safe_feat`](@ref) to robustly apply feature
or reduction functions in the presence of `missing` and `NaN` values.

---

## Aggregate

### Curried constructor

```@docs
aggregate(; win, features)
```

### Internal method

```@docs
aggregate(X::AbstractArray, idx::AbstractVector{Vector{Int}}, float_type::Type; win, features)
```

---

## Reduce Size

### Curried constructor

```@docs
reducesize(; win, reducefunc)
```

### Internal method

```@docs
reducesize(X::AbstractArray, idx::AbstractVector{Vector{Int}}, float_type::DataType; win, reducefunc)
```

---

## Utility Functions

### safe\_feat

```@docs
safe_feat
```

### get\_window\_ranges

```julia
get_window_ranges(intervals::Tuple, cartidx::CartesianIndex) -> Tuple{Vararg{UnitRange{Int}}}
```

Extract window ranges from a tuple of interval vectors using a `CartesianIndex`.
Returns a tuple where each element is `intervals[i][cartidx[i]]`.

Used internally by [`aggregate`](@ref) and [`reducesize`](@ref) to map a
multi-dimensional window index to the corresponding `UnitRange` per dimension.

```julia
intervals = ([1:3, 4:6], [1:5, 6:10])
get_window_ranges(intervals, CartesianIndex(2, 1))  # → (4:6, 1:5)
```

---

## See Also

- [Windowing](@ref windowing) — window functions used by `aggregate` and `reducesize`.
- [`TreatmentGroup`](@ref) — configuration object where `aggrfunc` is specified.
- [Output Datasets](@ref output_dataset) — dataset types produced after treatment.