```@meta
CurrentModule = DataTreatments
```

# Normalization

`DataTreatments.jl` delegates normalization logic to the external package
[`Normalization.jl`](https://github.com/JuliaML/Normalization.jl).

In practice:

- core normalization algorithms are provided by `Normalization.jl`;
- `DataTreatments.jl` re-exports the main normalization API (`fit`, `fit!`, `normalize`, `normalize!`);
- `DataTreatments.jl` adds integration for nested/multidimensional dataset elements via its extension layer (`ext/NormalizationExt.jl`);

## Notes on `dims`

`dims` supports:

- `nothing`: global fit on all values,
- `1`: column-wise,
- `2`: row-wise.

When using grouped processing (`groups=...` in `DataTreatment`), keep normalization semantics consistent with grouping logic.

## Examples

```@example
using DataTreatments

X = [rand(4, 2) for _ in 1:10, _ in 1:5]

# Build a normalization spec in DataTreatments
ns = ZScore(dims=1, method=:std)

# Apply normalization (delegated to Normalization.jl + DataTreatments extension)
Y = normalize(X, ns)
```

# References

- `Normalization.jl` package documentation and source code.
- `DataTreatments.jl` extension module: `ext/NormalizationExt.jl`.
- `DataTreatments.jl` normalization spec definitions: `src/normalize.jl`.

# Acknowledgements

We thank the maintainers and contributors of **Normalization.jl** for providing the
core normalization framework used by `DataTreatments.jl`.