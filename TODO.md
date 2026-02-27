- Handle datasets with missing values (wine ds)
- Load datasets from csv (quite common)
- Handle NaN in normalization (see Normalization.jl)
- Introduce type (Float32/Float64)
- error when windowing can't handle tiny datas

## [Unreleased]

### Changed
- Normalization responsibilities are now delegated to **Normalization.jl**.
- `DataTreatments.jl` re-exports normalization API (`fit`, `fit!`, `normalize`, `normalize!`) from `Normalization.jl`.
- Internal normalization integration for nested/multidimensional elements is provided through `ext/NormalizationExt.jl`.
- `NormSpec` remains the `DataTreatments.jl` convenience interface to select normalization type and `dims`.

### Documentation
- Updated normalization docs to clarify delegation to `Normalization.jl`.
- Added references and acknowledgements for `Normalization.jl`.