```@meta
CurrentModule = DataTreatments
```

# [Imbalance](@id imbalance)

Wrappers around
[Imbalance.jl](https://juliaai.github.io/Imbalance.jl/stable/)
for handling class imbalance in datasets. Each struct stores the
resampling strategy and its parameters, and can be passed to
[`load_dataset`](@ref) via a [`TreatmentGroup`](@ref).

## Oversampling

Oversampling methods generate new synthetic or duplicated samples
for the minority class.

```@docs
RandomOversampler
RandomWalkOversampler
ROSE
SMOTE
BorderlineSMOTE1
SMOTEN
SMOTENC
```

## Undersampling

Undersampling methods reduce the number of samples in the majority
class.

```@docs
RandomUndersampler
ClusterUndersampler
ENNUndersampler
TomekUndersampler
```

## References

The resampling algorithms are provided by Imbalance.jl:
- **Repository**: https://github.com/JuliaAI/Imbalance.jl
- **Docs**: https://juliaai.github.io/Imbalance.jl/stable/