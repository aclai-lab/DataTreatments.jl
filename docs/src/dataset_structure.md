```@meta
CurrentModule = DataTreatments
```

# Dataset Structure

## TargetStructure

```@docs
TargetStructure
get_values(::TargetStructure)
get_labels(::TargetStructure)
```

## DatasetStructure

```@docs
DatasetStructure
```

### Base methods

```@docs
Base.size(::DatasetStructure)
Base.length(::DatasetStructure)
Base.eachindex(::DatasetStructure)
```

### Getter methods

```@docs
get_vnames(::DatasetStructure)
get_datatype(::DatasetStructure)
get_dims(::DatasetStructure)
get_valididxs(::DatasetStructure)
get_missingidxs(::DatasetStructure)
get_nanidxs(::DatasetStructure)
get_hasmissing(::DatasetStructure)
get_hasnans(::DatasetStructure)
```