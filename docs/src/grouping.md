```@meta
CurrentModule = DataTreatments
```
# [Grouping](@id grouping)

## Why group features?

Often you work with datasets whose columns have different units of measurement. A simple example is a medical dataset where we have audio files together with recordings of electromagnetic pulses.

Several machine learning algorithms explicitly require the input data to be normalized (typically in a 0–1 range). We could normalize column by column, but this risks flattening the dataset with a consequent loss of information.

That is why the possibility to group columns according to a certain logic is useful.

The DataTreatments package provides this functionality not only for multidimensional datasets, but also for tabular datasets.

```@docs
groupby(::DataTreatment, ::Symbol...)
groupby(::DataFrame, ::Vector{Vector{Symbol}})
```

