```@meta
CurrentModule = DataTreatments
```
# [DataTreatment](@id datatreatment)

```@setup doda
import Random.seed!
seed!(1234)
using Statistics
```

## What is a data treatment?

**DataTreatments.jl** provides tools for manipulating and analyzing multidimensional datasets, meaning datasets whose elements are not single numbers but signals (e.g., audio or time-series sensor inputs) or higher‑dimensional structures (e.g., images and beyond).

### Why is data treatment necessary?

Multidimensional data such as audio or images often have very large sizes due to high resolution. In many cases, we cannot perform analysis or machine learning directly on such large data. Therefore, data‑compression algorithms are useful to reduce dimensionality while minimizing information loss.

One of the most widely used approaches is **windowing**.

### Two common scenarios

- **`reducesize`**  
  The output dataset keeps the same overall structure as the input, but with smaller elements.

  **Example (reducesize):**

  ```@repl doda
  using DataTreatments

  X = [rand(4) for _ in 1:3, _ in 1:2]

  vnames = [:ch1, :ch2];
  win = splitwindow(nwindows=2);
  features = (mean, maximum);

  dt_rs = DataTreatment(X, :reducesize; vnames, win, features);

  get_dataset(dt_rs)
  ```

- **`aggregate`**  
  The dataset is resized and also transformed into a tabular dataset, where windows become consecutive columns in the output.

  **Example (reducesize):**
  ```@repl doda
  using DataTreatments

  X = [rand(4) for _ in 1:2, _ in 1:2]

  vnames = [:ch1, :ch2];
  win = splitwindow(nwindows=2);
  features = (mean,);

  dt_ag = DataTreatment(X, :aggregate; vnames, win, features);

  get_dataset(dt_ag)
  get_featureid(dt_ag)
  ```

### Note

Windowing is especially useful for normalizing datasets whose elements have different sizes (common with audio files). By using a fixed number of windows, object sizes are normalized and subsequent analysis becomes more reliable.