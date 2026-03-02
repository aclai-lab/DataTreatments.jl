```@example
using DataTreatments
using DataFrames
using CategoricalArrays
```

```@example
# ---------------------------------------------------------------------------- #
#                     dataset with only discrete features                      #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    str_col  = ["red", "blue", "green", "red", "blue"],                        # AbstractString
    sym_col  = [:circle, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical(["small", "medium", "large", "small", "large"]),    # CategoricalValue
    uint_col = UInt32[1, 2, 3, 4, 5],                                          # UInt32
    int_col  = Int[10, 20, 30, 40, 50]                                         # Int
)
```

```@example