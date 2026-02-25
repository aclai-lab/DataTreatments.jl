using Test
using DataTreatments
const DT = DataTreatments

using MLJ, DataFrames
using SoleData: Artifacts

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

df = DataFrame(
    vec_col  = [rand(Float64, 5), missing, rand(Float64, 5), rand(Float64, 5), missing],
    str_col  = ["hello", "world", missing, "foo", "bar"],
    int_col  = [1, missing, 3, 4, missing],
    float_col = [1.1, 2.2, missing, 4.4, missing]
)

# ---------------------------------------------------------------------------- #
#                            tabular datatreatment                             #
# ---------------------------------------------------------------------------- #
dt = DataTreatment(Xc)