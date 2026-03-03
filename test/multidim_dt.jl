using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using CategoricalArrays

# ---------------------------------------------------------------------------- #
#          dataset with only 2D multidimensional features features             #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# missing
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [missing, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), missing]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X

# nan
df = DataFrame(
    ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
)

is_multidim_dataset(df) == true
has_uniform_element_size(df) == false

dt = DataTreatment(df) |> get_X
dt = DataTreatment(df; float_type=Float32) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5)) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), float_type=Float32) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; win=adaptivewindow(nwindows=2, overlap=0.5), aggrtype=:reducesize, float_type=Float32) |> get_X

dt = DataTreatment(df; features=(mean,)) |> get_X
dt = DataTreatment(df; features=(mean,), float_type=Float32) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize) |> get_X
dt = DataTreatment(df; features=(mean,), aggrtype=:reducesize, float_type=Float32) |> get_X