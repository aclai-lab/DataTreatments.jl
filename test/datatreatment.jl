using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
    DataFrame(
        str_col  = [missing, "blue", "green", "red", "blue"],
        sym_col  = [:circle, :square, :triangle, :square, missing],
        img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5],
        int_col  = Int[10, 20, 30, 40, 50],
        V1 = [NaN, missing, 3.0, 4.0, 5.6],
        V2 = [2.5, missing, 4.5, 5.5, NaN],
        ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
        V4 = [4.1, NaN, NaN, 7.1, 5.5],
        V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
        ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
        ts3 = [[1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2], NaN, NaN, missing, [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]],
        V3 = [3.2, 4.2, 5.2, missing, 2.4],
        ts4 = [[6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8], missing, [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8], [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8], [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]],
        img1 = [create_image(i) for i in 1:5],
        cat_col  = categorical(["small", "medium", missing, "small", "large"]),
        uint_col = UInt32[1, 2, 3, 4, 5],
        img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
    )
end

df = build_test_df()

t_classif = ["classA", "classB", "classC", "classA", "classB"]
t_regress = [1.2, 3.4, 2.2, 4.8, 0.9]

ds = load_dataset(df)
ds = load_dataset(df, t_classif)
ds = load_dataset(df, t_regress)

# @btime load_dataset(df);
# # 313.931 μs (3577 allocations: 227.88 KiB)
# # 436.629 μs (3908 allocations: 252.18 KiB)

# using Test
# using DataTreatments

# using MLJ
# using DataFrames, Random
# using SoleData: Artifacts

# using CategoricalArrays

# # ---------------------------------------------------------------------------- #
# #                                load dataset                                  #
# # ---------------------------------------------------------------------------- #
# Xc, yc = @load_iris
# Xc = DataFrame(Xc)

# Xr, yr = @load_boston
# Xr = DataFrame(Xr)

# natopsloader = Artifacts.NatopsLoader()
# Xts, yts = Artifacts.load(natopsloader)


# ds = load_dataset(Xc, yc)

# ds =load_dataset(
#     Xc, yc,
#     TreatmentGroup(name_expr=["petal_length", "petal_width"], grouped=true)
# )

# ds =load_dataset(Xr, yr)

# ds =load_dataset(
#     Xts, yts,
#     TreatmentGroup(
#         dims=1,
#         aggrfunc=aggregate(
#             features=(mean, maximum),
#             win=(adaptivewindow(nwindows=5, overlap=0.4),)
#         ),
#         groupby=:feat
#     )
# )

# ds =load_dataset(
#     Xts, yts,
#     TreatmentGroup(
#         dims=1,
#         aggrfunc=reducesize(
#             reducefunc=mean,
#             win=(splitwindow(nwindows=5),)
#         )
#     );
# )

# # ds =load_dataset(
# #     Xts, yts,
# #     TreatmentGroup(
# #         dims=1,
# #         aggrfunc=SF.reducesize()
# #     );
# #     data_type=tabular
# # )
# # @test isempty(SF.get_data(ds))

# # @inferred load_dataset(Xr, yr)
# # @code_warntype load_dataset(Xr, yr)