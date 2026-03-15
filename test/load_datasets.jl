using DataTreatments

using Downloads
using CSV, DataFrames
using CategoricalArrays

const FloatType = Float32

# ---------------------------------------------------------------------------- #
#                      proposed method to work with csvs                       #
# ---------------------------------------------------------------------------- #

# datasets to tests
# ds = dataset nanme
# y = supervised column
# rm = columns to be igbnored (es: ids)
datasets = [
    (ds="banknote", y="class"),
    (ds="breast-cancer", y="diagnosis", rm="id"),
    (ds="car_evaluation", y="decision"),
    (ds="cryotherapy", y="Result_of_Treatment"),
    (ds="diabetes", y="Outcome"),
    (ds="divorce", y="Class"),
    (ds="haberman", y="status"),
    (ds="heart", y="target"),
    (ds="house-votes-84", y="party"),
    (ds="HTRU_2", y="class"),
    (ds="Mammographic_Masses", y="Severity"),
    (ds="monk", y="'class'", rm="id"),
    (ds="mushrooms", y="class"),
    (ds="Occupancy", y="Occupancy"),
    (ds="penguins", y="species"),
    (ds="seeds", y="V8"),
    (ds="soybean-small", y="v36"),
    (ds="statlog", y="presence"),
    (ds="tic-tac-toe", y="class"),
    (ds="urinary", y="diagnosis", rm="sample_id"),
    (ds="Vehicle", y="Class"),
]

# base URL for datasets on PasoStudio73 GitHub
base_url = "https://raw.githubusercontent.com/PasoStudio73/Datasets/main/csv/"

# download datasets
Threads.@threads for dataset in datasets
    url = base_url * dataset.ds * ".csv"
    local_path = joinpath(@__DIR__, "datasets", dataset.ds * ".csv")  
    mkpath(dirname(local_path))
    isfile(local_path) || Downloads.download(url, local_path)
end

# local filepath
filepaths = [joinpath(@__DIR__, "datasets/", dataset.ds * ".csv") for dataset in datasets]

datatreatments = Vector{DataTreatment}(undef, length(datasets))

# construct DataTreatment objects from loaded datasets
Threads.@threads for i in eachindex(filepaths)
    dataset = CSV.read(filepaths[i], DataFrame)

    target, features = haskey(datasets[i], :y) ?
        (dataset[!, datasets[i].y], select(dataset, Not(datasets[i].y))) :
        (nothing, dataset)

    # check if there's any columns to be removed in advance from the dataset
    haskey(datasets[i], :rm) && select!(features, Not(datasets[i].rm))

    datatreatments[i] = DataTreatment(
        features,
        target;
        float_type=FloatType
    )
end

