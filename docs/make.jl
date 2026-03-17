using Documenter
using DataTreatments

DocMeta.setdocmeta!(DataTreatments, :DocTestSetup, :(using DataTreatments); recursive = true)

makedocs(;
    modules=[DataTreatments],
    authors="Michele Ghiotti, Federico Manzella, Riccardo Pasini",
    repo=Documenter.Remotes.GitHub("PasoStudio73", "DataTreatments.jl"),
    sitename="DataTreatments.jl",
    format=Documenter.HTML(;
        size_threshold=4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PasoStudio73.github.io/DataTreatments.jl",
        edit_link="main", # possibly this line is dangerous after publishing
        assets=String[],
    ),
    pages=[
        "Dataset Structure" => "dataset_structure.md",
        # "Home" => "index.md",
        # "DataTreatment" => "treatment.md",
        # "Grouping" => "grouping.md",
        # "Normalization" => "normalization.md",
        # "FeatureSet" => "featureset.md",
        # "Api" => "api.md",
    ],
    warnonly=:true,
)

deploydocs(;
    repo = "github.com/PasoStudio73/DataTreatments.jl",
    devbranch = "main",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
