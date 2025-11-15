using Documenter
using DataTreatments

DocMeta.setdocmeta!(DataTreatments, :DocTestSetup, :(using DataTreatments); recursive = true)

makedocs(;
    modules=[DataTreatments],
    authors="Riccardo Pasini",
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
        "Home"          => "index.md",
        "Api"           => "api.md",
        "FeatureSet"    => "featureset.md",
        "Normalization" => "normalization.md",
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
