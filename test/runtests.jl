using Test
using DataTreatments

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("FeatureSet", ["featureset.jl",]),
    ("Windowing",  ["windowing.jl",]),
    ("Dataset Inspect", ["inspecting.jl"]),
    ("Load Dataset", ["load_dataset.jl"]),
    ("Examples", ["examples.jl"]),
    ("Multidim Treatments",  ["multidim_treatment.jl",]),
    ("Treatment Groups",  ["treatment_group.jl",]),
    ("DataTreatment", ["datatreatment.jl"]),
]

@testset "DataTreatments.jl" begin
    for ts in eachindex(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
    println()
end

