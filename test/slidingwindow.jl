using Test
using DataTreatments

@testset "1-dimensional windowing" begin
    wfunc = movingwindow(winsize=10, winstep=5)
    windows = wfunc(100)
    @test length(windows) == 19
    @test windows isa Vector{UnitRange{Int64}}

    wfunc = wholewindow()
    windows = wfunc(100)
    @test windows == UnitRange{Int64}[1:100]

    wfunc = splitwindow(nwindows=5)
    windows = wfunc(100)
    @test windows == UnitRange{Int64}[1:20,21:40,41:60,61:80,81:100]

    wfunc = adaptivewindow(nwindows=5, overlap=0.2)
    windows = wfunc(100)
    @test windows == UnitRange{Int64}[1:24,17:44,37:64,57:84,77:100]

    A = rand(200)
    windows = @evalwindow A movingwindow(winsize=10, winstep=5)
end


@testset "2-dimensional windowing" begin
    A = rand(200, 120)

    windows = @evalwindow A movingwindow(winsize=10, winstep=5)
    @test length(windows) == 2
    @test windows isa Tuple
    @test windows[1] isa Vector{UnitRange{Int64}}
    @test windows[2] isa Vector{UnitRange{Int64}}
    @test length(windows[1]) == 39
    @test length(windows[2]) == 23

    windows = @evalwindow A wholewindow()
    @test windows == (UnitRange{Int64}[1:200], UnitRange{Int64}[1:120])

    win = (splitwindow(nwindows=5), splitwindow(nwindows=2))
    windows = @evalwindow A win...
    @test windows == 
        (UnitRange{Int64}[1:40, 41:80, 81:120, 121:160, 161:200],
        UnitRange{Int64}[1:60, 61:120])

    windows = @evalwindow A adaptivewindow(nwindows=5, overlap=0.2)
    @test windows ==
        (UnitRange{Int64}[1:48, 33:88, 73:128, 113:168, 153:200],
        UnitRange{Int64}[1:29, 20:53, 44:77, 68:101, 92:120])
end

@testset "3-dimensional windowing" begin
    A = rand(200, 120, 50)

    windows = @evalwindow A movingwindow(winsize=10, winstep=5)
    @test length(windows) == 3
    @test windows isa Tuple
    @test all(w -> w isa Vector{UnitRange{Int64}}, windows)
    @test length(windows[1]) == 39  # (200-10)÷5 + 1
    @test length(windows[2]) == 23  # (120-10)÷5 + 1
    @test length(windows[3]) == 9   # (50-10)÷5 + 1
    @test windows[1][1] == 1:10
    @test windows[2][1] == 1:10
    @test windows[3][1] == 1:10

    windows = @evalwindow A movingwindow(winsize=10) wholewindow()
    @test length(windows) == 3
    @test length(windows[1]) == 20
    @test windows[2] == UnitRange{Int64}[1:120]
    @test windows[3] == UnitRange{Int64}[1:50]
    @test windows[1][1] == 1:10
    @test windows[2][1] == 1:120
    @test windows[3][1] == 1:50

    x_wfunc = movingwindow(winsize=10, winstep=5)
    y_wfunc = adaptivewindow(nwindows=5, overlap=0.2)
    z_wfunc = splitwindow(nwindows=2)
    windows = @evalwindow A x_wfunc y_wfunc z_wfunc

    # Test x dimension (movingwindow)
    @test windows[1] isa Vector{UnitRange{Int64}}
    @test length(windows[1]) == 39
    @test windows[1][1] == 1:10
    @test windows[1][2] == 6:15

    # Test y dimension (adaptivewindow with 5 windows and 20% overlap)
    @test windows[2] isa Vector{UnitRange{Int64}}
    @test length(windows[2]) == 5
    @test windows[2][1] == 1:29
    @test windows[2][3] == 44:77
    @test windows[2][5] == 92:120

    # Test z dimension (splitwindow into 2 parts)
    @test windows[3] isa Vector{UnitRange{Int64}}
    @test length(windows[3]) == 2
    @test windows[3] == UnitRange{Int64}[1:25, 26:50]
end