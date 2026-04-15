# ---------------------------------------------------------------------------- #
#                       oversampling imbalance structs                         #
# ---------------------------------------------------------------------------- #
struct RandomOversampler{T} <: AbstractBalance
    balance::Base.Callable
    ratios::Union{Real,Dict}
    rng::Int
    try_preserve_type::Bool

    function RandomOversampler(;
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            random_oversample,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

struct RandomWalkOversampler{T} <: AbstractBalance
    balance::Base.Callable
    ratios::Union{Real,Dict}
    rng::Int
    try_preserve_type::Bool

    function RandomWalkOversampler(;
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            _random_walk_oversample,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

function _random_walk_oversample(X, y; kwargs...)
    cat_inds = X isa AbstractMatrix{<:Union{Int,Union{Missing,Int}}} ? collect(1:size(X, 2)) : Int[]
    random_walk_oversample(X, y, cat_inds; kwargs...)
end

struct ROSE{T} <: AbstractBalance
    balance::Base.Callable
    s::AbstractFloat
    ratios::T
    rng::Int
    try_preserve_type::Bool

    function ROSE(;
        s::AbstractFloat=1.0,
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            rose,
            s,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

struct SMOTE{T} <: AbstractBalance
    balance::Base.Callable
    k::Int
    ratios::T
    rng::Int
    try_preserve_type::Bool

    function SMOTE(;
        k::Int=5,
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            smote,
            k,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

struct BorderlineSMOTE1{T} <: AbstractBalance
    balance::Base.Callable
    m::Int
    k::Int
    ratios::T
    rng::Int
    try_preserve_type::Bool
    verbosity::Int

    function BorderlineSMOTE1(;
        m::Int=5,
        k::Int=5,
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true,
        verbosity::Int=0
    ) where {T<:Union{Real,Dict}}
        new{T}(
            borderline_smote1,
            m,
            k,
            ratios,
            rng,
            try_preserve_type,
            verbosity
        )
    end
end

struct SMOTEN{T} <: AbstractBalance
    balance::Base.Callable
    k::Int
    ratios::T
    rng::Int
    try_preserve_type::Bool

    function SMOTEN(;
        k::Int=5,
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            _smoten,
            k,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

function _smoten(X, y; kwargs...)
    @show typeof(X)
    return X isa AbstractMatrix{<:Union{Int,Union{Missing,Int}}} ?
        smoten(X, y; kwargs...) :
        error("SMOTEN works only on discrete datasets, consider using " *
        "SMOTE or SMOTENC instead.")
end

struct SMOTENC{T} <: AbstractBalance
    balance::Base.Callable
    k::Int
    knn_tree::AbstractString
    ratios::T
    rng::Int
    try_preserve_type::Bool

    function SMOTENC(;
        k::Int=5,
        knn_tree::AbstractString="Brute",
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            _smotenc,
            k,
            knn_tree,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

function _smotenc(X, y; kwargs...)
    cat_inds = X isa AbstractMatrix{<:Union{Int,Union{Missing,Int}}} ? collect(1:size(X, 2)) : Int[]
    smotenc(X, y, cat_inds; kwargs...)
end

# ---------------------------------------------------------------------------- #
#                      undersampling imbalance structs                         #
# ---------------------------------------------------------------------------- #
struct RandomUndersampler{T} <: AbstractBalance
    balance::Base.Callable
    ratios::Union{Real,Dict}
    rng::Int
    try_preserve_type::Bool

    function RandomUndersampler(;
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            random_undersample,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

struct ClusterUndersampler{T} <: AbstractBalance
    balance::Base.Callable
    mode::AbstractString
    maxiter::Int
    ratios::Union{Real,Dict}
    rng::Int

    function ClusterUndersampler(;
        mode::AbstractString="nearest", 
        maxiter::Int=100, 
        ratios::T=1.0,
        rng::Int=42,
    ) where {T<:Union{Real,Dict}}
        new{T}(
            cluster_undersample,
            mode,
            maxiter,
            ratios,
            rng
        )
    end
end

struct ENNUndersampler{T} <: AbstractBalance
    balance::Base.Callable
    k::Int
    keep_condition::AbstractString
    force_min_ratios::Bool
    min_ratios::Union{Real,Dict}
    rng::Int
    try_preserve_type::Bool

    function ENNUndersampler(;
        k::Int=5,
        keep_condition::AbstractString="mode", 
        force_min_ratios::Bool=false, 
        min_ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            enn_undersample,
            k,
            keep_condition,
            force_min_ratios,
            min_ratios,
            rng,
            try_preserve_type
        )
    end
end

struct TomekUndersampler{T} <: AbstractBalance
    balance::Base.Callable
    force_min_ratios::Bool
    min_ratios::Union{Real,Dict}
    rng::Int
    try_preserve_type::Bool

    function TomekUndersampler(;
        force_min_ratios::Bool=false, 
        min_ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            tomek_undersample,
            force_min_ratios,
            min_ratios,
            rng,
            try_preserve_type
        )
    end
end

# ---------------------------------------------------------------------------- #
#                               internal getters                               #
# ---------------------------------------------------------------------------- #
_get_balfunc(b::AbstractBalance) = b.balance

function _get_balkw(b::AbstractBalance)
    names = Tuple(filter(!=(:balance), fieldnames(typeof(b))))
    vals = Tuple(getfield(b, n) for n in names)
    return NamedTuple{names}(vals)
end

