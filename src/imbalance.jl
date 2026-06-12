# ---------------------------------------------------------------------------- #
#                       oversampling imbalance structs                         #
# ---------------------------------------------------------------------------- #
"""
    RandomOversampler(; ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for random oversampling of the minority class by duplicating
existing samples.

# Keyword Arguments
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `random_oversample`](https://juliaai.github.io/Imbalance.jl/\
stable/oversamplers/random_oversample/)
"""
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

"""
    RandomWalkOversampler(; ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for random walk oversampling, which generates synthetic samples
by performing random walks between existing minority class samples.
Categorical columns are automatically detected when the matrix element
type is `Int` or `Union{Missing,Int}`.

# Keyword Arguments
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `random_walk_oversample`](https://juliaai.github.io/\
Imbalance.jl/stable/oversamplers/random_walk_oversample/)
"""
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
    cat_inds = X isa AbstractMatrix{<:Union{Int,Union{Missing,Int}}} ?
        collect(1:size(X, 2)) : Int[]
    random_walk_oversample(X, y, cat_inds; kwargs...)
end

"""
    ROSE(; s=1.0, ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for Random Over-Sampling Examples (ROSE), which generates
synthetic samples by smoothing the feature space around existing
minority class samples using a Gaussian kernel.

# Keyword Arguments
- `s::AbstractFloat=1.0`: Bandwidth of the Gaussian kernel used to
  generate synthetic samples.
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `rose`](https://juliaai.github.io/Imbalance.jl/\
stable/oversamplers/rose/)
"""
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

"""
    SMOTE(; k=5, ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for Synthetic Minority Over-sampling TEchnique (SMOTE), which
generates synthetic samples by interpolating between existing minority
class samples and their k-nearest neighbours.

# Keyword Arguments
- `k::Int=5`: Number of nearest neighbours to consider when generating
  synthetic samples.
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `smote`](https://juliaai.github.io/Imbalance.jl/\
stable/oversamplers/smote/)
"""
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

"""
    BorderlineSMOTE1(; m=5, k=5, ratios=1.0, rng=42,
                       try_preserve_type=true, verbosity=0)

Wrapper for Borderline-SMOTE1, a variant of SMOTE that focuses
synthetic sample generation on minority class samples near the
decision boundary (the "borderline" region).

# Keyword Arguments
- `m::Int=5`: Number of nearest neighbours used to determine whether
  a sample is borderline.
- `k::Int=5`: Number of nearest neighbours used to generate synthetic
  samples.
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.
- `verbosity::Int=0`: Verbosity level (0 = silent).

# See Also
- [Imbalance.jl `borderline_smote1`](https://juliaai.github.io/\
Imbalance.jl/stable/oversamplers/borderline_smote/)
"""
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

"""
    SMOTEN(; k=5, ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for SMOTE for Nominal features (SMOTEN), which generates
synthetic samples for datasets with exclusively categorical (integer-
encoded) features. Raises an error if applied to non-integer matrices;
consider [`SMOTE`](@ref) or [`SMOTENC`](@ref) for mixed data.

# Keyword Arguments
- `k::Int=5`: Number of nearest neighbours to consider when generating
  synthetic samples.
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `smoten`](https://juliaai.github.io/Imbalance.jl/\
stable/oversamplers/smoten/)
"""
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
    return X isa AbstractMatrix{<:Union{Int,Union{Missing,Int}}} ?
        smoten(X, y; kwargs...) :
        error("SMOTEN works only on discrete datasets, consider using " *
        "SMOTE or SMOTENC instead.")
end

"""
    SMOTENC(; k=5, ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for SMOTE for Nominal and Continuous features (SMOTENC), which
handles datasets with mixed feature types. Automatically dispatches to
[`smoten`] for integer matrices or [`smote`] for continuous matrices.

# Keyword Arguments
- `k::Int=5`: Number of nearest neighbours to consider when generating
  synthetic samples.
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `smotenc`](https://juliaai.github.io/Imbalance.jl/\
stable/oversamplers/smotenc/)
"""
struct SMOTENC{T} <: AbstractBalance
    balance::Base.Callable
    k::Int
    ratios::T
    rng::Int
    try_preserve_type::Bool

    function SMOTENC(;
        k::Int=5,
        ratios::T=1.0,
        rng::Int=42,
        try_preserve_type::Bool=true
    ) where {T<:Union{Real,Dict}}
        new{T}(
            _smotenc,
            k,
            ratios,
            rng,
            try_preserve_type
        )
    end
end

function _smotenc(X, y; kwargs...)
    if X isa AbstractMatrix{<:Union{Int,Union{Missing,Int}}}
        smoten(X, y; kwargs...)
    else
        smote(X, y; kwargs...)
    end
end

# ---------------------------------------------------------------------------- #
#                      undersampling imbalance structs                         #
# ---------------------------------------------------------------------------- #
"""
    RandomUndersampler(; ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for random undersampling of the majority class by randomly
removing existing samples.

# Keyword Arguments
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `random_undersample`](https://juliaai.github.io/\
Imbalance.jl/stable/undersamplers/random_undersample/)
"""
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

"""
    ClusterUndersampler(; mode="nearest", maxiter=100, ratios=1.0, rng=42)

Wrapper for cluster-based undersampling, which reduces the majority
class by clustering samples and retaining representative points.

# Keyword Arguments
- `mode::AbstractString="nearest"`: Strategy for selecting the
  representative sample from each cluster. Options: `"nearest"`,
  `"center"`, `"random"`.
- `maxiter::Int=100`: Maximum number of iterations for the clustering
  algorithm.
- `ratios::Union{Real,Dict}=1.0`: Desired ratio of minority to majority
  class. A `Real` applies the same ratio to all classes; a `Dict` maps
  class labels to target ratios.
- `rng::Int=42`: Random seed for reproducibility.

# See Also
- [Imbalance.jl `cluster_undersample`](https://juliaai.github.io/\
Imbalance.jl/stable/undersamplers/cluster_undersample/)
"""
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

"""
    ENNUndersampler(; k=5, keep_condition="mode", force_min_ratios=false,
                      min_ratios=1.0, rng=42, try_preserve_type=true)

Wrapper for Edited Nearest Neighbours (ENN) undersampling, which
removes majority class samples that are misclassified by their k
nearest neighbours.

# Keyword Arguments
- `k::Int=5`: Number of nearest neighbours used to evaluate each
  sample.
- `keep_condition::AbstractString="mode"`: Condition to retain a
  sample. Options: `"mode"`, `"all"`, `"only_min"`, `"foreign"`.
- `force_min_ratios::Bool=false`: If `true`, enforce `min_ratios`
  even if the majority class is already below the threshold.
- `min_ratios::Union{Real,Dict}=1.0`: Minimum allowed ratio of
  minority to majority class after undersampling.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `enn_undersample`](https://juliaai.github.io/\
Imbalance.jl/stable/undersamplers/enn_undersample/)
"""
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

"""
    TomekUndersampler(; force_min_ratios=false, min_ratios=1.0,
                        rng=42, try_preserve_type=true)

Wrapper for Tomek Links undersampling, which removes majority class
samples that form Tomek links (pairs of samples from different classes
that are each other's nearest neighbour).

# Keyword Arguments
- `force_min_ratios::Bool=false`: If `true`, enforce `min_ratios`
  even if the majority class is already below the threshold.
- `min_ratios::Union{Real,Dict}=1.0`: Minimum allowed ratio of
  minority to majority class after undersampling.
- `rng::Int=42`: Random seed for reproducibility.
- `try_preserve_type::Bool=true`: Attempt to preserve the original
  element type after resampling.

# See Also
- [Imbalance.jl `tomek_undersample`](https://juliaai.github.io/\
Imbalance.jl/stable/undersamplers/tomek_undersample/)
"""
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
"""
    _get_balfunc(b::AbstractBalance) -> Base.Callable

Returns the underlying Imbalance.jl resampling function stored in `b`.
"""
_get_balfunc(b::AbstractBalance) = b.balance

"""
    _get_balkw(b::AbstractBalance) -> NamedTuple

Returns all fields of `b` except `balance` as a `NamedTuple`, to be
splatted as keyword arguments into the resampling function.
"""
function _get_balkw(b::AbstractBalance)
    names = Tuple(filter(!=(:balance), fieldnames(typeof(b))))
    vals = Tuple(getfield(b, n) for n in names)
    return NamedTuple{names}(vals)
end

