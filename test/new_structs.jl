using DataFrames

df = DataFrame(
    vec_col  = [rand(Float64, 5), missing, rand(Float64, 5), rand(Float64, 5), missing],
    str_col  = ["hello", "world", missing, "foo", "bar"],
    int_col  = [1, missing, 3, 4, missing],
    float_col = [1.1, 2.2, missing, 4.4, missing]
)

abstract type AbstractFeatureId end
abstract type AbstractMultiDimFeatId <: AbstractFeatureId end

struct FeatureId_T{T} <: AbstractFeatureId
    vname::Symbol

    FeatureId_T(T::Type, vname::Symbol) = new{T}(vname)
end

struct FeatureId_MT{T} <: AbstractMultiDimFeatId
    vname::Symbol
    feat::Base.Callable
    nwin::Int64
    size::Tuple{Vararg{Int64}}

    function FeatureId_MT(
        T::Type,
        vname::Symbol,
        feat::Base.Callable,
        nwin::Int64,
        size::Tuple{Vararg{Int64}})
        new{T}(vname, feat, nwin, size)
    end
end

struct FeatureId_RD{T} <: AbstractMultiDimFeatId
    vname::Symbol
    nwin::Int64
    size::Tuple{Vararg{Int64}}

    function FeatureId_RD(
        T::Type,
        vname::Symbol,
        nwin::Int64,
        size::Tuple{Vararg{Int64}})
        new{T}(vname, nwin, size)
    end
end

function FeatureId(::Union{Missing, AbstractArray{T}}, vname::Symbol) where {T}
    FeatureId_T(nonmissingtype(T), vname)
end

function FeatureId(
    x::AbstractArray{<:Union{Missing, AbstractArray{T}}},
    vname::Symbol,
    feat::Base.Callable,
    nwin::Int64
) where {T}
    FeatureId_MT(nonmissingtype(T), vname, feat, nwin, size(x))
end

function FeatureId(
    x::AbstractArray{<:Union{Missing, AbstractArray{T}}},
    vname::Symbol,
    nwin::Int64
) where {T}
    FeatureId_RD(nonmissingtype(T), vname, nwin, size(x))
end

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
# value access methods
Base.getproperty(f::Union{FeatureId_T,FeatureId_MT,FeatureId_T}, s::Symbol) = getfield(f, s)
Base.propertynames(::FeatureId_T) = (:vname)
Base.propertynames(::FeatureId_MT) = (:vname, :feat, :nwin, :size)
Base.propertynames(::FeatureId_T) = (:vname, :nwin, :size)

get_vname(f::AbstractFeatureId) = f.vname
get_feat(f::FeatureId_MT) = f.feat
get_nwin(f::AbstractMultiDimFeatId) = f.nwin
get_size(f::AbstractMultiDimFeatId) = f.size

# ---------------------------------------------------------------------------- #
#                               Base.show                                      #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, f::FeatureId_T{T}) where {T}
    print(io, "FeatureId_T{$(T)}(vname=$(f.vname))")
end

function Base.show(io::IO, f::FeatureId_MT{T}) where {T}
    feat_name = nameof(f.feat)
    print(io, "FeatureId_MT{$(T)}(vname=$(f.vname), feat=$(feat_name), nwin=$(f.nwin), size=$(f.size))")
end

function Base.show(io::IO, f::FeatureId_RD{T}) where {T}
    print(io, "FeatureId_RD{$(T)}(vname=$(f.vname), nwin=$(f.nwin), size=$(f.size))")
end

function Base.show(io::IO, ::MIME"text/plain", f::AbstractFeatureId)
    print(io, "$(typeof(f)): ")
    show(io, f)
end
