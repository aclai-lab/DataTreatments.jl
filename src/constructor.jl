# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# base type for metadata containers
abstract type AbstractFeatureId end
abstract type AbstractDataTreatment end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const ValidVnames = Union{Symbol, String}

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
is_multidim_dataframe(X::AbstractArray)::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

# ---------------------------------------------------------------------------- #
#                                  FeatureId                                   #
# ---------------------------------------------------------------------------- #
struct FeatureId{T<:ValidVnames} <: AbstractFeatureId
    vname :: Symbol
    feat  :: Base.Callable
    nwin  :: Int64

    function FeatureId(var::ValidVnames, feat::Base.Callable, nwin::Int64)
        new{typeof(var)}(var, feat, nwin)
    end
end

# value access methods
Base.getproperty(f::FeatureId, s::Symbol) = getfield(f, s)
Base.propertynames(::FeatureId)           = (:vname, :feat, :nwin)

get_vname(f::FeatureId)   = f.vname
get_feature(f::FeatureId) = f.feat
get_nwin(f::FeatureId)    = f.nwin

function Base.show(io::IO, f::FeatureId)
    feat_name = nameof(f.feat)
    if f.nwin == 1
        print(io, "$(feat_name)($(f.vname))")
    else
        print(io, "$(feat_name)($(f.vname))_w$(f.nwin)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", f::FeatureId)
    print(io, "FeatureId: ")
    show(io, f)
end

# ---------------------------------------------------------------------------- #
#                                 constructor                                  #
# ---------------------------------------------------------------------------- #
struct DataTreatment{T, S} <: AbstractDataTreatment
    dataset    :: AbstractMatrix{T}
    featureid  :: Vector{FeatureId}
    reducefunc :: Base.Callable
    aggrtype   :: Symbol

    function DataTreatment(
        X          :: AbstractMatrix,
        aggrtype   :: Symbol;
        vnames     :: Vector{<:ValidVnames},
        win        :: Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
        features   :: Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
        reducefunc :: Base.Callable=mean,
        norm       :: Bool=false
    )
        is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
            "does not contain multidimensional data."))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        win isa Base.Callable && (win = (win,))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        intervals = @evalwindow first(X) win...
        nwindows = prod(length.(intervals))

        Xresult, Xinfo = if aggrtype == :aggregate begin
            (aggregate(X, intervals; features),
            if nwindows == 1
                # single window: apply to whole time series
                [FeatureId(Symbol("$(f)($(v))"), f, 1)
                    for f in features, v in vnames] |> vec
            else
                # multiple windows: apply to each interval
                [FeatureId(Symbol("$(f)($(v))_w$(i)"), f, i)
                    for i in 1:nwindows, f in features, v in vnames] |> vec
            end
            )
        end

        elseif aggrtype == :reducesize begin
            (reducesize(X, intervals; reducefunc),
            [FeatureId(Symbol(v), reducefunc, 1)
                for v in vnames] |> vec
            )
        end

        else
            error("Unknown treatment type: $treat")
        end

        new{eltype(Xresult), core_eltype(Xresult)}(Xresult, Xinfo, reducefunc, aggrtype)
    end

    function DataTreatment(
        X      :: AbstractDataFrame,
        args...;
        vnames :: Union{Vector{<:ValidVnames}, Nothing}=nothing,
        kwargs...
    )
        isnothing(vnames) && (vnames = propertynames(X))
        DataTreatment(Matrix(X), args...; vnames, kwargs...)
    end
end

# value access methods
Base.getproperty(dt::DataTreatment, s::Symbol) = getfield(dt, s)
Base.propertynames(::DataTreatment) = (:dataset, :featureid, :reducefunc, :aggrtype)

get_dataset(dt::DataTreatment)    = dt.dataset
get_featureid(dt::DataTreatment)  = dt.featureid
get_reducefunc(dt::DataTreatment) = dt.reducefunc
get_aggrtype(dt::DataTreatment)   = dt.aggrtype

# Convenience methods for common operations
get_vnames(dt::DataTreatment)   = unique([get_vname(fid)   for fid in dt.featureid])
get_features(dt::DataTreatment) = unique([get_feature(fid) for fid in dt.featureid])
get_nwindows(dt::DataTreatment) = maximum([get_nwin(fid)   for fid in dt.featureid])

# Size and iteration methods
Base.size(dt::DataTreatment)   = size(dt.dataset)
Base.size(dt::DataTreatment, dim::Int) = size(dt.dataset, dim)
Base.length(dt::DataTreatment) = length(dt.featureid)
Base.eltype(dt::DataTreatment) = eltype(dt.dataset)

# Indexing methods
Base.getindex(dt::DataTreatment, i::Int) = dt.dataset[:, i]
Base.getindex(dt::DataTreatment, i::Int, j::Int) = dt.dataset[i, j]
Base.getindex(dt::DataTreatment, ::Colon, j::Int) = dt.dataset[:, j]
Base.getindex(dt::DataTreatment, i::Int, ::Colon) = dt.dataset[i, :]
Base.getindex(dt::DataTreatment, I...) = dt.dataset[I...]

function Base.show(io::IO, dt::DataTreatment)
    nrows, ncols = size(dt.dataset)
    print(io, "DataTreatment($(dt.aggrtype), $(nrows)×$(ncols), $(length(dt.featureid)) features)")
end

function Base.show(io::IO, ::MIME"text/plain", dt::DataTreatment)
    nrows, ncols = size(dt.dataset)
    nfeatures = length(dt.featureid)
    
    println(io, "DataTreatment:")
    println(io, "  Type: $(dt.aggrtype)")
    println(io, "  Dimensions: $(nrows)×$(ncols)")
    println(io, "  Features: $(nfeatures)")
    println(io, "  Reduction function: $(nameof(dt.reducefunc))")
    
    if nfeatures <= 10
        println(io, "  Feature IDs:")
        for (i, fid) in enumerate(dt.featureid)
            println(io, "    $(i). ", fid)
        end
    else
        println(io, "  Feature IDs (showing first 5 and last 5):")
        for i in 1:5
            println(io, "    $(i). ", dt.featureid[i])
        end
        println(io, "    ⋮")
        for i in (nfeatures-4):nfeatures
            println(io, "    $(i). ", dt.featureid[i])
        end
    end
end