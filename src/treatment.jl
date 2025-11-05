core_eltype(x) = eltype(x) <: AbstractArray ? core_eltype(eltype(x)) : eltype(x)

# ---------------------------------------------------------------------------- #
#                            reducesize functions                              #
# ---------------------------------------------------------------------------- #
function applyfeat(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)::AbstractArray
    reduced = similar(X, length.(intervals)...)

    @inbounds map!(reduced, CartesianIndices(reduced)) do cart_idx
        ranges = ntuple(i -> intervals[i][cart_idx[i]], length(intervals))
        reducefunc(@view X[ranges...])
    end
end

# function reducesize(
#     X::AbstractArray,
#     intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
#     reducefunc :: Base.Callable=mean   
# )::AbstractArray
#     Xresult = similar(X)

#     Threads.@threads for i in eachindex(X)
#         @inbounds Xresult[i] = applyfeat(X[i], intervals; reducefunc)
#     end
#     return Xresult
# end

# ---------------------------------------------------------------------------- #
#                             aggregate functions                              #
# ---------------------------------------------------------------------------- #
# Internal function that computes aggregate on a single array element
# function _aggregate(
#     X        :: AbstractArray;
#     features :: Tuple{Vararg{Base.Callable}}=(mean,)
# )::AbstractArray
#     @views @inbounds collect(f(X) for f in features)
# end

function aggregate(
    X          :: AbstractArray,
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    reducefunc :: Base.Callable=mean
)
    Xresult = similar(X)
    Threads.@threads for i in eachindex(X)
        @inbounds Xresult[i] = applyfeat(X[i], intervals; reducefunc)
    end
    return Xresult
end

function reducesize(
    X         :: AbstractArray,
    intervals :: Tuple{Vararg{Vector{UnitRange{Int64}}}};
    features  :: Tuple{Vararg{Base.Callable}}=(mean,),
)
    nwindows = prod(length.(intervals))
    nfeats = nwindows * length(features)
    Xresult = Array{core_eltype(Xm)}(undef, size(X, 1), size(X, 2) * nfeats)
    
    @inbounds Threads.@threads for colidx in axes(X, 2)
        for rowidx in axes(X,1)
            reduced = mapreduce(vcat, features) do feat
                vec(applyfeat(X[rowidx,colidx], intervals; reducefunc=feat))
            end
            
            base_idx = (colidx - 1) * nfeats
            @inbounds copyto!(view(Xresult, rowidx, base_idx+1:base_idx+nfeats), vec(reduced))
        end
    end
    return Xresult
end

