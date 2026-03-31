function _impute(data::AbstractArray, impute::Tuple{Vararg{<:Impute.Imputor}})
    Impute.declaremissings(data; values=(NaN, "NULL"))

    for im in impute
        Impute.impute!(data, im; dims=2)
    end
    any(ismissing.(data)) || (data = disallowmissing(data))

    return data
end

function _impute(
    data::AbstractMatrix{T},
    impute::Tuple{Vararg{<:Impute.Imputor}}
) where {T<:Union{Missing,Float,AbstractArray{<:Float}}}
    Impute.declaremissings(data; values=(NaN, "NULL"))

    for im in impute
        Impute.impute!(data, im)
    end

    any(ismissing.(data)) || (data = disallowmissing(data))

    return data
end

function _impute(
    data::T,
    impute::Tuple{Vararg{<:Impute.Imputor}}
) where {T<:Union{Missing,Float,AbstractArray{<:Float}}}   
    if !ismissing(data) && typeof(data) <: AbstractArray
        Impute.declaremissings(data; values=(NaN, "NULL"))
        for im in impute
            Impute.impute!(data, im)
        end

        any(ismissing.(data)) || (data = disallowmissing(data))
    end

    return data
end
