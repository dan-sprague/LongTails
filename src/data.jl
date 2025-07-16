struct LongTailsDataSet
    X::Matrix{Number}
    K::Matrix{Int}
    sj::Vector{Float64}
    norm_counts::Matrix{Float64}
end

function LongTailsDataSet(X::Matrix{T}, K::Matrix{Int}) where T <: Number
    sj = factors(K)
    norm_counts = K ./ sj
    return LongTailsDataSet(X, K,sj, norm_counts)
end

function convertToFactor!(metadata::DataFrame)
        
    for factor in propertynames(metadata)
        if !(eltype(metadata[!,factor]) <: Number)
            metadata[!,factor] = categorical(metadata[!,factor])
        end
    end
end 


function buildExtendedDesignMatrix(metadata)
    extended_metadata = vcat(metadata, metadata[end:end, :])

    for factor in propertynames(extended_metadata)
        if !(eltype(extended_metadata[!,factor]) <: Number)
            extended_metadata[end,factor] = "_"
            levels!(extended_metadata[!,factor],vcat("_",levels(metadata[!,factor])...))
        end
    end

    extended_metadata
end
