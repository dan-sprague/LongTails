struct LongTailsDataSet
    K::Matrix{Int}
    effLengths::Matrix
end


function logGeoMeanZeros(x)
    all(x .== 0) ? 0.0 : sum(log.(x[x .> 0])) / length(x)
end


"""
    scalingFactors(data::LongTailsDataSet)

    Scales the counts in `data.K` by the effective lengths in `data.effLengths`
    and for sequencing depth differences across samples.

    Returns a matrix with scaling offsets for each gene and sample.

"""
function scalingFactors(data::LongTailsDataSet)

    L = data.effLengths
    L ./= exp.(mean(log.(L);dims=1))

    effLengthNormalizedCounts = data.K ./ L 

    ### Correcting what I believe is a bug in the original code
    ### The original code passed normalized counts to estimateSizeFactorsForMatrix
    ### for the counts arg, but the geomeans used for normalization where
    ### calculated from the original counts, not the normalized counts.
    logGeoMeans = map(logGeoMeanZeros, eachcol(effLengthNormalizedCounts))
    mask = @. effLengthNormalizedCounts > 0 & !isinf(logGeoMeans)'

    sj = zeros(Float64,size(data.K,1))

    for i in axes(data.K,1)
        z = log.(effLengthNormalizedCounts[i,:]) .- logGeoMeans
        sj[i] = z[mask[i,:]] |> median |> exp
    end 


    sj ./ geomean(sj)

    nf = L .* sj

    nf ./ exp.(mean(log.(nf);dims=1))

end


normalize(data)

function convertToFactor!(metadata::DataFrame)
        
    for factor in propertynames(metadata)
        if !(eltype(metadata[!,factor]) <: Number)
            metadata[!,factor] = categorical(metadata[!,factor])
        end
    end
end 

K = simulation.counts 
L ./= exp.(mean(log.(L);dims=1))
len_normed = K ./ L 

log.(len_normed) .- mean(log.(len_normed);dims=1)


function buildExtendedDesignMatrix(metadata::DataFrame)
    extended_metadata = vcat(metadata, metadata[end:end, :])

    for factor in propertynames(extended_metadata)
        if !(eltype(extended_metadata[!,factor]) <: Number)
            extended_metadata[end,factor] = "_"
            levels!(extended_metadata[!,factor],vcat("_",levels(metadata[!,factor])...))
        end
    end

    extended_metadata
end
