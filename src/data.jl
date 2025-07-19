struct Design
    matrix::Matrix{Float64}
    expandedMatrix::Matrix{Float64}
    formula::Formula
end

function Design(formula::Formula, metadata::DataFrame)
    metadata = convertToFactor!(metadata)
    designMatrix = modelmatrix(formula, metadata)
    extendedDesignMatrix = buildExtendedDesignMatrix(metadata)

    Design(designMatrix, extendedDesignMatrix, formula)
end


struct LongTailsDataSet
    K::Matrix{Int64}
    effLengths::Matrix{Float64}
    nf::Matrix{Float64}
    μ::Vector{Float64}
    σ²::Vector{Float64}

    design::Design
end


function LongTailsDataSet(K::Matrix{Int}, effLengths::Matrix{Float64},
    metadata::DataFrame, formula::Formula)
    nf = scalingFactors(K, effLengths)
    design = Design(formula, metadata)
    LongTailsDataSet(K,
        effLengths,
        nf,
        vec(mean(K; dims=1)),
        vec(var(K; dims=1)),
        design
    )
end


counts(d::LongTailsDataSet;norm=false) = norm ? d.K ./ d.nf : d.K
designMatrix(d::LongTailsDataSet;expanded=false) = expanded ? d.design.expandedMatrix : d.design.matrix
μ(d::LongTailsDataSet) = d.μ
σ²(d::LongTailsDataSet) = d.σ²
nf(d::LongTailsDataSet) = d.nf


"""
    logGeoMeanZeros(x)

    Computes the log geometric mean of a vector `x`, treating zeros as missing values.
    If all values in `x` are zero, returns 0.0.
"""
function logGeoMeanZeros(x)
    all(x .== 0) ? 0.0 : sum(log.(x[x .> 0])) / length(x)
end


"""
    scalingFactors(data::LongTailsDataSet)

    Scales the counts in `data.K` by the effective lengths in `data.effLengths`
    and for sequencing depth differences across samples.

    Returns a matrix with scaling offsets for each gene and sample.

"""
function scalingFactors(K::Matrix{Int}, effLengths::Matrix{Float64})

    L = effLengths
    L ./= exp.(mean(log.(L);dims=1))

    effLengthNormalizedCounts = K ./ L 

    ### Correcting what I believe is a bug in the original code
    logGeoMeans = map(logGeoMeanZeros, eachcol(effLengthNormalizedCounts))
    mask = @. effLengthNormalizedCounts > 0 & !isinf(logGeoMeans)'

    sj = zeros(Float64,size(K,1))

    for i in axes(K,1)
        z = log.(effLengthNormalizedCounts[i,:]) .- logGeoMeans
        sj[i] = z[mask[i,:]] |> median |> exp
    end 


    sj ./ geomean(sj)

    nf = L .* sj

    nf ./ exp.(mean(log.(nf);dims=1))

end


function simpleScalingFactors(counts)
    logGeoMeans = map(logGeoMeanZeros, eachcol(counts))
    z = (log.(counts) .- logGeoMeans')
    sj = map(x -> median(x[.!isinf.(x)]), eachrow(z)) .|> exp

    sj ./ geomean(sj)
end



"""
    convertToFactor!(metadata::DataFrame)

Converts non-numeric columns in the `metadata` DataFrame to categorical factors.
"""
function convertToFactor!(metadata::DataFrame)
        
    for factor in propertynames(metadata)
        if !(eltype(metadata[!,factor]) <: Number)
            metadata[!,factor] = categorical(metadata[!,factor])
        end
    end
end 


"""
    buildExtendedDesignMatrix(metadata::DataFrame)

    Builds an extended design matrix from the metadata DataFrame by duplicating the last row, as described in the original code and paper.

"""
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
