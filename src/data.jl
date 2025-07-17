struct LongTailsDataSet
    K::Matrix{Int}
    effLengths::Matrix
end

function geomeanZeros(x)
    all(x .== 0) ? 0.0 : exp(mean(log.(x[x .> 0])) / length(x))
end

function normalize(data::LongTailsDataSet)

    L = data.effLengths
    L ./= exp.(mean(log.(L);dims=1))

    geomeanz = map(geomeanZeros, eachcol(K))

    log_sj = median(log(len_normed) .- mean(log.(len_normed);dims=1);dims=2)
    sj = exp.(log_sj)

    sj = len_normed * sj 

    sj / exp.(mean(log.(sj);dims=1))

end

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
