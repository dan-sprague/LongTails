using Random 

"""
    PowerLaw

A power law distribution with parameters α and x_min, and normalization constant C.

"""
struct PowerLaw
    α::Float64
    x_min::Int64
    C::Float64
end

"""
    PowerLaw(;α=2.0,x_min=1)

Constructs a PowerLaw distribution with parameters α and x_min, and normalization constant C.

"""
PowerLaw(;α=2.0,x_min=1) = PowerLaw(
    α,
    x_min,
    (α - 1) * x_min^(α - 1)
)


"""
    logpdf(d::PowerLaw,x)

The log probability density function of a PowerLaw distribution, over a vector or scalar x.
"""
function Distributions.logpdf(d::PowerLaw,x)
    sum(@. _logpdf(d,x))
end


"""
    _logpdf(d::PowerLaw,x)
The log probability density function of a PowerLaw distribution.
"""
function _logpdf(d::PowerLaw,x)
    ((d.α - 1) / d.x_min) + (-d.α  * log(x / d.x_min))
end




function Base.rand(rng::AbstractRNG,d::PowerLaw)
    U = rand(rng)
    X = ((d.C / ((d.α - 1) * U))^(1 / (d.α - 1)))

    X 
end

function Base.rand(rng::AbstractRNG,d::PowerLaw,n::Integer)
    [rand(rng,d) for i in 1:n]
end



function Base.rand(d::PowerLaw,n::Integer)
    rand(Random.default_rng(),d,n)
end

struct DifferentialTranscriptome
    distribution::PowerLaw
    design::Matrix
    αtr_σd::Float64
    αtr_a1::Float64
    αtr_a0::Float64
    avg_effective_length::Float64
    n_genes::Int
end

function αtr_sample(μ̄,t::DifferentialTranscriptome)
    exp(rand(Normal(log((t.αtr_a1 / μ̄) + t.αtr_a0),t.αtr_σd)))
end


function Base.rand(rng::AbstractRNG,T::DifferentialTranscriptome)


    designMatrix = T.design
    d = T.distribution
    αtr_σd = T.αtr_σd
    n_cov = size(designMatrix,2) - 1
    K = zeros(Int,size(designMatrix,1),T.n_genes)
    β = rand(Normal(),n_cov,T.n_genes) .* rand(Bernoulli(0.1),T.n_genes)'
    α = zeros(Float64,T.n_genes)
    baseMean = rand(d,T.n_genes)

    sizeFactors = sim_library_size(size(designMatrix,1))
    logSizeFactors = log.(sizeFactors)
    for i in 1:T.n_genes
        offset = logSizeFactors
        log_counts = offset .+ designMatrix * vcat(log(baseMean[i]),β[:,i])
        counts = exp.(log_counts)
        α[i] = αtr_sample(mean(counts), T)

        K[:,i] .= rand.(NegBin2.(counts,Ref(α[i])))

    end

    (counts = K,parameters = (β = vcat(baseMean',β),sj = sizeFactors,α = α),
     design = designMatrix, distribution = d, αtr_σd = αtr_σd,
     avg_effective_length = T.avg_effective_length)
end


function sim_library_size(n; cv=0.3)
    σ = sqrt(log(cv^2 + 1))  # Convert CV to log-normal σ

    factors = exp.(rand(Normal(0, σ), n))
    factors ./ exp(mean(log.(factors))) 
end


function sim_effective_lengths(n_genes::Int)

    lengths = exp.(rand(Normal(log(2000), 0.8), n_genes))
    clamp.(lengths, 200, 100000)
end