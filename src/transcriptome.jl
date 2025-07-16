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
    d::PowerLaw
    σd::Float64
    X::Matrix
    n::Int
end

function DifferentialTranscriptome(d::PowerLaw,σd,X,n)
    DifferentialTranscriptome(d,σd,X,n)
end


 

function αtr_sample(μ̄,σ)
    exp(rand(Normal(log((0.5 / μ̄) + 0.025),σ)))
end


function Base.rand(rng::AbstractRNG,T::DifferentialTranscriptome)


    designMatrix = T.X
    d = T.d
    σd = T.σd
    n_cov = size(designMatrix,2) - 1
    K = zeros(Int,size(designMatrix,1),T.n)
    β = rand(Normal(),n_cov,T.n)
    baseMean = rand(d,T.n)

    sizeFactors = sim_library_size(size(designMatrix,1))

    for i in 1:T.n

        log_counts = log.(sizeFactors) .+ designMatrix * vcat(log(baseMean[i]),β[:,i])
        counts = exp.(log_counts)
        α = αtr_sample(mean(counts), σd)

        K[:,i] .= rand.(NegBin2.(counts,Ref(α)))

    end

    (K,vcat(baseMean',β),sizeFactors)
end


  function sim_library_size(n; cv=0.3)
      σ = sqrt(log(cv^2 + 1))  # Convert CV to log-normal σ

      factors = exp.(rand(Normal(0, σ), n))
      factors ./ exp(mean(log.(factors))) 
  end
