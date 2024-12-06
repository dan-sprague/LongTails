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
    _logpdf(d::PowerLaw,x)

The log probability density function of a PowerLaw distribution.

"""
function _logpdf(d::PowerLaw,x)
    ((d.α - 1) / d.x_min) + (-d.α  * log(x / d.x_min))
end


"""
    logpdf(d::PowerLaw,x)

The log probability density function of a PowerLaw distribution, over a vector or scalar x.
"""
function logpdf(d::PowerLaw,x)
    sum(@. _logpdf(d,x))
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

struct Transcriptome
    c::Vector{Float64}
    σd::Float64
    K::Int
end

function Transcriptome(d::PowerLaw,σd,K)
    Transcriptome(rand(d,20_000),σd,K)
end


function αtr_sample(x,σ)
    exp(rand(Normal(log(x),σ)))
end

function Base.rand(rng::AbstractRNG,d::Transcriptome)

    α = @. αtr_sample(atr_sim(d.c),d.σd)
    θ = nbreg_transform.(d.c,α)

    reduce(hcat,map(θ -> rand(NegativeBinomial(θ...),d.K),θ))

end


function diffsim(X::Matrix;perc_expanding=0.05)
    expanding = rand(Bernoulli(perc_expanding),lastindex(X))
    FC = rand(Gamma(gamma_reg_transform(2,.005)...),size(X,1)) .* expanding
    FC[FC .== 0] .= 1
    FC = map(x -> rand(Bernoulli(0.5)) == 1 ? x : 1 / x,FC)
    rand(Transcriptome(X .* FC,1.5,3))
end