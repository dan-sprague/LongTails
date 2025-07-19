using SpecialFunctions


struct NegBin2
    μ::Float64
    α::Float64
end


Base.Broadcast.broadcastable(n::NegBin2) = Ref(n)

function logpmf(n::NegBin2,x::Int)
    @assert x >= 0 "x must be greater than or equal to 0"
    ϕ = 1 / n.α
    loggamma(ϕ + x) - loggamma(ϕ) - loggamma(x + 1) + (x * log(n.μ)) + (ϕ * log(ϕ)) - ((x + ϕ) * log(ϕ + n.μ)) 
end


function Base.rand(rng::AbstractRNG, d::NegBin2)
    # Gamma(shape, scale) where scale = 1/rate
    shape = 1/d.α
    scale = d.μ * d.α  # This is 1/rate, so rate = 1/(μ*α)
    λ = rand(rng, Gamma(shape, scale))
    rand(rng, Poisson(λ))
end




"""
    nb_alpha_cr_nll(x,d,μ̂,μ̄,α)

Negative binomial regression negative log likelihood function with Cox-Reid regularization.
"""
function nb_alpha_cr_nll(y,X,μ̂,log_α)
    μj = X * μ̂
    α = exp(log_α[1])
    d = NegBin2(mean(μj),α)
    W = diagm(@. 1 / ((1 / μj) + α))
    cr = 0.5 * ∑(log.(diag(cholesky(X' * W * X))))

    -(∑(logpmf.(d,y)) + cr)

end

function irls(data::NormalizedLongTailsDataSet,β,σr)
    μ = exp.(model.X * β)
    λr  = 1 / (σr^2) 
    z = log(μ / model.sj) + ((model.y .- μ) ./ μ)

    W = Diagonal(@. μ / (1 + (α * μ)))

    (transpose(model.X) * W * model.X + λr * I) \ (transpose(model.X) * W * z)
end


"""
    nbreg_transform(μ,α)

Takes the mean and dispersion parameter of a negative binomial distribution and returns the shape and scale parameters of the corresponding gamma distribution.
"""
function nbreg_transform(μ,α)
    σ = μ + (α * μ^2)
    r = μ^2 / (σ - μ)
    p = μ / σ

    r,p
end
