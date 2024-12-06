using Distributions
using Optim 
using StatsBase
using LinearAlgebra

∑ = sum 

"""
    nbreg_nll(x,μ,α)

Negative binomial regression negative log likelihood function.
"""
function nbreg_nll(x,μ,α)


    μ = exp(μ[1])
    α = exp(α[1])
    
    r,p = nbreg_transform(μ,α)

    ∑(-logpdf(NegativeBinomial(r,p),x))


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


"""
    gamma_reg_transform(μ,ϕ)

Takes the mean and dispersion parameter of a gamma distribution and returns the shape and scale parameters of the corresponding gamma distribution.
"""
function gamma_reg_transform(μ,ϕ)
    k = 1 / ϕ
    θ = @. μ * ϕ
    k,θ
end

atr(x,a1,a0) = (a1 / x)  + a0

"""
    normalize!(X)

Normalizes the columns of a matrix by dividing each column by the geometric mean of the row. Equivalent to DESeq2's median-of-ratios normalization.
"""
function normalize!(X;thr = 1)
    sj = map(median,eachcol(X ./ map(harmmean,eachrow(X))))
    X ./= sj'   
end
    
"""
    atr_sim(x;a1=1.0,a0=0.01)

Simulates dispersion trend prior for negative binomial regression.
"""
atr_sim(x;a1=1.0,a0=0.01) = (a1 / x) + a0;