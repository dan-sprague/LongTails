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

function nb_alpha_cr_nll(x,d,μ̂,μ̄,α)
    nb_nll = nbreg_nll(x,log(μ̄),α)
    w = diagm(1 ./ ((1 ./ (d * μ̂)) .+ exp.(α)))

    cr = 0.5 * log(det(d' * w * d))
    nb_nll + cr

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
    atr_sim(x;a1=1.0,a0=0.01)

Simulates dispersion trend prior for negative binomial regression.
"""
atr_sim(x;a1=1.0,a0=0.01) = (a1 / x) + a0;