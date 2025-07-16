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
    nb_alpha_cr_nll(x,d,μ̂,μ̄,α)

Negative binomial regression negative log likelihood function with Cox-Reid regularization.
"""
function nb_alpha_cr_nll(x,d,μ̂,μ̄,α)
    nb_nll = nbreg_nll(x,log(μ̄),α)
    w = diagm(1 ./ ((1 ./ (d * μ̂)) .+ exp.(α)))

    cr = 0.5 * log(det(d' * w * d))
    nb_nll + cr

end