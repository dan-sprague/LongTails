  """
    gamma_reg_transform(μ,ϕ)

Takes the mean and dispersion parameter of a gamma distribution and returns the shape and scale parameters of the corresponding gamma distribution.
"""
function gamma_reg_transform(μ,ϕ)
    k = 1 / ϕ
    θ = @. μ * ϕ
    k,θ
end


function gamma_trend_nll(α,x,a1,a0,ϕ)
    
    a1 = exp(a1)
    a0 = exp(a0)
    ϕ = exp(ϕ)
    μ = atr.(x,a1,a0)

    k,θ = gamma_reg_transform(μ,ϕ)
    sum(@. -logpdf(Gamma(k,θ),α))
end
