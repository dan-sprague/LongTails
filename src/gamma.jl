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


function gamma_irls_identity(X, y; max_iter=100, tol=1e-6)
    n, p = size(X)
    β = randn(p)
    
    @inbounds for iter in 1:max_iter
        η = X * β
        μ = η
        μ = max.(μ, 1e-6)  # Ensure μ > 0
        
        W = diagm(1 ./ (μ .^ 2))
        z = η + (y - μ)
        
        β_new = (X' * W * X) \ (X' * W * z)
        
        if norm(β_new - β) < tol
            return β_new
        end
        β = β_new
    end
    
    return β
end


X = [ones(length(mu)) 1 ./ mu]

@time gamma_irls_identity(X, simulation.parameters.α[mask]; max_iter=100, tol=1e-4)


