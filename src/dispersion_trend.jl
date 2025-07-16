function fit_dispersion_trend(α_init::Vector{Float64}, μ̄::Vector{Float64}; 
    max_iter::Int=100, 
    tol::Float64=1e-6,
    init_params::Vector{Float64}=[0.0, 0.0, 0.0])
    
    # Input validation
    length(α_init) == length(μ̄) || throw(DimensionMismatch("α_init and μ̄ must have same length"))
    
    # Initialize parameters
    params_old = copy(init_params)
    params_current = copy(init_params)
    idx = trues(length(α_init))
    SS_LFC = Inf
    iter = 0
    
    # Optimization settings
    opt_settings = Optim.Options(iterations=1000)
    
    while SS_LFC > tol && iter < max_iter
        # Create views of current valid data
        α_subset = @view α_init[idx]
        μ_subset = @view μ̄[idx]
        
        # Fit gamma GLM
        res = optimize(
            θ -> gamma_trend_nll(α_subset, μ_subset, θ...),
            params_current,
            NelderMead(),
            opt_settings
        )
        
        params_current = Optim.minimizer(res)
        
        # Calculate fitted values and ratios
        α_fit = atr.(μ̄, exp(params_current[1]), exp(params_current[2]))
        α_ratio = α_fit ./ α_init
        
        # Update valid indices
        idx = @. (1e-4 <= α_ratio <= 15) * idx
        
        # Check convergence
        SS_LFC = sum(@. (params_current[[1,2]] - params_old[[1,2]])^2)
        
        params_old = copy(params_current)
        iter += 1
    end
    
    # Warn if not converged
    if iter == max_iter
        @warn "Reached maximum iterations ($max_iter) without convergence"
    end
    
    return (
        parameters=params_current,
        iterations=iter,
        converged=SS_LFC ≤ tol,
        final_indices=idx
    )
end



atr(x,a1,a0) = (a1 / x)  + a0
    
"""
    atr_sim(x;a1=1.0,a0=0.01)

Simulates dispersion trend prior for negative binomial regression.
"""
atr_sim(x;a1=1.0,a0=0.01) = (a1 / x) + a0;