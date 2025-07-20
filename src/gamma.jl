function gamma_irls_identity(X, y; max_iter=100, tol=1e-6, filter_outliers=true)
    n, p = size(X)
    β = [0.1, 1.0] 
    
    good_idx = trues(n)
    
    for _ in 1:max_iter

        X_sub = @view X[good_idx, :]
        y_sub = @view y[good_idx]
        
        η = X_sub * β
        μ = max.(η, 1e-6)  
        
        w = 1 ./ (μ .^ 2)  
        z = η + (y_sub - μ)
        

        XtWX = X_sub' * (w .* X_sub)
        XtWz = X_sub' * (w .* z)
        
        β_new = XtWX \ XtWz
        
        # Check for positive coefficients
        if !all(β_new .> 0)
            error("Gamma IRLS failed: negative coefficients")
        end
        
        if filter_outliers
            fitted_all = X * β_new
            residuals = y ./ max.(fitted_all, 1e-6)
            
            good_idx = (residuals .>= 1e-4) .& (residuals .<= 15.0)
            if sum(good_idx) == 0
                error("All observations filtered out")
            end
        end
        
        # Check convergence using log fold change criterion
        if sum(log.(β_new ./ β).^2) < tol
            return β_new,good_idx
        end
        
        β = β_new
    end
    
    @warn "IRLS did not converge after $max_iter iterations"
    return β,good_idx
end
