"""
    normalize!(X)

Normalizes the columns of a matrix by dividing each column by the geometric mean of the row. Equivalent to DESeq2's median-of-ratios normalization.
"""
function normalize!(X;thr = 1)
    sj = map(median,eachcol(X ./ map(harmmean,eachrow(X))))
    X ./= transpose(sj)   
end


meancount(t,X) = transpose(X) * t 

function method_of_moments(X,t)
    μ = meancount(t,X)
    s² = map(x -> var(t[x .== 1]),eachcol(X))
    
    α_est = @. (s² - μ) / μ^2
    weights = μ.^2
    α_est = sum(α_est .* weights) / sum(weights)

    max(α_est,1e-8)
end

μ̂(X,θ,b) =  X * θ .+ b
