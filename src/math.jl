
"""
    normalize(X)

Normalizes the columns of a matrix by dividing each column by the geometric mean of the row. Equivalent to DESeq2's median-of-ratios normalization.
"""
function factors(X)
    map(median,eachcol(X ./ map(geomean,eachcol(X))))
end


meancount(t,X) = transpose(X) * t 

function method_of_moments(x)
    μ = mean(x)
    s² = var(x)

    max((s² - μ) / μ^2,1e-4)
end

function linear_μ(model::NegBin2)
    X * ((model.X'model.X) \ (model.X'model.norm_counts))
end


