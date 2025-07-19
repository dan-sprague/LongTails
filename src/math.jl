
"""
    normalize(X)

Normalizes the columns of a matrix by dividing each column by the geometric mean of the row. Equivalent to DESeq2's median-of-ratios normalization.
"""
function factors(X)
    map(median,eachcol(X ./ map(geomean,eachcol(X))))
end


meancount(t,X) = transpose(X) * t 

function method_of_moments(x::LongTailsDataSet)
    xim = mean(1 ./ nf(x))
    mu,var = μ(x), σ²(x)
    
    @. max((var - mu) / mu^2,1e-4)
end

function linear_μ(data::LongTailsDataSet)
    QR = qr(designMatrix(data))
    QR \ counts(data;norm=true)
end



