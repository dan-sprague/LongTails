
"""
    normalize(X)

Normalizes the columns of a matrix by dividing each column by the geometric mean of the row. Equivalent to DESeq2's median-of-ratios normalization.
"""
function factors(X)
    map(median,eachcol(X ./ map(geomean,eachcol(X))))
end


meancount(t,X) = transpose(X) * t 

function method_of_moments(x::LongTailsDataSet)
    xim = mean(1 ./ x.nf)
    @. max((x.σ² - xim*x.μ) / x.μ,1e-4)
end

function linear_μ(data::LongTailsDataSet)
    QR = qr(X)

    QR \ counts(data;norm=true)
end


