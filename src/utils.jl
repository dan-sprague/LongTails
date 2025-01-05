function clean_zeros(T)
    T = T[:,.!(map(x -> all(x .== 0),eachcol(T)))]
end

"""
    normalize!(X)

Normalizes the columns of a matrix by dividing each column by the geometric mean of the row. Equivalent to DESeq2's median-of-ratios normalization.
"""
function normalize!(X;thr = 1)
    sj = map(median,eachcol(X ./ map(harmmean,eachrow(X))))
    X ./= sj'   
end


