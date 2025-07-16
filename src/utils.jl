function clean_zeros(T)
    mask = map(t -> any((transpose(X) * t) .== 0),eachcol(T))

    T[:,.!mask]
end


