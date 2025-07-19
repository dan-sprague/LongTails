function clean_zeros(T)
    T[:,vec(sum(T .!= 0;dims=1) .!= 0)]
end


