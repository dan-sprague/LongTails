function clean_zeros(T)
    vec(sum(T .!= 0;dims=1) .!= 0)
end


