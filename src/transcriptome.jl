using Random 

"""
    PowerLaw

A power law distribution with parameters α and x_min, and normalization constant C.

"""
struct PowerLaw
    α::Float64
    x_min::Int64
    C::Float64
end

"""
    PowerLaw(;α=2.0,x_min=1)

Constructs a PowerLaw distribution with parameters α and x_min, and normalization constant C.

"""
PowerLaw(;α=2.0,x_min=1) = PowerLaw(
    α,
    x_min,
    (α - 1) * x_min^(α - 1)
)


"""
    logpdf(d::PowerLaw,x)

The log probability density function of a PowerLaw distribution, over a vector or scalar x.
"""
function Distributions.logpdf(d::PowerLaw,x)
    sum(@. _logpdf(d,x))
end

function Base.rand(rng::AbstractRNG,d::PowerLaw)
    U = rand(rng)
    X = ((d.C / ((d.α - 1) * U))^(1 / (d.α - 1)))

    X 
end

function Base.rand(rng::AbstractRNG,d::PowerLaw,n::Integer)
    [rand(rng,d) for i in 1:n]
end



function Base.rand(d::PowerLaw,n::Integer)
    rand(Random.default_rng(),d,n)
end

struct Transcriptome
    c::Vector{Float64}
    σd::Float64
    K::Int
    n::Int
end

function Transcriptome(d::PowerLaw,σd,K,n)
    Transcriptome(rand(d,n),σd,K,n)
end


function αtr_sample(x,σ)
    exp(rand(Normal(log(x),σ)))
end

function Base.rand(rng::AbstractRNG,d::Transcriptome)

    α = @. αtr_sample(atr_sim(d.c),d.σd)
    θ = nbreg_transform.(d.c,α)

    reduce(hcat,map(θ -> rand(NegativeBinomial(θ...),d.K),θ))

end


function FC!(T::Matrix,X::Vector;perc_expanding=0.05,a=2,b=0.005)
    
    n_genes = size(T,2)
    α,β = gamma_reg_transform(a,b)

    gene_FC = rand(Bernoulli(perc_expanding), n_genes) .* rand(Gamma(α,β),n_genes)


    FC_matrix = ones(n_genes,length(X))
    FC_matrix[:,X .== 1] .= gene_FC 

    T .= Int.(ceil.(T .* permutedims(FC_matrix)))
end


