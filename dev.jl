using Plots,StatsPlots
using DataFrames,CSV  
using Optim
using StatsBase
const newaxis = [CartesianIndex()]
x = 1:50:1000

# Add random noise to the log of the dispersion trend for simulation purposes

df = CSV.read("ENCFF545VIZ.tsv",DataFrame)
x = @. Int(round(df.est_counts))
x = x[x .> 0]
x = sample(x,1000)
θ = nbreg_transform.(x,atr_sim.(x))

dists_1 = map(θ -> rand(NegativeBinomial(θ...),3),θ)


perc_expanding = 0.05

expanding = [rand(Bernoulli(perc_expanding)) for i in eachindex(x)]

FC = rand(Gamma(gamma_reg_transform(2,.005)...),size(x,1)) .* expanding
FC[FC .== 0] .= 1
FC = map(x -> rand(Bernoulli(0.5)) == 1 ? x : 1 / x,FC)
θ2 = nbreg_transform.(x .* FC,atr_sim.(x .* FC))

dists_2 = map(θ -> rand(NegativeBinomial(θ...),3),θ2)



X = permutedims(vcat(stack(dists_1),stack(dists_2)))
X = X[.!(map(x -> all(x .== 0),eachrow(X))),:]
μ̄ = dropdims(mean(X;dims=2);dims=2)

α_init = zeros(Float64,size(X,1))

d = [1,1,1,2,2,2]

d = Int.(d .== unique(d)')

a = 1
x = X[1,:]
μ̂ = X[:,:] * d ./ sum(d;dims=1)
μ̄ = dropdims(mean(μ̂;dims=2);dims=2)


function nb_alpha_cr_nll(x,d,μ̂,μ̄,α)
    nb_nll = nbreg_nll(x,log(μ̄),α)
    w = diagm(1 ./ ((1 ./ (d * μ̂)) .+ exp.(α)))

    cr = 0.5 * log(det(d' * w * d))
    nb_nll + cr

end


function gamma_trend_nll(α,x,a1,a0,ϕ)
    
    a1 = exp(a1)
    a0 = exp(a0)
    ϕ = exp(ϕ)
    μ = atr.(x,a1,a0)

    k,θ = gamma_reg_transform(μ,ϕ)
    sum(@. -logpdf(Gamma(k,θ),α))
end



Threads.@threads for i in axes(X,1) 
    α_init[i] = exp(Optim.minimizer(optimize(α -> nb_alpha_cr_nll(X[i,:],d,μ̂[i,:],μ̄[i],α),[0.0]))[1])

end

scatter(μ̄,α_init,axis=:log)

a1,a0, = exp.(Optim.minimizer(optimize(θ -> gamma_trend_nll(α_init,μ̄,θ...),[0.0,0.0,0.0],))[1:2])

scatter!(μ̄,atr.(μ̄,a1,a0))

using Random

include("src/negbin.jl")
include("src/transcriptome.jl")


function FC!(T::Matrix,X::Vector;perc_expanding=0.05,a=2,b=0.005)
    
    n_genes = size(T,2)
    α,β = gamma_reg_transform(a,b)

    gene_FC = rand(Bernoulli(perc_expanding), n_genes) .* rand(Gamma(α,β),n_genes)


    FC_matrix = ones(n_genes,length(X))
    FC_matrix[:,X .== 1] .= gene_FC 

    T .= Int.(ceil.(T .* permutedims(FC_matrix)))
end

@time T = rand(Transcriptome(PowerLaw(),1.5,6,100))

X = [0,0,0,1,1,1]

@time FC!(T,X;perc_expanding=1)


T = T[:,.!(map(x -> all(x .== 0),eachcol(T)))]
μ̄ = dropdims(mean(T;dims=1);dims=1)

α_init = zeros(Float64,size(T,2))



X = Int.(X .== unique(X)')

μ̂_gr = (permutedims(X) * T) .* permutedims((1 ./ sum(X;dims=1)))
μ̄ = dropdims(mean(T;dims=1);dims=1)


α_init = zeros(Float64,size(T,2))
for i in axes(T,2) 
    α_init[i] = exp(Optim.minimizer(optimize(α -> nb_alpha_cr_nll(T[:,i],X,μ̂[:,i],μ̄[i],α),[0.0]))[1])
end
