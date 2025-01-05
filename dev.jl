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
using Distributions
using Distributions
using Optim 
using StatsBase
using LinearAlgebra
using Base.Threads
using Plots

const newaxis = [CartesianIndex()]

include("src/negbin.jl")
include("src/transcriptome.jl")
include("src/utils.jl")
T = rand(Transcriptome(PowerLaw(),1.0,6,10000))

X = [0,0,0,1,1,1]

FC!(T,X;perc_expanding=1)

T = clean_zeros(T)
μ̄ = dropdims(mean(T;dims=1);dims=1)

α_init = zeros(Float64,size(T,2))



X = Int.(X .== unique(X)')

μ̂_gr = (permutedims(X) * T) .* permutedims((1 ./ sum(X;dims=1)))
μ̄ = dropdims(mean(T;dims=1);dims=1)


α_init = zeros(Float64,size(T,2))
@threads for i in axes(T,2) 
    α_init[i] = exp(Optim.minimizer(optimize(α -> nb_alpha_cr_nll(T[:,i],X,μ̂_gr[:,i],μ̄[i],α),[0.0]))[1])
end

a1,a0, = exp.(Optim.minimizer(optimize(θ -> gamma_trend_nll(α_init,μ̄,θ...),[0.0,0.0,0.0],))[1:2])

xplot = sort(μ̄)
yplot = atr.(xplot,a1,a0)

p1 = scatter(μ̄,α_init,axis=:log,label=:none,markerstrokewidth=0.0,color=:grey,alpha=0.2,grid=:none,
    xlabel="Mean expression",ylabel="Dispersion parameter",fontfamily = "Helvetica",dpi=300,
    size=(400,300),legend=:none,tickfontsize=10,guidefontsize=12)
plot!(xplot,yplot,label=:none,markerstrokewidth=0.0,color=:red,alpha=1.0,linewidth=2,)


α_init 

α_fit = atr.(μ̄,a1,a0)

α_ratio = α_fit ./ α_init

idx = @. 1e-4 <= α_ratio <= 15 
α_init = zeros(Float64,sum(idx))


T_tmp,μ̂_gr_tmp,μ̄_tmp =  subset(idx,T,μ̂_gr,μ̄)

@threads for i in axes(T,2) 
    α_init[i] = exp(Optim.minimizer(optimize(α -> nb_alpha_cr_nll(T[:,i],X,μ̂_gr[:,i],μ̄[i],α),[0.0]))[1])
end


function iter_fit_atr(T,X,μ̂_gr,μ̄;max_iter = 100)

    converged = false
    iter = 0
    rel_diff = 1.0

    
    α_0 = zeros(Float64,size(T,2))
    @threads for i in axes(T,2) 
        α_init[i] = exp(Optim.minimizer(optimize(α -> nb_alpha_cr_nll(T[:,i],X,μ̂_gr[:,i],μ̄[i],α),[0.0]))[1])
    end
    a1,a0, = exp.(Optim.minimizer(optimize(θ -> gamma_trend_nll(α_init,μ̄,θ...),[0.0,0.0,0.0],))[1:2])

    α_fit = atr.(μ̄,a1,a0)

    α_ratio = α_fit ./ α_init

    idx = @. 1e-4 <= α_ratio <= 15

    T_tmp,μ̂_gr_tmp,μ̄_tmp =  subset(idx,T,μ̂_gr,μ̄)



end