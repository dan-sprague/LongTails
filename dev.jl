using Random
using Distributions
using Optim 
using StatsBase
using LinearAlgebra
using Base.Threads
using Plots
using SpecialFunctions
using DataFrames
using StatsModels
using CategoricalArrays
∑ = sum 

const newaxis = [CartesianIndex()]

include("src/data.jl")
include("src/likelihood.jl")
include("src/negbin2.jl")
include("src/transcriptome.jl")
include("src/utils.jl")
include("src/dispersion_trend.jl")
include("src/math.jl")
include("src/gamma.jl")


metadata = DataFrame(
    sample = ["sample1","sample2","sample3","sample4","sample5","sample6"],
    condition = ["control","control","control","treatment","treatment","treatment"],
    batch = ["batch1","batch2","batch1","batch2","batch1","batch2"],
    temp = [1.1,2.3,5.6,1.2,3.4,2.1],
#    y = [1,50,55,2,35,3]
)

FORMULA = "logFC ~ condition + batch"

f = eval(Meta.parse("@formula($FORMULA + 1)"))
design = Design(f,metadata)



config = (distribution = PowerLaw(),
        design = designMatrix(design;expanded=false),
        αtr_σd = 0.1,
        αtr_a1 = 0.5,
        αtr_a0 = 0.02,
        avg_effective_length = 2000,
        n_genes = 10000
)
simulation = rand(DifferentialTranscriptome(config...))
T = simulation.counts 
mask = clean_zeros(T)

data = LongTailsDataSet(T[:,mask], ones(Float64,size(T[:,mask])),design)

α_mom = method_of_moments(data)


log_α_mom = log.(α_mom)


X = [ones(length(μ(data))) 1 ./ μ(data)]
(a0,a1),good_idx = gamma_irls_identity(X, simulation.parameters.α[mask]; max_iter=100, tol=1e-4)
x_tr = 1:1e4
y_tr = atr.(x_tr,a1,a0)

@time (a0,a1),good_idx = gamma_irls_identity(X, α_mom; max_iter=100, tol=1e-6)

x = 1:1e4
y = atr.(x,a1,a0)
function atr(μ, a1, a0)
    (a1 * (1 / μ)) + a0
end


scatter(μ(data)[good_idx],simulation.parameters.α[mask][good_idx],axis=:log,markerstrokewidth=0.0,color=:grey,alpha=0.2,grid=:none,
    xlabel="Mean expression",ylabel="Dispersion parameter",fontfamily = "Arial",dpi=300,
    size=(400,300),tickfontsize=10,guidefontsize=12,label="Simulated Truth",legend=:bottomright)
plot!(x_tr,y_tr,color=:red,linewidth=2,label="α prior (Simulated Truth)")
scatter!(μ(data)[good_idx],α_mom[good_idx],axis=:log,markerstrokewidth=0.0,color=:green,alpha=0.2,grid=:none,
    xlabel="Mean expression",ylabel="Dispersion parameter",fontfamily = "Arial",dpi=300,
    size=(400,300),tickfontsize=10,guidefontsize=12,label="MoM estimates")
plot!(x,y,color=:black,linewidth=2,label="α_mom prior Fit",)




QR = qr(designMatrix)

QR \ T

function cr_grad(X,μ,α)
    W = diagm(@. 1 / ((1 / μ) + α))
    dW = diagm(@. -1/(1/μ + α)^2)

    M = X' * W * X
    det(M) * tr((M \ (X' * dW * X)))
end

function cr_hessian(X, μ̂, α)

    μ = X * μ̂
    # First derivatives of W
    
    # Second derivative of W
    W = diagm(@. 1 / ((1 / μ) + α))
    dW = diagm(@. -1/(1/μ + α)^2)
    d2W = Diagonal(@. 2/(1/μ + α)^3)
    
    M = X' * W * X
    M_inv = M \ I
    
    # Using matrix calculus for second derivative of log determinant
    term1 = tr(M_inv * X' * dW * X)
    term2 = -tr(M_inv * X' * dW * X * M_inv * X' * dW * X)
    term3 = tr(M_inv * X' * d2W * X)
    
    return 0.5 * (term1 + term2 + term3)
end


@time cr_hessian(X,μ̂[:,1],1)

function alt_∂α(x,μ,α)

    ϕ = 1 /α
    acc = 0.0
    @inbounds for j in 0:(x-1)
        acc += (1 / (j + ϕ))
    end
    α^(-2) * (log(1 + α * μ) - acc + ((x - μ) / (α * (1 + α * μ))))
 
end


function alt_∂²α(x,μ,α)
    acc = 0.0
    @inbounds for j in 0:(x-1)
        acc += (j / 1 + α*j)^2
    end

    acc + 
    (2 * α^(-3) * log(1 + α*μ)) - 
    ((2* α^(-2) * μ) / (1 + α*μ)) - 
    (((x + α^(-1)) * μ^2)/(1 + α*μ)^2)
end



iter

xplot = sort(μ̄)
yplot = atr.(xplot,a1,a0)

p1 = scatter(μ̄[fit.final_indices],α_init[fit.final_indices],axis=:log,label=:none,markerstrokewidth=0.0,color=:grey,alpha=0.2,grid=:none,
    xlabel="Mean expression",ylabel="Dispersion parameter",fontfamily = "Arial",dpi=300,
    size=(400,300),legend=:none,tickfontsize=10,guidefontsize=12)
plot!(xplot,yplot,label=:none,markerstrokewidth=0.0,color=:red,alpha=1.0,linewidth=2,)
