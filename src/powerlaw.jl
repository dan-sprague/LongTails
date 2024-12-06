


using Random
struct PowerLaw
    α::Float64
    x_min::Int64
    C::Float64
end

PowerLaw(α,x_min) = PowerLaw(
    α,
    x_min,
    (α - 1) * x_min^(α - 1)
)

function _logpdf(d::PowerLaw,x)
    ((d.α - 1) / d.x_min) + (-d.α  * log(x / d.x_min))
end

function logpdf(d::PowerLaw,x)
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

samples = rand(PowerLaw(2,1),20_000)
samples = sort(samples;rev=true) ./ sum(samples)
plot(samples,scale=:log10)


function cdf(d::PowerLaw,x)
    (d.C / (d.α - 1)) * (x^-(d.α - 1))
end




function powerlaw(x,α)
    x^-(α)
end


freqs(n;α = 0.8) = map(i -> powerlaw(i,α),1:n)


1 / 27

f = freqs(10_000)
Z = sum(f)

pᵢ =  f ./ Z 