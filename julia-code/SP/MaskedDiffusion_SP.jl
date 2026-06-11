module Eq

include("common.jl")
include("methods.jl")
include("helpers.jl")

using LinearAlgebra
using PyCall
using Optimization
using OptimizationOptimJL
using Optim
using SpecialFunctions # for gamma, √π, erf


############### PARAMS ################

mutable struct OrderParams <: AbstractParams
    q::Float64
    δq::Float64
end

mutable struct ExtParams <: AbstractParams
    α::Float64
    λ::Float64
end

struct Params <: AbstractParams
    ϵ::Float64 # accuracy
    ψ::Float64 # damping
    maxiters::Int
    verb::Int
end

mutable struct ThermFunc <: AbstractParams
    ϕ::Float64
end

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, params::Params) = shortshow(io, params)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

Base.:(==)(x::OrderParams, y::OrderParams) = recursiveeq(x, y)
Base.:(==)(x::ExtParams, y::ExtParams) = recursiveeq(x, y)
Base.:(==)(x::Params, y::Params) = recursiveeq(x, y)
Base.:(==)(x::ThermFunc, y::ThermFunc) = recursiveeq(x, y)

collect(op::OrderParams) = [getfield(op, f) for f in fieldnames(typeof(op))]

###################################################################################

function σ(z)
    return 1 / (1 + exp(-z))
end

function l(q::Float64, μ::Float64)
    return -∫d(t -> ∫D(z -> log(σ(√(t * (1 - t) * q) * z + (1 - t) * μ))), [0, 1])
end

function fₑ(q::Float64, δq::Float64, μ::Float64, x::Float64)
    -μ^2 / (2 * δq) + x * μ * √q / δq - l(q, μ)
end

function Fₑ(q::Float64, δq::Float64)
    return ∫D(
        x ->
            begin
                μ̂ = iμ_newton(q, δq, x)
                return fₑ(q, δq, μ̂, x)
            end
    )
end

function Gₑ(q::Float64, δq::Float64)
    return Fₑ(q, δq) - q / (2 * δq)
end

function ϕ(q::Float64, δq::Float64, α::Float64, λ::Float64)
    return α * Gₑ(q, δq) + q / (2 * δq) - λ * q / 2
end


function ε_greedy(t, q, δq)
    return ∫D(
        x ->
            begin
                μ̂ = iμ_newton(q, δq, x)
                return Φ(μ̂ / √q * √((1 - t) / t))
            end
    )
end

function ε_fair(t, q, δq)
    return ∫D(
        x ->
            begin
                μ̂ = iμ_newton(q, δq, x)
                return ∫D(y -> σ(y * √((1 - t) * t * q) + μ̂ * (1 - t)))
            end
    )
end

############################## Derivatives ############################

function ∂q_l(q::Float64, μ::Float64)
    return ∫d(t -> ∫D(z -> σ(√(t * (1 - t) * q) * z + (1 - t) * μ) * z * √(t * (1 - t)) / (2 * √q)), [0, 1])
end

function ∂μ_l(q::Float64, μ::Float64)
    return ∫d(t -> ∫D(z -> σ(√(t * (1 - t) * q) * z + (1 - t) * μ) * (1 - t)), [0, 1]) - 1 / 2
end

function ∂q_fₑ(q::Float64, δq::Float64, μ::Float64, x::Float64)
    return μ * x / (2 * δq * √q) - ∂q_l(q, μ)
end

function ∂δq_fₑ(q::Float64, δq::Float64, μ::Float64, x::Float64)
    return (μ^2 / 2 - √q * μ * x) / (δq^2)
end

function ∂q_Fₑ(q::Float64, δq::Float64)
    out = ∫D(x -> begin
        μ̂ = iμ_newton(q, δq, x)
        return ∂q_fₑ(q, δq, μ̂, x)
    end)
    return out[1]
end

function ∂δq_Fₑ(q::Float64, δq::Float64)
    out = ∫D(x -> begin
        μ̂ = iμ_newton(q, δq, x)
        return ∂δq_fₑ(q, δq, μ̂, x)
    end)
    return out[1]
end

function ∂q_Gₑ(q::Float64, δq::Float64)
    return ∂q_Fₑ(q, δq) - 1 / (2 * δq)
end

function ∂δq_Gₑ(q::Float64, δq::Float64)
    return ∂δq_Fₑ(q, δq) + q / (2 * δq^2)
end

f_q(q::Float64, δq::Float64, α::Float64) = 2 * δq^2 * α * ∂δq_Gₑ(q, δq)
f_δq(q::Float64, δq::Float64, α::Float64, λ::Float64) = 1 / (λ - 2α * ∂q_Gₑ(q, δq))

############ Thermodynamic functions ############

function all_therm_func(op, ep)
    return ThermFunc(ϕ(op.q, op.δq, ep.α, ep.λ))

end

#################################################

function iμ_newton(q, δq, x)
    μ_0 = 5.0
    _, μ = newton(μ -> begin
            return -μ / δq + x * √q / δq - Eq.∂μ_l(q, μ)
        end, μ_0, NewtonMethod(dx=1e-6, verb=0, atol=1e-8))
    return μ
end

#################################################

function converge!(op::OrderParams, ep::ExtParams, pars::Params)
    @extract ep:α λ
    @extract pars:maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    for it = 1:maxiters
        oldΔ = Δ
        Δ = 0.0

        verb > 1 && println("it=$it ψ=$ψ")

        @update op.q f_q Δ ψ verb op.q op.δq ep.α
        @update op.δq f_δq Δ ψ verb op.q op.δq ep.α ep.λ

        verb > 1 && println(" Δ=$Δ\n")

        ok = Δ < ϵ * (1 - ψ)
        ok && break
        if Δ > oldΔ
            ψ = 1 - (1 - ψ) * 0.7
        else
            ψ *= 0.95
        end
    end

    if verb > 0
        println(ok ? "converged" : "failed", " (it=$it Δ=$Δ)")
        println(op)
        println(ep)
    end

    return ok
end

function converge(; q=0.1, δq=0.1, μ=0.1, α=0.1, λ=0.1,
    ϵ=1e-6, ψ=0.0, maxiters=10_000, verb=1)

    op = OrderParams(q, δq)
    ep = ExtParams(α, λ)
    pars = Params(ϵ, ψ, maxiters, verb)

    converge!(op, ep, pars)
    tf = all_therm_func(op, ep)
    if verb > 0
        println(tf)
    end
    return op, ep, tf, pars
end


function span(; q=0.1, δq=0.1, α=0.1,
    λ=1.0,
    ϵ=1e-5, ψ=0.0, maxiters=1_000, verb=1,
    resfile=nothing)

    default_resfile = "results2.txt"
    resfile ≡ nothing && (resfile = default_resfile)

    if !isfile(resfile)
        open(resfile, "w") do f
            print(f, "#1=α 2=λ ")
            i0 = 2
            allheadersshow(f, OrderParams, ThermFunc, i0=i0)
        end
    end

    op = OrderParams(q, δq)
    ep = ExtParams(first(α), first(λ))
    pars = Params(ϵ, ψ, maxiters, verb)

    results = []

    for b in λ
        for z in α
            println("\n########  NEW ITER  ########\n")
            ep.α = z
            ep.λ = b
            verb > 0 && println("α=$(ep.α) λ=$(ep.λ)")

            ok = converge!(op, ep, pars)

            tf = all_therm_func(op, ep)
            push!(results, (ok, deepcopy(op), deepcopy(ep), deepcopy(tf)))
            verb > 0 && println(tf)
            if ok
                open(resfile, "a") do rf
                    print(rf, "$(ep.α) $(ep.λ)")
                    println(rf, " ", plainshow(op), " ", plainshow(tf))
                end
            end
            ok || break
        end
    end
    return results
end

function compute_ε(; infile=nothing, outfile=nothing, t, algorithm=:greedy, α=nothing, atol=1e-8)
    if outfile === nothing
        outfile = replace(infile, ".txt" => "_ε_$(algorithm)_t_$t.txt")
    end

    input_df = file_to_df(infile)
    if α !== nothing
        input_df = filter(row -> row.α in α, input_df)
    end
    output_df = DataFrame()

    for b in t
        for row in eachrow(input_df)
            α = row.α
            λ = row.λ
            q = row.q
            δq = row.δq
            ϕ = row.ϕ

            accuracy = algorithm == :greedy ? ε_greedy(b, q, δq) : ε_fair(b, q, δq)
            row = (; α, λ, q, δq, ϕ, t=b, ε=accuracy)
            save_row!(row, output_df, outfile)
        end
    end
end

function maximise_μ(; infile=nothing, outfile=nothing, α, t, xrange=-5:0.01:5)
    if outfile === nothing
        outfile = replace(infile, ".txt" => "_μ.txt")
    end

    input_df = file_to_df(infile)
    input_df_α = filter(row -> row.α in α, input_df)
    output_df = DataFrame()

    for time in t
        for row in eachrow(input_df_α)
            for x in xrange
                α_val = row.α
                λ = row.λ
                q = row.q
                δq = row.δq
                ϕ = row.ϕ

                μ̂ = iμ_newton(q, δq, x)

                row = (; α=α_val, λ, q, δq, ϕ, t=time, x, μ=μ̂)
                save_row!(row, output_df, outfile)
            end
        end
    end
end

end # module

