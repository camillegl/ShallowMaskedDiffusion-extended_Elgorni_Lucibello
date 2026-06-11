# Compute the Shannon (α = 1) replica-symmetric entropy density of the
# hidden-manifold output distribution with a noiseless sign channel, sweeping
# the load γ = N/D. Writes a CSV to data/entropy_sign_shannon.csv.
#
# Usage from julia-code/hiddenmanifold/:
#   julia --project=. scripts/sweep_shannon.jl                 # default γ-grid (0.05..10)
#   julia --project=. scripts/sweep_shannon.jl 0.05 20 0.05    # γmin γmax step
#
# CLI args (positional, all optional):
#   γmin    lower edge of the γ-grid (default 0.05)
#   γmax    upper edge of the γ-grid (default 10.0)
#   step    γ step in the dense region (default 0.05 below 1, 0.10 above 1)

include(joinpath(@__DIR__, "..", "SignChannel.jl"))
using .SignChannel
using DelimitedFiles

const ROOT     = abspath(joinpath(@__DIR__, ".."))
const CSV_PATH = joinpath(ROOT, "data", "entropy_sign_shannon.csv")

function parse_args()
    γmin = length(ARGS) ≥ 1 ? parse(Float64, ARGS[1]) : 0.05
    γmax = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 10.0
    step = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : NaN
    return γmin, γmax, step
end

function default_grid(γmin, γmax)
    sort(unique(vcat(
        range(γmin,         min(0.95, γmax); step = 0.05),
        range(max(1.0,γmin), γmax;            step = 0.10),
        γmin ≤ 2.0 ≤ γmax ? [2.0] : Float64[],
    )))
end

function uniform_grid(γmin, γmax, step)
    sort(unique(vcat(collect(γmin:step:γmax),
                     γmin ≤ 2.0 ≤ γmax ? [2.0] : Float64[])))
end

function main()
    γmin, γmax, step = parse_args()
    γs = isnan(step) ? default_grid(γmin, γmax) : uniform_grid(γmin, γmax, step)

    println("Computing Shannon entropy at $(length(γs)) values of γ ∈ [$γmin, $γmax]…")
    sh = SignChannel.span(γs; q0 = 1e-3, ψ = 0.3, atol = 1e-10, verb = 0)

    bad = [r.γ for r in sh if !r.converged]
    isempty(bad) || @warn "Shannon saddle did not converge at γ = $bad"

    header = ["gamma", "q", "qhat", "I", "s_cond", "s_prior", "s"]
    mat = fill(NaN, length(γs), length(header))
    for (i, r) in enumerate(sh)
        mat[i, :] = [r.γ, r.q, r.q̂, r.I, r.s_cond, r.s_prior, r.s]
    end

    open(CSV_PATH, "w") do io
        println(io, join(header, ","))
        writedlm(io, mat, ',')
    end
    println("Wrote ", CSV_PATH, "  ($(size(mat,1)) rows × $(size(mat,2)) cols)")
end

main()
