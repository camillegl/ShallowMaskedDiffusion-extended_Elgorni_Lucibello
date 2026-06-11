# Sweep the Rényi-entropy saddle (α ≠ 1) for the noiseless sign channel,
# writing one CSV per requested α value. The γ-grid is read from the Shannon
# CSV and q₁ is warm-started from the Shannon q* at each γ; on subsequent
# sweeps the previous α's solution is also used as a fallback warm start.
#
# Output: data/entropy_sign_renyi_alpha<α>.csv
#
# Usage from julia-code/hiddenmanifold/:
#   julia --project=. scripts/sweep_renyi.jl                       # default αs
#   julia --project=. scripts/sweep_renyi.jl 0.5 0.9 2.0 5.0 10.0  # custom αs
#
# Pre-requisite: data/entropy_sign_shannon.csv must exist
#   (run scripts/sweep_shannon.jl first).

include(joinpath(@__DIR__, "..", "SignChannelRenyi.jl"))
import .SignChannelRenyi as R
using DelimitedFiles
using Printf

const ROOT        = abspath(joinpath(@__DIR__, ".."))
const SHANNON_CSV = joinpath(ROOT, "data", "entropy_sign_shannon.csv")

const DEFAULT_αs  = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9, 2.0, 5.0, 10.0]

function parse_alphas()
    isempty(ARGS) && return DEFAULT_αs
    return parse.(Float64, ARGS)
end

function load_shannon()
    isfile(SHANNON_CSV) ||
        error("Shannon CSV not found at $SHANNON_CSV — run scripts/sweep_shannon.jl first")
    data, hdr = readdlm(SHANNON_CSV, ',', header=true)
    hdr = vec(hdr)
    γs = vec(data[:, findfirst(==("gamma"), hdr)])
    qs = vec(data[:, findfirst(==("q"),     hdr)])
    return γs, qs
end

function try_solve(γ, n, q1_init, ψ; atol=1e-10, maxiters=5000)
    try
        return R.renyi_entropy_density(γ, n;
            q1 = clamp(q1_init, 1e-5, 0.999),
            ψ = ψ, atol = atol, maxiters = maxiters)
    catch err
        @warn "Solver threw at γ=$γ ψ=$ψ" exception=err
        return (; γ, n, α=n+1, q1=NaN, q1hat=NaN, Phi_n=NaN, S_prior=NaN,
                  s_out=NaN, s_prior=NaN, s=NaN, iters=0, converged=false)
    end
end

# Try a sequence of damping factors before giving up at a given γ.
function robust_solve(γ, n, q1_init)
    res = try_solve(γ, n, q1_init, 0.5)
    res.converged && return res
    for (ψ, iters) in ((0.7, 8000), (0.85, 12000), (0.95, 20000),
                       (0.98, 40000), (0.995, 80000))
        seed = isfinite(res.q1) ? res.q1 : q1_init
        alt  = try_solve(γ, n, seed, ψ; maxiters=iters)
        alt.converged && return alt
    end
    return res
end

function sweep(γs, q1_inits, n)
    rows = NamedTuple[]
    for (γ, q1i) in zip(γs, q1_inits)
        res = robust_solve(γ, n, q1i)
        push!(rows, res)
        @printf("  %7.3f  %9.6f  %12.6e  %9.6f  %9.6f  %+9.6f  %7d%s\n",
                res.γ, res.q1, res.q1hat, res.s, res.s_out, res.s_prior,
                res.iters, res.converged ? "" : "  [✗]")
    end
    return rows
end

function write_csv(path, γs, rows, α, n)
    header = ["gamma", "q1", "q1hat", "Phi_n", "S_prior", "s_out", "s_prior", "s"]
    mat = fill(NaN, length(γs), length(header))
    for (i, r) in enumerate(rows)
        r.converged || continue
        mat[i, :] = [r.γ, r.q1, r.q1hat, r.Phi_n, r.S_prior,
                     r.s_out, r.s_prior, r.s]
    end
    open(path, "w") do io
        println(io, "# Rényi α=$α (n=$n) RS entropy density for the noiseless sign channel (q₀=0 ansatz)")
        println(io, "# See §4.3 of notes/notes_hiddenmanifold.typ (eq:sign-renyi-main)")
        println(io, join(header, ","))
        writedlm(io, mat, ',')
    end
    nconv = count(r -> r.converged, rows)
    println("Wrote $path  ($nconv/$(length(γs)) converged)")
end

function main()
    αs = parse_alphas()
    γs, qs = load_shannon()

    # Carry the previous α's q₁ forward as an additional warm-start hint.
    prev_q1 = copy(qs)

    for α in αs
        n = α - 1.0
        out_csv = joinpath(ROOT, "data", "entropy_sign_renyi_alpha$α.csv")

        println()
        println("="^72)
        println("# α = $α (n = $n)")
        println("# γ-grid: $(length(γs)) points from $(γs[1]) to $(γs[end])")
        @printf("# %-7s  %-9s  %-12s  %-9s  %-9s  %-9s  %-7s\n",
                "γ", "q₁*", "q̂₁*", "s_α", "s_out", "s_prior", "iters")

        # Better warm start: average of Shannon q* and the previous α's q₁.
        warm = [0.5 * (qs[i] + prev_q1[i]) for i in eachindex(γs)]
        rows = sweep(γs, warm, n)

        bad = [r.γ for r in rows if !r.converged]
        isempty(bad) || @warn "α=$α did not converge at γ = $bad"

        write_csv(out_csv, γs, rows, α, n)

        prev_q1 = [r.converged ? r.q1 : prev_q1[i] for (i, r) in enumerate(rows)]
    end
end

main()
