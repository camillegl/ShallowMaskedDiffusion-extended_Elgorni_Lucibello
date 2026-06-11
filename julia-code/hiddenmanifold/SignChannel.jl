"""
    SignChannel

Bayes-optimal RS saddle-point solver for the Shannon entropy of the
hidden-manifold output distribution with the noiseless sign channel
P_out(y|x) = δ_{y, sign(x)}.

Implements the formulas from notes/notes_hiddenmanifold.typ (Section 4):

    P̃(y|z)     = Φ(yz / √(1-q))                            [eq:sign-Ptilde]

    I(q)       = (1/(1-q)) E_{t ~ N(0, q/(1-q))}
                   [ φ²(t) / (Φ(t)(1-Φ(t))) ]                [eq:sign-fisher-t]

    q̂          = γ I(q)                                      [eq:sign-saddle]
    q          = q̂ / (1+q̂)

    s          = E_{t ~ N(0, q*/(1-q*))} [ h₂(Φ(t)) ]
                 + (1/(2γ)) [ -log(1-q*) - q* ]               [eq:sign-shannon]
"""
module SignChannel

using SpecialFunctions: erfc, erfcx, logerfc
using QuadGK
using Printf

export solve_saddle, entropy_density, span,
       fisher, cond_entropy, h2_Φ, J_fisher

# ---------------------------------------------------------------------------
# Special-function helpers (numerically stable on the full real line)
# ---------------------------------------------------------------------------

const _SQRT2  = sqrt(2.0)
const _SQRT2π = sqrt(2π)
const _LOG2   = log(2.0)
# Beyond |t| ≈ 25 both J(t) and h₂(Φ(t)) are < 1e-130 so contribute nothing
# to any Gaussian-weighted average, and erfcx(-t/√2) starts to overflow.
const _T_MAX  = 25.0

@inline ϕnorm(t) = exp(-t^2 / 2) / _SQRT2π
@inline Φnorm(t) = erfc(-t / _SQRT2) / 2

# log Φ(t) and log(1-Φ(t)) via logerfc (stable for all t).
@inline logΦ(t)   = logerfc(-t / _SQRT2) - _LOG2
@inline log1mΦ(t) = logerfc( t / _SQRT2) - _LOG2

"""
    J_fisher(t)

Fisher-information integrand of the probit channel in the rescaled coordinate
`t = z/√(1-q)`:

    J(t) = φ²(t) / (Φ(t)(1-Φ(t))) = 2 / (π · erfcx(t/√2) · erfcx(-t/√2)).

The product form is finite even where φ², Φ, and 1-Φ all vanish.
For |t| beyond `_T_MAX` falls back to the Mills-tail asymptotic
J(t) ≈ |t| φ(t).
"""
function J_fisher(t)
    a = abs(t)
    a > _T_MAX && return a * exp(-t^2 / 2) / _SQRT2π
    e1 = erfcx( t / _SQRT2)
    e2 = erfcx(-t / _SQRT2)
    return 2 / (π * e1 * e2)
end

"""
    h2_Φ(t)

Binary entropy of `Φ(t)` (in nats), using stable `logerfc`.
"""
function h2_Φ(t)
    abs(t) > _T_MAX && return 0.0
    Φt = Φnorm(t)
    return -Φt * logΦ(t) - (1 - Φt) * log1mΦ(t)
end

# ---------------------------------------------------------------------------
# Gaussian quadrature against N(0, σ²)
# ---------------------------------------------------------------------------

"""
    gauss_int(f, σ; rtol, atol)

`E_{t ~ N(0, σ²)} f(t)` via adaptive QuadGK. Substitutes `t = σ u` so the
quadrature is always against a *standard* normal (whose support and scale
are O(1)) regardless of σ — this avoids the failure mode where, for σ ≪ 1,
the original integrand is a tall narrow spike that adaptive routines miss.
"""
function gauss_int(f, σ; rtol = 1e-10, atol = 1e-12)
    σ < 1e-12 && return f(0.0)
    g = u -> f(σ * u) * exp(-u^2 / 2) / _SQRT2π
    val, _ = quadgk(g, -Inf, 0.0, Inf; rtol = rtol, atol = atol)
    return val
end

# ---------------------------------------------------------------------------
# Saddle-point ingredients
# ---------------------------------------------------------------------------

"""
    fisher(q; kwargs...)

Channel Fisher information `I(q)` of the probit effective channel at overlap `q`.
"""
function fisher(q; kwargs...)
    0 ≤ q < 1 || throw(DomainError(q, "q must be in [0,1)"))
    σ = sqrt(q / (1 - q))
    return gauss_int(J_fisher, σ; kwargs...) / (1 - q)
end

"""
    cond_entropy(q; kwargs...)

Conditional output entropy `E_{t ~ N(0, q/(1-q))} [h₂(Φ(t))]` (nats).
"""
function cond_entropy(q; kwargs...)
    0 ≤ q < 1 || throw(DomainError(q, "q must be in [0,1)"))
    σ = sqrt(q / (1 - q))
    return gauss_int(h2_Φ, σ; kwargs...)
end

# Single saddle map: given q, return (q_new, q̂, I(q)).
function saddle_step(q, γ; kwargs...)
    Iq = fisher(q; kwargs...)
    q̂  = γ * Iq
    return q̂ / (1 + q̂), q̂, Iq
end

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

"""
    solve_saddle(γ; q0=0.5, ψ=0.5, atol=1e-10, maxiters=10_000, verb=0, quad_kwargs=(;))

Damped fixed-point iteration of the Bayes-optimal RS saddle for the noiseless
sign channel at load `γ = N/D`. Returns

    (; γ, q, q̂, I, iters, converged).

`ψ ∈ [0,1)` is the damping factor (0 = no damping). Pass quadrature tolerances
through `quad_kwargs`, e.g. `quad_kwargs = (rtol=1e-12,)`.
"""
function solve_saddle(γ; q0 = 0.5, ψ = 0.5, atol = 1e-10,
                      maxiters = 10_000, verb = 0, quad_kwargs = (;))
    γ > 0 || throw(DomainError(γ, "γ must be > 0"))
    q  = clamp(q0, 1e-12, 1 - 1e-12)
    q̂  = NaN
    Iq = NaN
    converged = false
    it = 0
    for outer it in 1:maxiters
        qnew, q̂, Iq = saddle_step(q, γ; quad_kwargs...)
        Δ = abs(qnew - q)
        q = (1 - ψ) * qnew + ψ * q
        verb > 1 && @printf("  it=%4d  q=%.12f  q̂=%.6e  I=%.6f  Δ=%.2e\n",
                            it, q, q̂, Iq, Δ)
        if Δ < atol
            converged = true
            break
        end
    end
    verb > 0 && println(converged ? "✓ converged" : "✗ failed",
                       @sprintf("  γ=%.4f  q=%.10f  q̂=%.6e  iters=%d", γ, q, q̂, it))
    return (; γ, q, q̂, I = Iq, iters = it, converged)
end

"""
    entropy_density(γ; kwargs...)

Bayes-optimal RS Shannon entropy density (nats) of the hidden-manifold output
distribution at load `γ` for the noiseless sign channel. Splits the total into
the conditional-entropy term `s_cond` and the prior-overlap term `s_prior`.
"""
function entropy_density(γ; kwargs...)
    sol     = solve_saddle(γ; kwargs...)
    s_cond  = cond_entropy(sol.q)
    s_prior = (-log(1 - sol.q) - sol.q) / (2γ)
    return (; sol.γ, sol.q, sol.q̂, sol.I,
              s_cond, s_prior, s = s_cond + s_prior,
              sol.iters, sol.converged)
end

"""
    span(γs; q0=0.5, verb=1, kwargs...)

Sweep `entropy_density` over a list of loads `γs`, warm-starting `q` from the
previous solution. Returns a `Vector{NamedTuple}`.
"""
function span(γs; q0 = 0.5, verb = 1, kwargs...)
    rows = NamedTuple[]
    q = q0
    for γ in γs
        res = entropy_density(γ; q0 = q, verb = max(verb - 1, 0), kwargs...)
        push!(rows, res)
        q = res.q
        verb > 0 && @printf("γ=%8.4f  q=%.6f  q̂=%.4e  I=%9.4f  s=%.6f  (cond=%.6f, prior=%+.6f)\n",
                            res.γ, res.q, res.q̂, res.I, res.s, res.s_cond, res.s_prior)
    end
    return rows
end

end # module
