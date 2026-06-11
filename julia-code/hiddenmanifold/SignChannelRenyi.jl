"""
    SignChannelRenyi

Replica-symmetric quenched Rényi entropy of order α = n+1 (with α > 0, α ≠ 1)
for the hidden-manifold output distribution under the noiseless sign channel
P_out(y|x) = δ_{y, sign(x)}.

The cross-block overlap vanishes (q₀* = 0) on the physical Rényi saddle for
the sign channel at every (γ, n) tested — this collapses the
five-dimensional system @eq:Q-inv-renyi + @eq:RS-renyi-conj of §3 to a
*single* scalar equation for q₁. The implementation here imposes q₀ = 0
from the start (§4.3 of notes/notes_hiddenmanifold.typ).

# Equations

  Φ_n(q₁) = log E_{t ~ N(0, ρ)} [ Φ(t)^(n+1) + Φ(-t)^(n+1) ],
                                               ρ := q₁/(1-q₁)        [eq:sign-Phin-q0]

  q̂₁* = q₁* / ((1-q₁*)(1+n q₁*))                                     [eq:sign-q1hat]

  Self-consistency:
    q₁ / ((1-q₁)(1+n q₁)) = (2γ)/(n(n+1)) · ∂_{q₁} Φ_n(q₁)           [eq:sign-saddle-renyi]

  Prior at saddle:
    S_prior*/m|_{m→0} = (n/2) log(1-q₁*) + (1/2) log(1 + n q₁*)      [eq:sign-prior-simple]

  Rényi entropy density:
    s_α = -log(1-q₁*)/(2γ) - log(1+n q₁*)/(2nγ)
          - (1/n) log E_t [ Φ(t)^(n+1) + Φ(-t)^(n+1) ]                [eq:sign-renyi-main]

# Saddle solver

`solve_renyi_saddle` runs a damped fixed-point on the inversion q̂₁ ↔ q₁ pair:
at each step we evaluate ∂_{q₁} Φ_n via central differences, read off
q̂₁ = (2γ)/(n(n+1)) · ∂_{q₁} Φ_n, and recover q₁ as the positive root of the
quadratic q̂₁ · n · q₁² + (1 - q̂₁(n-1)) q₁ - q̂₁ = 0.

The Shannon limit n = 0 is excluded — use SignChannel.solve_saddle.
"""
module SignChannelRenyi

using SpecialFunctions: erfc, erfcx, logerfc
using QuadGK
using Printf

export Phi_n, dPhi_n_dq1, q1_from_q1hat, q1hat_from_q1,
       solve_renyi_saddle, renyi_entropy_density, span_renyi

# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

const _SQRT2  = sqrt(2.0)
const _SQRT2π = sqrt(2π)
const _LOG2   = log(2.0)

@inline logΦ(t)   = logerfc(-t / _SQRT2) - _LOG2
@inline log1mΦ(t) = logerfc( t / _SQRT2) - _LOG2

"""
    log_F(t, n)

Stable evaluation of `log[Φ(t)^(n+1) + Φ(-t)^(n+1)]` via log-sum-exp.
"""
@inline function log_F(t, n)
    α  = n + 1
    a  = α * logΦ( t)
    b  = α * logΦ(-t)
    M  = max(a, b)
    return M + log1p(exp(min(a, b) - M))
end

# ---------------------------------------------------------------------------
# Gaussian quadrature against N(0, σ²); substitute t = σ u so the quadrature
# is always against a *standard* normal regardless of σ.
# ---------------------------------------------------------------------------

function gauss_int(f, σ; rtol = 1e-10, atol = 1e-12)
    σ < 1e-14 && return f(0.0)
    g = u -> f(σ * u) * exp(-u^2 / 2) / _SQRT2π
    val, _ = quadgk(g, -Inf, 0.0, Inf; rtol = rtol, atol = atol)
    return val
end

# ---------------------------------------------------------------------------
# Output integral Φ_n(q₁)   (q₀ = 0 specialization)
# ---------------------------------------------------------------------------

"""
    Phi_n(q1, n; rtol, atol)

`Φ_n(q₁) = log E_{t ~ N(0, q₁/(1-q₁))} [Φ(t)^(n+1) + Φ(-t)^(n+1)]`.

Sanity: `Φ_n(0) = -n·log 2`, `Φ_0 ≡ 0`.
"""
function Phi_n(q1, n; rtol = 1e-9, atol = 1e-11)
    0 ≤ q1 < 1 || throw(DomainError(q1, "need 0 ≤ q₁ < 1"))
    σ = sqrt(q1 / (1 - q1))
    Eval = gauss_int(t -> exp(log_F(t, n)), σ; rtol = rtol, atol = atol)
    return log(max(Eval, 1e-300))
end

"""
    dPhi_n_dq1(q1, n; h, kwargs...)

Central-difference derivative `∂Φ_n(q₁)/∂q₁`. Step `h` is auto-clipped to
stay inside `0 < q₁ < 1`; forward differences near `q₁ = 0`.
"""
function dPhi_n_dq1(q1, n; h = 1e-4, kwargs...)
    eps1 = max(min(h, q1 / 4, (1 - q1) / 4), 1e-8)
    if q1 < 2 * eps1
        Φ0  = Phi_n(q1,        n; kwargs...)
        Φ0p = Phi_n(q1 + eps1, n; kwargs...)
        return (Φ0p - Φ0) / eps1
    end
    Φp = Phi_n(q1 + eps1, n; kwargs...)
    Φm = Phi_n(q1 - eps1, n; kwargs...)
    return (Φp - Φm) / (2 * eps1)
end

# ---------------------------------------------------------------------------
# q̂₁ ↔ q₁ inversion (eq:sign-q1hat)
# ---------------------------------------------------------------------------

"""
    q1hat_from_q1(q1, n)

Forward map `q̂₁ = q₁ / ((1-q₁)(1+n q₁))` from @eq:sign-q1hat.
"""
@inline q1hat_from_q1(q1, n) = q1 / ((1 - q1) * (1 + n * q1))

"""
    q1_from_q1hat(q1hat, n)

Inverse of @eq:sign-q1hat: solves `q̂₁ · n · q₁² + (1 - q̂₁(n-1)) q₁ - q̂₁ = 0`
for the unique root in [0, 1). Returns `NaN` if no such root exists.

Two reductions for the degenerate cases:
  * `n = 0` (Shannon, excluded by the solver but allowed here):  q₁ = q̂₁/(1+q̂₁)
  * `q̂₁ = 0`: q₁ = 0
"""
function q1_from_q1hat(q1hat, n)
    abs(q1hat) < 1e-300 && return 0.0
    abs(n) < 1e-12 && return q1hat / (1 + q1hat)

    a = q1hat * n
    b = 1 - q1hat * (n - 1)
    c = -q1hat
    disc = b^2 - 4 * a * c        # = (1 - q̂₁(n-1))² + 4 n q̂₁²
    disc < 0 && return NaN

    sd = sqrt(disc)
    # The two roots; pick the one in [0, 1).
    r1 = (-b + sd) / (2a)
    r2 = (-b - sd) / (2a)
    for r in (r1, r2)
        (0 ≤ r < 1) && return r
    end
    return NaN
end

# ---------------------------------------------------------------------------
# Damped fixed-point solver for q₁
# ---------------------------------------------------------------------------

"""
    solve_renyi_saddle(γ, n; q1, ψ, atol, rtol, maxiters, h, verb, quad_kwargs)

Damped fixed-point iteration of @eq:sign-saddle-renyi at load γ = N/D and
replica index n (so α = n+1). Each step:

  1. evaluate `d1 = ∂_{q₁} Φ_n(q₁)`,
  2. set `q̂₁ = (2γ)/(n(n+1)) · d1`,                            [@eq:RS-renyi-conj]
  3. recover `q₁` from @eq:sign-q1hat by solving the quadratic.

Convergence: `|Δq₁| < atol` AND `|Δq̂₁| < atol + rtol·|q̂₁|`. Relative tolerance
on q̂₁ matters near the Hartley limit α → 0 where q̂₁ → ∞.

Returns `(; γ, n, α, q1, q1hat, iters, converged)`.

`n = 0` is the Shannon limit and is rejected — use `SignChannel.solve_saddle`.
"""
function solve_renyi_saddle(γ, n;
        q1 = 0.5, ψ = 0.5, atol = 1e-10, rtol = 1e-8,
        maxiters = 2000, h = 1e-4, verb = 0,
        quad_kwargs = (;))
    γ > 0 || throw(DomainError(γ, "γ must be > 0"))
    n != 0 || throw(DomainError(n, "n = 0 is the Shannon limit; use SignChannel"))

    q1 = clamp(q1, 1e-7, 1 - 1e-6)
    q1hat = q1hat_from_q1(q1, n)
    converged = false
    it = 0

    for outer it in 1:maxiters
        d1 = dPhi_n_dq1(q1, n; h = h, quad_kwargs...)
        q1hat_new = (2γ) / (n * (n + 1)) * d1

        q1_new = q1_from_q1hat(q1hat_new, n)
        if !isfinite(q1_new) || q1_new ≤ 0 || q1_new ≥ 1
            verb > 0 && @printf("  [✗] iter=%d  invalid q₁_new=%s  q̂₁=%.4e\n",
                                it, string(q1_new), q1hat_new)
            break
        end

        q1_new = clamp(q1_new, 1e-7, 1 - 1e-6)
        Δq1   = abs(q1_new - q1)
        Δq1h  = abs(q1hat_new - q1hat)

        q1    = (1 - ψ) * q1_new    + ψ * q1
        q1hat = (1 - ψ) * q1hat_new + ψ * q1hat

        verb > 1 && @printf("  it=%4d  q₁=%.10f  q̂₁=%.6e  Δq₁=%.2e Δq̂₁=%.2e\n",
                            it, q1, q1hat, Δq1, Δq1h)

        if Δq1 < atol && Δq1h < atol + rtol * abs(q1hat)
            converged = true
            break
        end
    end

    verb > 0 && println(converged ? "✓ converged" : "✗ failed",
                        @sprintf("  γ=%.4f n=%.4f q₁=%.8f iters=%d",
                                 γ, n, q1, it))

    return (; γ, n, α = n + 1, q1, q1hat, iters = it, converged)
end

# ---------------------------------------------------------------------------
# Final entropy density (eq:sign-renyi-main)
# ---------------------------------------------------------------------------

"""
    renyi_entropy_density(γ, n; kwargs...)

Replica-symmetric Rényi entropy density (nats) of order α = n+1 for the sign
channel at load γ. Splits the total as

    s_α = s_prior + s_out,

with
    s_prior = -log(1-q₁*)/(2γ) - log(1+n q₁*)/(2nγ),
    s_out   = -Φ_n(q₁*) / n.

Sanity: at γ → 0 the saddle gives q₁* → 0, s_prior → 0, and Φ_n(0) = -n log 2,
so s_out → log 2 = s_α (uninformative limit, all four Rényi orders agree).
"""
function renyi_entropy_density(γ, n; quad_kwargs = (;), kwargs...)
    sol = solve_renyi_saddle(γ, n; quad_kwargs = quad_kwargs, kwargs...)
    Φ   = Phi_n(sol.q1, n; quad_kwargs...)
    Spr = n / 2 * log(1 - sol.q1) + 1 / 2 * log(1 + n * sol.q1)
    s_out   = -Φ / n
    s_prior = -Spr / (n * γ)
    return (; sol.γ, sol.n, sol.α, sol.q1, sol.q1hat,
              Phi_n = Φ, S_prior = Spr,
              s_out, s_prior, s = s_out + s_prior,
              sol.iters, sol.converged)
end

"""
    span_renyi(γs, n; q1, verb, kwargs...)

Sweep `renyi_entropy_density` over a list of loads at fixed `n`, warm-starting
`q₁` from the previous solution. Returns a `Vector{NamedTuple}`.
"""
function span_renyi(γs, n; q1 = 0.5, verb = 1, kwargs...)
    rows = NamedTuple[]
    qq1 = q1
    for γ in γs
        res = renyi_entropy_density(γ, n;
                  q1 = qq1,
                  verb = max(verb - 1, 0), kwargs...)
        push!(rows, res)
        qq1 = res.q1
        verb > 0 && @printf("γ=%8.4f  n=%.3f  q₁=%.6f  Φ_n=%+.6f  s_α=%.6f  (out=%.6f, prior=%+.6f)%s\n",
                            res.γ, res.n, res.q1, res.Phi_n,
                            res.s, res.s_out, res.s_prior,
                            res.converged ? "" : "  [✗ unconverged]")
    end
    return rows
end

end # module
