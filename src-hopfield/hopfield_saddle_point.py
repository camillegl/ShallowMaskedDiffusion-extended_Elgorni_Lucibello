"""RS saddle-point equations and T=0 theory curves for the clamped Hopfield model.

Implements the equations derived in notes/notes_hopfield.typ.
A fraction `1 - t` of the spins is clamped to the first pattern; the free
fraction `t` obeys the RS equations

    m = <tanh(beta(M + sqrt(alpha r) z))>_z
    q = <tanh^2(beta(M + sqrt(alpha r) z))>_z
    r = qtilde / (1 - beta(1 - qtilde))^2

with M = (1-t) + t m and qtilde = (1-t) + t q.

Run as a script for a demo sweep:
    uv run python src-hopfield/hopfield_saddle_point.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import erf, roots_hermitenorm

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

def gauss_hermite(n: int = 200):
    z, w = roots_hermitenorm(n)
    return z, w / np.sqrt(2 * np.pi)


def _logcosh(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return a + np.log1p(np.exp(-2.0 * a))


# ---------------------------------------------------------------------------
# Finite-temperature RS saddle point
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    m: float
    q: float
    r: float
    M: float
    qtilde: float
    iterations: int
    converged: bool
    delta: float


def solve_saddle(
    alpha: float,
    t: float,
    beta: float,
    *,
    m0: float = 0.9,
    q0: float = 0.9,
    n_quad: int = 200,
    tol: float = 1e-10,
    max_iter: int = 20_000,
    damping: float = 0.5,
) -> Solution:
    """Fixed-point iteration for the RS saddle-point equations.

    `damping` is the weight of the new trial (1.0 = no damping, 0.0 = frozen).
    Initial condition `(m0, q0) ≈ (1, 1)` selects the retrieval branch;
    `(0, 0)` selects the paramagnetic / spin-glass branch.
    """
    z, w = gauss_hermite(n_quad)
    m, q = m0, q0
    delta = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        M = (1.0 - t) + t * m
        qt = (1.0 - t) + t * q
        denom = 1.0 - beta * (1.0 - qt)
        if denom <= 0.0:
            raise ValueError(
                f"replica kernel singular: 1 - beta(1-qtilde) = {denom:.3e} "
                f"(alpha={alpha}, t={t}, beta={beta})"
            )
        r = qt / denom**2
        arg = beta * (M + np.sqrt(alpha * r) * z)
        th = np.tanh(arg)
        m_trial = float(w @ th)
        q_trial = float(w @ (th * th))
        delta = max(abs(m_trial - m), abs(q_trial - q))
        m = (1.0 - damping) * m + damping * m_trial
        q = (1.0 - damping) * q + damping * q_trial
        if delta < tol:
            break

    M = (1.0 - t) + t * m
    qt = (1.0 - t) + t * q
    r = qt / (1.0 - beta * (1.0 - qt)) ** 2
    return Solution(
        m=m, q=q, r=r, M=M, qtilde=qt,
        iterations=it, converged=delta < tol, delta=delta,
    )


def solve_saddle_newton(
    alpha: float,
    t: float,
    beta: float,
    *,
    m0: float = 0.9,
    q0: float = 0.9,
    n_quad: int = 400,
    tol: float = 1e-12,
) -> Solution:
    """Scipy hybrd (Powell) root finder on F(m, q) = (trial_m - m, trial_q - q).

    Near the spinodal the fixed-point Jacobian has an eigenvalue approaching 1,
    making the Picard iteration of `solve_saddle` crawl and leaving residual
    wiggles in parameter sweeps. Newton/Powell converges quadratically.
    """
    z, w = gauss_hermite(n_quad)

    def residual(mq):
        m, q = mq
        M = (1.0 - t) + t * m
        qt = (1.0 - t) + t * q
        denom = 1.0 - beta * (1.0 - qt)
        if denom <= 0.0:
            return np.array([1e3, 1e3])
        r = qt / (denom * denom)
        arg = beta * (M + np.sqrt(alpha * r) * z)
        th = np.tanh(arg)
        return np.array([float(w @ th) - m, float(w @ (th * th)) - q])

    res = root(residual, [m0, q0], method="hybr",
               options={"xtol": tol, "maxfev": 2000})
    m, q = float(res.x[0]), float(res.x[1])
    M = (1.0 - t) + t * m
    qt = (1.0 - t) + t * q
    denom = 1.0 - beta * (1.0 - qt)
    if denom <= 0.0:
        raise ValueError(
            f"replica kernel singular: 1 - beta(1-qtilde) = {denom:.3e} "
            f"(alpha={alpha}, t={t}, beta={beta})"
        )
    r = qt / (denom * denom)
    delta = float(np.max(np.abs(res.fun)))
    return Solution(m=m, q=q, r=r, M=M, qtilde=qt,
                    iterations=int(res.nfev), converged=bool(res.success) and delta < 1e-6,
                    delta=delta)


def free_energy(
    sol: Solution, alpha: float, t: float, beta: float, *, n_quad: int = 200
) -> float:
    """RS free energy per spin, f_RS (not -beta f_RS)."""
    z, w = gauss_hermite(n_quad)
    M, qt = sol.M, sol.qtilde
    arg = beta * (M + np.sqrt(alpha * sol.r) * z)
    mbf = 0.5 * beta * M * M
    mbf -= 0.5 * alpha * beta * beta * t * sol.r * (1.0 - sol.q)
    denom = 1.0 - beta * (1.0 - qt)
    mbf += 0.5 * alpha * (beta * qt / denom - np.log(denom))
    mbf += t * float(w @ _logcosh(arg))
    return -mbf / beta


def at_stability(
    sol: Solution, alpha: float, t: float, beta: float, *, n_quad: int = 200
) -> float:
    """Replicon eigenvalue indicator. RS is stable when this is negative.

    Returns alpha t^2 <sech^4(beta(M + sqrt(alpha r) z))> - (1-beta(1-qtilde))^2.
    """
    z, w = gauss_hermite(n_quad)
    arg = beta * (sol.M + np.sqrt(alpha * sol.r) * z)
    sech4 = np.cosh(arg) ** (-4)
    lhs = alpha * t * t * float(w @ sech4)
    rhs = (1.0 - beta * (1.0 - sol.qtilde)) ** 2
    return lhs - rhs


# ---------------------------------------------------------------------------
# T=0 analytics
# ---------------------------------------------------------------------------

_SQRT_PI = float(np.sqrt(np.pi))


def g_T0(y: np.ndarray | float) -> np.ndarray | float:
    """g(y) = erf(y) - (2y/sqrt(pi)) exp(-y^2), see notes eq:T0."""
    return erf(y) - (2.0 * y / _SQRT_PI) * np.exp(-(y ** 2))


def g_prime_T0(y: np.ndarray | float) -> np.ndarray | float:
    return (4.0 * y * y / _SQRT_PI) * np.exp(-(y ** 2))


def h_T0(y: np.ndarray | float) -> np.ndarray | float:
    """h(y) = y g'(y) - g(y); the spinodal condition is h(y*) = (1-t)/t."""
    return (2.0 * y / _SQRT_PI) * (2.0 * y * y + 1.0) * np.exp(-(y ** 2)) - erf(y)


H_MAX_T0 = float(6.0 / (np.e * _SQRT_PI) - erf(1.0))  # ≈ 0.4027
T_STAR_T0 = 1.0 / (1.0 + H_MAX_T0)                    # ≈ 0.7129


@dataclass
class SolutionT0:
    y: float
    m: float
    alpha: float
    t: float
    converged: bool


def solve_T0(
    alpha: float, t: float, *,
    branch: str = "high",
    y_max: float = 30.0, n_scan: int = 4000,
) -> SolutionT0:
    """Solve y*sqrt(2 alpha) = (1-t) + t g(y).

    branch="high"  returns the largest-y  (retrieval / informed) root.
    branch="low"   returns the smallest-y (uninformed) root.
    Returns NaN when no root exists in [0, y_max].
    """
    if t <= 0.0:
        return SolutionT0(y=np.nan, m=np.nan, alpha=alpha, t=t, converged=False)

    sa = np.sqrt(2.0 * alpha)

    def F(y):
        return y * sa - (1.0 - t) - t * g_T0(y)

    ys = np.linspace(1e-8, y_max, n_scan)
    Fs = F(ys)
    sign_changes = np.where(np.diff(np.sign(Fs)) != 0)[0]
    if len(sign_changes) == 0:
        return SolutionT0(y=np.nan, m=np.nan, alpha=alpha, t=t, converged=False)

    i = int(sign_changes[-1] if branch == "high" else sign_changes[0])
    y_star = brentq(F, ys[i], ys[i + 1], xtol=1e-12, rtol=1e-12)
    return SolutionT0(y=float(y_star), m=float(erf(y_star)),
                      alpha=alpha, t=t, converged=True)


def spinodal_alpha_T0(t: float) -> float:
    """Retrieval-spinodal alpha_c(t) at T=0; NaN if t < t* (no spinodal)."""
    if t < T_STAR_T0 or t > 1.0:
        return float("nan")
    target = (1.0 - t) / t
    if target > H_MAX_T0:
        return float("nan")
    y_lo, y_hi = 1.0, 20.0
    y_star = brentq(lambda y: h_T0(y) - target, y_lo, y_hi,
                    xtol=1e-12, rtol=1e-12)
    return float(8.0 * t * t * y_star ** 4 / np.pi * np.exp(-2.0 * y_star ** 2))


# ---------------------------------------------------------------------------
# T=0 theory curve cache
# ---------------------------------------------------------------------------

TS = np.linspace(0.02, 1.00, 2000)
_DATA_FILE = DATA_DIR / "hopfield_T0_theory.npz"


def _alpha_indices(stored: list[float], want: list[float]) -> list[int] | None:
    idx = []
    for a in want:
        matches = [i for i, s in enumerate(stored) if abs(s - a) < 1e-9]
        if not matches:
            return None
        idx.append(matches[0])
    return idx


def compute_T0(alphas: list[float], ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute T=0 theory curves for both branches.

    Returns (ms_high, ms_low), each shape (A, T).
    ms_low[k, i] is NaN where only one root exists (branches coincide).
    """
    A, T = len(alphas), len(ts)
    ms_high = np.full((A, T), np.nan)
    ms_low  = np.full((A, T), np.nan)
    for k, alpha in enumerate(alphas):
        print(f"  alpha={alpha:g} ...")
        for i, t in enumerate(ts):
            hi = solve_T0(alpha, float(t), branch="high")
            lo = solve_T0(alpha, float(t), branch="low")
            if hi.converged:
                ms_high[k, i] = hi.m
            if lo.converged:
                ms_low[k, i] = lo.m
    return ms_high, ms_low


def load_or_compute_T0(
    alphas: list[float], ts: np.ndarray = TS
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ms_high, ms_low) from cache if grid matches, else recompute and save."""
    if _DATA_FILE.exists():
        npz = np.load(_DATA_FILE)
        stored_alphas = npz["alphas"].tolist()
        stored_ts     = npz["ts"]
        if np.allclose(stored_ts, ts, atol=0.0, rtol=1e-10):
            idx = _alpha_indices(stored_alphas, alphas)
            if idx is not None:
                print(f"loading T=0 theory from {_DATA_FILE}")
                return npz["ms_high"][idx], npz["ms_low"][idx]
        print("T=0 theory cache mismatch — recomputing")

    print(f"computing T=0 theory curves ({len(alphas)} alphas × {len(ts)} t-points) ...")
    ms_high, ms_low = compute_T0(alphas, ts)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_DATA_FILE, ts=ts, alphas=np.array(alphas),
             ms_high=ms_high, ms_low=ms_low)
    print(f"saved to {_DATA_FILE}")
    return ms_high, ms_low


# ---------------------------------------------------------------------------

def _demo() -> None:
    print("retrieval branch, single points")
    for alpha, t, beta in [(0.05, 1.0, 5.0), (0.10, 1.0, 10.0), (0.10, 0.9, 20.0)]:
        sol = solve_saddle(alpha, t, beta, m0=0.99, q0=0.99)
        f = free_energy(sol, alpha, t, beta)
        at = at_stability(sol, alpha, t, beta)
        print(
            f"  alpha={alpha:.3f} t={t:.2f} beta={beta:5.1f} -> "
            f"m={sol.m:.4f} q={sol.q:.4f} r={sol.r:.4f} f={f:+.4f} "
            f"AT={at:+.3e} iters={sol.iterations}"
        )

    print("\nalpha sweep at t=1.0, beta=10.0 (warm-started from retrieval)")
    m, q = 0.999, 0.999
    for alpha in np.linspace(0.02, 0.16, 8):
        sol = solve_saddle(alpha, 1.0, 10.0, m0=m, q0=q)
        m, q = sol.m, sol.q
        print(f"  alpha={alpha:.3f} -> m={m:.4f} q={q:.4f} conv={sol.converged}")

    print("\nt sweep at alpha=0.10, beta=10.0 (warm-started from retrieval)")
    m, q = 0.999, 0.999
    for t in np.linspace(1.0, 0.5, 11):
        sol = solve_saddle(0.10, t, 10.0, m0=m, q0=q)
        m, q = sol.m, sol.q
        print(f"  t={t:.2f} -> m={m:.4f} q={q:.4f} conv={sol.converged}")


if __name__ == "__main__":
    _demo()
