"""Theoretical plots for the clamped Hopfield model.

Produces four figures, all saved to notes/plots/:

  hopfield_m_vs_t_beta{B}.png       -- finite-β RS retrieval branch, m vs t
  hopfield_phase_diagram_beta{B}.png -- finite-β RS phase diagram (α, t)
  hopfield_T0_m_vs_t.png            -- T=0 retrieval branch, m vs t
  hopfield_T0_phase_diagram.png     -- T=0 phase diagram with analytical spinodal

    uv run python src-hopfield/plot_hopfield.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hopfield_saddle_point import (
    T_STAR_T0,
    TS as TS_THEORY,
    load_or_compute_T0,
    solve_saddle,
    solve_saddle_newton,
    solve_T0,
    spinodal_alpha_T0,
)

PLOTS = Path(__file__).resolve().parents[1] / "notes" / "plots"

# 12 visually distinct colours (tab20 dark palette + two light partners)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


# ---------------------------------------------------------------------------
# finite-β: m vs t
# ---------------------------------------------------------------------------

BETA = 10.0
ALPHAS_MVT = [0.05, 0.10, 0.138, 0.15, 0.175, 0.20, 0.25]
TS_MVT = np.linspace(0.05, 1.00, 96)
N_QUAD = 500


def _sweep_t_beta(alpha: float, beta: float) -> np.ndarray:
    """Retrieval-branch m(t) at finite β via warm-started Newton."""
    ms = np.full(len(TS_MVT), np.nan)
    m, q = 0.999, 0.999
    for i, t in enumerate(TS_MVT):
        try:
            sol = solve_saddle_newton(alpha, t, beta, m0=m, q0=q,
                                      n_quad=N_QUAD, tol=1e-12)
        except ValueError:
            m, q = 0.999, 0.999
            continue
        if sol.converged and sol.delta < 1e-8 and sol.m > 0.1:
            ms[i] = sol.m
            m, q = sol.m, sol.q
        else:
            m, q = 0.999, 0.999
    return ms


def plot_m_vs_t_beta() -> None:
    out = PLOTS / f"hopfield_m_vs_t_beta{int(BETA)}.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, alpha in enumerate(ALPHAS_MVT):
        print(f"  beta={BETA} alpha={alpha} ...")
        ms = _sweep_t_beta(alpha, BETA)
        color = COLORS[k % len(COLORS)]
        ax.plot(TS_MVT, ms, "o-", ms=2.5, lw=1.4, color=color,
                label=rf"$\alpha={alpha:g}$")
    ax.set_xlabel(r"$t$ (fraction of free spins)")
    ax.set_ylabel(r"retrieval magnetisation $m$")
    ax.set_title(rf"RS ($\beta={BETA:g}$)")
    ax.set_xlim(TS_MVT.min(), TS_MVT.max()); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# finite-β: phase diagram
# ---------------------------------------------------------------------------

M_JUMP = 0.5


def _magnetisation_grid_beta(ts, alphas, beta) -> np.ndarray:
    M = np.full((len(ts), len(alphas)), np.nan)
    for i, t in enumerate(ts):
        m, q = 0.999, 0.999
        for j, a in enumerate(alphas):
            try:
                sol = solve_saddle(a, t, beta, m0=m, q0=q,
                                   damping=0.5, max_iter=5000, tol=1e-8)
                M[i, j] = sol.m
                if sol.converged and sol.m > 0.1:
                    m, q = sol.m, sol.q
                else:
                    m, q = 0.999, 0.999
            except ValueError:
                M[i, j] = 0.0
                m, q = 0.999, 0.999
    return M


def _spinodal_from_grid(M, alphas) -> np.ndarray:
    alpha_c = np.full(M.shape[0], np.nan)
    for i, row in enumerate(M):
        above = row > M_JUMP
        if above.all() or not above.any() or not above[0]:
            continue
        j = int(np.argmax(~above))
        alpha_c[i] = 0.5 * (alphas[j - 1] + alphas[j])
    return alpha_c


def plot_phase_diagram_beta() -> None:
    out = PLOTS / f"hopfield_phase_diagram_beta{int(BETA)}.png"
    ts = np.linspace(0.40, 1.00, 61)
    alphas = np.linspace(0.0, 0.22, 111)
    print(f"  beta={BETA}: computing {len(ts)}x{len(alphas)} grid ...")
    M = _magnetisation_grid_beta(ts, alphas, BETA)
    alpha_c = _spinodal_from_grid(M, alphas)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.pcolormesh(alphas, ts, M, cmap="viridis", vmin=0.0, vmax=1.0,
                       shading="auto")
    fig.colorbar(im, ax=ax).set_label(r"retrieval magnetisation $m$")
    mask = np.isfinite(alpha_c)
    if mask.any():
        ax.plot(alpha_c[mask], ts[mask], "-", color="white", lw=2.5)
        ax.plot(alpha_c[mask], ts[mask], "--", color="crimson", lw=1.4,
                label=r"spinodal $\alpha_c(t)$")
    ax.set_xlabel(r"$\alpha = P/N$"); ax.set_ylabel(r"$t$ (fraction of free spins)")
    ax.set_title(rf"RS phase diagram ($\beta={BETA:g}$)")
    ax.set_xlim(alphas.min(), alphas.max()); ax.set_ylim(ts.min(), ts.max())
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")

    print("\nspinodal alpha_c(t):")
    for t, ac in zip(ts, alpha_c):
        print(f"  t={t:.3f}  alpha_c={'  -  ' if not np.isfinite(ac) else f'{ac:.4f}'}")


# ---------------------------------------------------------------------------
# T=0: m vs t
# ---------------------------------------------------------------------------

ALPHAS_T0 = [0.10, 0.138, 0.15, 0.16, 0.175, 0.20, 0.25,
             0.30, 0.40, 0.50, 0.75, 1.00]


def plot_m_vs_t_T0(ms_high: np.ndarray, ms_low: np.ndarray) -> None:
    out = PLOTS / "hopfield_T0_m_vs_t.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, alpha in enumerate(ALPHAS_T0):
        color = COLORS[k % len(COLORS)]
        ax.plot(TS_THEORY, ms_high[k], "-", lw=1.6, color=color,
                label=rf"$\alpha={alpha:g}$")
        uninformed = np.where(np.abs(ms_high[k] - ms_low[k]) > 1e-3, ms_low[k], np.nan)
        ax.plot(TS_THEORY, uninformed, "--", lw=1.0, color=color, alpha=0.55)
    ax.set_xlabel(r"$t$ (fraction of free spins)")
    ax.set_ylabel(r"retrieval magnetisation $m$")
    ax.set_title(r"RS ($T=0$)")
    ax.set_xlim(TS_THEORY.min(), TS_THEORY.max()); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# T=0: phase diagram
# ---------------------------------------------------------------------------

def plot_phase_diagram_T0() -> None:
    out = PLOTS / "hopfield_T0_phase_diagram.png"
    ts = np.linspace(0.05, 1.00, 381)
    alphas = np.linspace(0.002, 1.00, 500)
    print(f"  T=0: computing {len(ts)}x{len(alphas)} grid ...")
    M = np.full((len(ts), len(alphas)), np.nan)
    for i, t in enumerate(ts):
        for j, a in enumerate(alphas):
            sol = solve_T0(a, t)
            if sol.converged:
                M[i, j] = sol.m

    print("  T=0: computing analytical spinodal ...")
    ts_line = np.linspace(T_STAR_T0, 1.00, 200)
    alpha_c = np.array([spinodal_alpha_T0(t) for t in ts_line])

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.pcolormesh(alphas, ts, M, cmap="viridis", vmin=0.0, vmax=1.0,
                       shading="auto")
    fig.colorbar(im, ax=ax).set_label(r"retrieval magnetisation $m$")
    mask = np.isfinite(alpha_c)
    ax.plot(alpha_c[mask], ts_line[mask], "-", color="white", lw=2.6)
    ax.plot(alpha_c[mask], ts_line[mask], "--", color="crimson", lw=1.4,
            label=r"spinodal $\alpha_c(t)$")
    ac_star, t_star = alpha_c[mask][0], ts_line[mask][0]
    ax.plot(ac_star, t_star, "o", ms=10, color="white", zorder=5)
    ax.plot(ac_star, t_star, "o", ms=7, color="orange", zorder=6,
            label=rf"$(\alpha_c^*,\,t^*) \approx ({ac_star:.3f},\,{t_star:.3f})$")
    ax.set_xlabel(r"$\alpha = P/N$"); ax.set_ylabel(r"$t$ (fraction of free spins)")
    ax.set_title(r"RS phase diagram ($T=0$)")
    ax.set_xlim(alphas.min(), alphas.max()); ax.set_ylim(ts.min(), ts.max())
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")


# ---------------------------------------------------------------------------

def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    print("=== T=0: theory curves ===")
    ms_high, ms_low = load_or_compute_T0(ALPHAS_T0, TS_THEORY)
    print("=== finite-beta: m vs t ===")
    plot_m_vs_t_beta()
    print("=== finite-beta: phase diagram ===")
    plot_phase_diagram_beta()
    print("=== T=0: m vs t ===")
    plot_m_vs_t_T0(ms_high, ms_low)
    print("=== T=0: phase diagram ===")
    plot_phase_diagram_T0()


if __name__ == "__main__":
    main()
