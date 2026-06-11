"""MCMC validation for the clamped Hopfield model.

Two experiments, both over the same (alpha, t) grid:

  T=0    async Glauber dynamics, run to convergence (max MAX_SWEEPS sweeps).
         Also reports sweeps-to-convergence plots.
  T=0.01 Metropolis-Hastings, exactly N_SWEEPS_T001 sweeps.

Two initialisations per experiment: uniform random ("random") and
informed ("pattern" = xi^1).  Theory lines always use the T=0 RS
saddle-point prediction.

    uv run python src-hopfield/mcmc_hopfield.py
"""

from __future__ import annotations

from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numba
import numpy as np

from hopfield_saddle_point import TS as TS_THEORY, load_or_compute_T0


# ---------------------------------------------------------------------------
# shared constants
# ---------------------------------------------------------------------------

N          = 20_000
N_SAMPLES  = 10
SEED       = 0

ALPHAS = [0.10, 0.138, 0.15, 0.16, 0.175, 0.20, 0.25,
          0.30, 0.40, 0.50, 0.75, 1.00]
TS = np.linspace(0.05, 1.00, 20)

MAX_SWEEPS_T0   = 500    # upper bound for T=0 convergence
N_SWEEPS_T001   = 400    # exact sweeps for T=0.01
BETA_T001       = 100.0  # 1 / 0.01

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATA_T0   = DATA_DIR / f"hopfield_T0_mcmc_N{N}_S{N_SAMPLES}_seed{SEED}.npz"
DATA_T001 = DATA_DIR / f"hopfield_T001_mcmc_N{N}_S{N_SAMPLES}_seed{SEED}.npz"

PLOTS = ROOT / "notes" / "plots"

# 12 visually distinct colours (tab20 dark palette + two light partners)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _set_numba_seed(seed: int) -> None:
    np.random.seed(seed)


@numba.njit(cache=True)
def _glauber_loop(
    J: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    free_idx: np.ndarray,
    xi1: np.ndarray,
    max_sweeps: int,
) -> tuple:
    """T=0 async Glauber.  Returns (m, n_sweeps_until_convergence)."""
    N       = J.shape[0]
    n_free  = len(free_idx)
    n_sweeps = 0
    for _ in range(max_sweeps):
        n_sweeps += 1
        for i in range(n_free - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = free_idx[i]; free_idx[i] = free_idx[j]; free_idx[j] = tmp
        flipped = 0
        for k in range(n_free):
            i = free_idx[k]
            hi = h[i]
            new_s = 1.0 if hi > 0.0 else (-1.0 if hi < 0.0 else x[i])
            delta = new_s - x[i]
            if delta != 0.0:
                for j2 in range(N):
                    h[j2] += J[i, j2] * delta
                x[i] = new_s
                flipped += 1
        if flipped == 0:
            break
    m = 0.0
    for k in range(n_free):
        m += x[free_idx[k]] * xi1[free_idx[k]]
    return (m / n_free if n_free > 0 else 1.0, n_sweeps)


@numba.njit(cache=True)
def _mh_loop(
    J: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    free_idx: np.ndarray,
    xi1: np.ndarray,
    n_sweeps: int,
    beta: float,
) -> float:
    """MH dynamics at inverse temperature beta.  Runs exactly n_sweeps.  Returns m."""
    N      = J.shape[0]
    n_free = len(free_idx)
    for _ in range(n_sweeps):
        for i in range(n_free - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = free_idx[i]; free_idx[i] = free_idx[j]; free_idx[j] = tmp
        for k in range(n_free):
            i = free_idx[k]
            delta_E = 2.0 * x[i] * h[i]
            if delta_E <= 0.0 or np.random.random() < np.exp(-beta * delta_E):
                delta = -2.0 * x[i]
                for j2 in range(N):
                    h[j2] += J[i, j2] * delta
                x[i] = -x[i]
    m = 0.0
    for k in range(n_free):
        m += x[free_idx[k]] * xi1[free_idx[k]]
    return m / n_free if n_free > 0 else 1.0


def _warmup_jit() -> None:
    print("warming up numba JIT ...")
    _J  = np.zeros((4, 4), dtype=np.float32)
    _x  = np.ones(4, dtype=np.float32)
    _h  = np.zeros(4, dtype=np.float32)
    _fi = np.array([0, 1], dtype=np.int64)
    _xi = np.ones(4, dtype=np.float32)
    _set_numba_seed(0)
    _glauber_loop(_J, _x.copy(), _h.copy(), _fi.copy(), _xi, 1)
    _mh_loop(_J, _x.copy(), _h.copy(), _fi.copy(), _xi, 1, BETA_T001)
    print("JIT ready.")


# ---------------------------------------------------------------------------
# shared initialisation helper
# ---------------------------------------------------------------------------

def _init_state(
    J: np.ndarray,
    xi1: np.ndarray,
    t: float,
    rng: np.random.Generator,
    init: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, h, free_idx, clamped_idx) for a given t and init mode."""
    N_loc  = J.shape[0]
    n_free = int(round(t * N_loc))
    idx          = rng.permutation(N_loc)
    free_idx     = idx[:n_free].astype(np.int64)
    clamped_idx  = idx[n_free:]
    x = (rng.integers(0, 2, N_loc).astype(np.float32) * 2.0 - 1.0
         if init == "random" else xi1.copy())
    x[clamped_idx] = xi1[clamped_idx]
    h = (J @ x).astype(np.float32)
    return x, h, free_idx, clamped_idx


# ---------------------------------------------------------------------------
# T=0 simulation
# ---------------------------------------------------------------------------

def _run_T0_one(J, xi1, t, rng, *, init) -> tuple[float, int]:
    N_loc  = J.shape[0]
    n_free = int(round(t * N_loc))
    if n_free == 0:
        return 1.0, 0
    x, h, free_idx, _ = _init_state(J, xi1, t, rng, init)
    _set_numba_seed(int(rng.integers(0, 2**31)))
    m, sw = _glauber_loop(J, x, h, free_idx, xi1, MAX_SWEEPS_T0)
    return float(m), int(sw)


def _sweep_alpha_T0(alpha, ts, n_samples, rng):
    P  = max(1, int(round(alpha * N)))
    mr = np.zeros((len(ts), n_samples))
    mp = np.zeros((len(ts), n_samples))
    sr = np.zeros((len(ts), n_samples))
    sp = np.zeros((len(ts), n_samples))
    for s in range(n_samples):
        xi  = rng.integers(0, 2, (N, P)).astype(np.float32) * 2.0 - 1.0
        J   = (xi @ xi.T / N).astype(np.float32)
        np.fill_diagonal(J, 0.0)
        xi1 = xi[:, 0].copy()
        for it, t in enumerate(ts):
            mr[it, s], sr[it, s] = _run_T0_one(J, xi1, float(t), rng, init="random")
            mp[it, s], sp[it, s] = _run_T0_one(J, xi1, float(t), rng, init="pattern")
    return mr, mp, sr, sp


def _run_simulation_T0(rng):
    A, T = len(ALPHAS), len(TS)
    mr = np.zeros((A, T, N_SAMPLES)); mp = np.zeros_like(mr)
    sr = np.zeros((A, T, N_SAMPLES)); sp = np.zeros_like(sr)
    for k, alpha in enumerate(ALPHAS):
        t0 = time()
        mr[k], mp[k], sr[k], sp[k] = _sweep_alpha_T0(alpha, TS, N_SAMPLES, rng)
        mid = len(TS) // 2
        print(f"[T=0    alpha={alpha:.3f}] {time()-t0:5.1f}s  "
              f"rand m={mr[k].mean(1)[mid]:+.3f} sw={sr[k].mean(1)[mid]:.1f}  "
              f"pat  m={mp[k].mean(1)[mid]:+.3f} sw={sp[k].mean(1)[mid]:.1f}")
    return mr, mp, sr, sp


def _alpha_indices(stored: list[float], want: list[float]) -> list[int] | None:
    """Return row indices to extract `want` from `stored`, or None if not a superset."""
    idx = []
    for a in want:
        matches = [i for i, s in enumerate(stored) if abs(s - a) < 1e-9]
        if not matches:
            return None
        idx.append(matches[0])
    return idx


def load_or_run_T0():
    if DATA_T0.exists():
        npz = np.load(DATA_T0)
        stored = npz["alphas"].tolist()
        idx = _alpha_indices(stored, ALPHAS)
        if idx is not None and np.allclose(npz["ts"], TS) and "sweeps_random" in npz:
            print(f"loading T=0 cache from {DATA_T0}")
            return (npz["data_random"][idx], npz["data_pattern"][idx],
                    npz["sweeps_random"][idx], npz["sweeps_pattern"][idx])
        print("T=0 cache mismatch — re-running")
    _warmup_jit()
    rng = np.random.default_rng(SEED)
    mr, mp, sr, sp = _run_simulation_T0(rng)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_T0, data_random=mr, data_pattern=mp,
             sweeps_random=sr, sweeps_pattern=sp,
             ts=TS, alphas=np.array(ALPHAS))
    print(f"saved T=0 data to {DATA_T0}")
    return mr, mp, sr, sp


# ---------------------------------------------------------------------------
# T=0.01 simulation
# ---------------------------------------------------------------------------

def _run_T001_one(J, xi1, t, rng, *, init) -> float:
    N_loc  = J.shape[0]
    n_free = int(round(t * N_loc))
    if n_free == 0:
        return 1.0
    x, h, free_idx, _ = _init_state(J, xi1, t, rng, init)
    _set_numba_seed(int(rng.integers(0, 2**31)))
    return float(_mh_loop(J, x, h, free_idx, xi1, N_SWEEPS_T001, BETA_T001))


def _sweep_alpha_T001(alpha, ts, n_samples, rng):
    P  = max(1, int(round(alpha * N)))
    mr = np.zeros((len(ts), n_samples))
    mp = np.zeros((len(ts), n_samples))
    for s in range(n_samples):
        xi  = rng.integers(0, 2, (N, P)).astype(np.float32) * 2.0 - 1.0
        J   = (xi @ xi.T / N).astype(np.float32)
        np.fill_diagonal(J, 0.0)
        xi1 = xi[:, 0].copy()
        for it, t in enumerate(ts):
            mr[it, s] = _run_T001_one(J, xi1, float(t), rng, init="random")
            mp[it, s] = _run_T001_one(J, xi1, float(t), rng, init="pattern")
    return mr, mp


def _run_simulation_T001(rng):
    A, T = len(ALPHAS), len(TS)
    mr = np.zeros((A, T, N_SAMPLES)); mp = np.zeros_like(mr)
    for k, alpha in enumerate(ALPHAS):
        t0 = time()
        mr[k], mp[k] = _sweep_alpha_T001(alpha, TS, N_SAMPLES, rng)
        mid = len(TS) // 2
        print(f"[T=0.01 alpha={alpha:.3f}] {time()-t0:5.1f}s  "
              f"rand m={mr[k].mean(1)[mid]:+.3f}  pat m={mp[k].mean(1)[mid]:+.3f}")
    return mr, mp


def load_or_run_T001():
    if DATA_T001.exists():
        npz = np.load(DATA_T001)
        stored = npz["alphas"].tolist()
        idx = _alpha_indices(stored, ALPHAS)
        if idx is not None and np.allclose(npz["ts"], TS):
            print(f"loading T=0.01 cache from {DATA_T001}")
            return npz["data_random"][idx], npz["data_pattern"][idx]
        print("T=0.01 cache mismatch — re-running")
    _warmup_jit()
    rng = np.random.default_rng(SEED)
    mr, mp = _run_simulation_T001(rng)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_T001, data_random=mr, data_pattern=mp,
             ts=TS, alphas=np.array(ALPHAS))
    print(f"saved T=0.01 data to {DATA_T001}")
    return mr, mp


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _make_mag_plot(
    data: np.ndarray,
    ms_high: np.ndarray,
    ms_low: np.ndarray,
    fmt: str,
    title: str,
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for k, alpha in enumerate(ALPHAS):
        color = COLORS[k % len(COLORS)]
        ax.plot(TS_THEORY, ms_high[k], "-", color=color, lw=1.3, alpha=0.75,
                label=rf"$\alpha={alpha:g}$")
        low = np.where(np.abs(ms_high[k] - ms_low[k]) > 1e-3, ms_low[k], np.nan)
        ax.plot(TS_THEORY, low, "--", color=color, lw=1.0, alpha=0.55, zorder=1)
        means = data[k].mean(axis=1)
        sems  = data[k].std(axis=1) / np.sqrt(N_SAMPLES)
        ax.errorbar(TS, means, yerr=sems, fmt=fmt, ms=4, color=color,
                    capsize=2, lw=0.8, mec="black", mew=0.4, zorder=2)
    ax.set_xlabel(r"$t$ (fraction of free spins)")
    ax.set_ylabel(r"retrieval magnetisation $m$")
    ax.set_title(title)
    ax.set_xlim(0, 1.02); ax.set_ylim(-0.05, 1.02)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")


def _make_sweeps_plot(
    sweeps: np.ndarray,
    fmt: str,
    title: str,
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for k, alpha in enumerate(ALPHAS):
        color = COLORS[k % len(COLORS)]
        means = sweeps[k].mean(axis=1)
        sems  = sweeps[k].std(axis=1) / np.sqrt(N_SAMPLES)
        ax.errorbar(TS, means, yerr=sems, fmt=fmt, ms=4, color=color,
                    capsize=2, lw=0.8, mec="black", mew=0.4,
                    label=rf"$\alpha={alpha:g}$")
    ax.set_xlabel(r"$t$ (fraction of free spins)")
    ax.set_ylabel("sweeps to convergence")
    ax.set_title(title)
    ax.set_xlim(0, 1.02); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")


# ---------------------------------------------------------------------------

def main() -> None:
    stem = f"$N={N}$, {N_SAMPLES} samples"

    ms_high, ms_low = load_or_compute_T0(ALPHAS, TS_THEORY)

    # T=0
    mr0, mp0, sr0, sp0 = load_or_run_T0()
    _make_mag_plot(mr0, ms_high, ms_low, "o", rf"$T=0$: uniform init ({stem})",
                   PLOTS / "hopfield_T0_m_vs_t_mcmc_random.png")
    _make_mag_plot(mp0, ms_high, ms_low, "^", rf"$T=0$: informed init ({stem})",
                   PLOTS / "hopfield_T0_m_vs_t_mcmc_pattern.png")
    _make_sweeps_plot(sr0, "o", rf"$T=0$: uniform init ({stem})",
                      PLOTS / "hopfield_T0_sweeps_mcmc_random.png")
    _make_sweeps_plot(sp0, "^", rf"$T=0$: informed init ({stem})",
                      PLOTS / "hopfield_T0_sweeps_mcmc_pattern.png")

    # T=0.01
    stem001 = f"{stem}, {N_SWEEPS_T001} sweeps"
    mr1, mp1 = load_or_run_T001()
    _make_mag_plot(mr1, ms_high, ms_low, "o", rf"$T=0.01$: uniform init ({stem001})",
                   PLOTS / "hopfield_T001_m_vs_t_mcmc_random.png")
    _make_mag_plot(mp1, ms_high, ms_low, "^", rf"$T=0.01$: informed init ({stem001})",
                   PLOTS / "hopfield_T001_m_vs_t_mcmc_pattern.png")


if __name__ == "__main__":
    main()
