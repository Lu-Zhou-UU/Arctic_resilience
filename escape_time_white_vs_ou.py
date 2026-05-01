"""
Extended Data Fig: Mean escape time τ(ΔF) for white noise vs Ornstein-Uhlenbeck (OU) forcing.
Shows that persistence (red noise) reduces effective escape times relative to white noise
for the same marginal variance σ², as stated in Methods (Stochastic extension).

Run from this directory or ensure asi_resilience_revised.py is on PYTHONPATH.

Extended Data Fig caption (draft):
  Mean escape time τ(ΔF) from the ice-covered minimum to the saddle in the reduced
  energy-balance model, for white noise (black) and Ornstein–Uhlenbeck forcing with
  decorrelation times τ_c = 1, 2 and 4 weeks (blue, orange, green). Same marginal
  variance σ² = (7 W m⁻²)² in all cases. Shaded regions show ±1 s.e. over realizations.
  Persistence (OU) reduces effective escape times relative to white noise across the
  bistable range.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Try to import EBM from asi_resilience_revised; else use embedded copy
try:
    from asi_resilience_revised import (
        R, d2U_dE2, find_steady_states, find_bifurcation_points, SIGMA,
    )
    print("Using EBM from asi_resilience_revised")
except ImportError:
    # Embedded minimal EBM (same as asi_resilience_revised)
    A, B, S = 80.0, 0.6, 150.0
    alpha_ice, alpha_ocean, delta_E = 0.7, 0.07, 10.0
    SIGMA = 7.0  # W m⁻²

    def albedo(E):
        return alpha_ocean + (alpha_ice - alpha_ocean) / (1 + np.exp(E / delta_E))

    def Ein(E):
        return (1 - albedo(E)) * S

    def R(E, dF):
        return Ein(E) - (A + B * E - dF)

    def dEin_dE(E):
        ex = np.exp(E / delta_E)
        return S * (alpha_ice - alpha_ocean) * ex / (delta_E * (1 + ex)**2)

    def d2U_dE2(E):
        return B - dEin_dE(E)

    def find_steady_states(dF):
        guesses = [-80, 0, 80]
        roots = set()
        for g in guesses:
            sol, _, flag, _ = fsolve(R, g, args=(dF,), full_output=True)
            if flag == 1:
                roots.add(round(float(sol[0]), 5))
        return sorted(list(roots))

    def find_bifurcation_points():
        slope_diff = lambda E: dEin_dE(E) - B
        E_bif_neg = fsolve(slope_diff, -15)[0]
        E_bif_pos = fsolve(slope_diff, 15)[0]
        df_at_E_neg = A + B * E_bif_neg - Ein(E_bif_neg)
        df_at_E_pos = A + B * E_bif_pos - Ein(E_bif_pos)
        return df_at_E_pos, df_at_E_neg

    print("Using embedded EBM")

# -----------------------------------------------------------------------------
# Time units: seconds (R and SIGMA are in W m⁻² = J m⁻² s⁻¹)
# -----------------------------------------------------------------------------
SEC_PER_DAY = 86400.0
SEC_PER_YEAR = 365.25 * SEC_PER_DAY
DT = SEC_PER_DAY          # 1-day time step
T_MAX_YEARS = 500.0
T_MAX = T_MAX_YEARS * SEC_PER_YEAR
N_STEPS = int(T_MAX / DT)

# OU correlation times: 1, 2, 4 weeks (in seconds)
TAU_C_WEEKS = [1, 2, 4]
TAU_C_SEC = [w * 7 * SEC_PER_DAY for w in TAU_C_WEEKS]

N_DF = 12                 # number of ΔF values across bistable range
N_REAL = 400              # realizations per (ΔF, noise type)

# Set to True for a fast draft (fewer ΔF points and realizations)
QUICK_RUN = False
if QUICK_RUN:
    N_DF = 5
    N_REAL = 80
    T_MAX_YEARS = 100.0
    T_MAX = T_MAX_YEARS * SEC_PER_YEAR
    N_STEPS = int(T_MAX / DT)


def get_ice_min_and_saddle(dF):
    """Return (E_ice, E_saddle) for ice-covered minimum and saddle; None if not bistable."""
    roots = find_steady_states(dF)
    stables = [r for r in roots if d2U_dE2(r) > 0]
    unstables = [r for r in roots if d2U_dE2(r) < 0]
    if len(stables) < 1 or len(unstables) < 1:
        return None
    E_ice = min(stables)
    E_saddle = unstables[0]
    if E_ice >= E_saddle:
        return None
    return E_ice, E_saddle


def simulate_escape_times_white(dF, sigma, n_real=N_REAL):
    """
    First-passage time from ice-covered minimum to saddle (or beyond).
    Langevin: dE/dt = R(E,dF) + σ ξ(t), ξ white.
    Returns 1D array of escape times in years (nan if no escape).
    """
    out = get_ice_min_and_saddle(dF)
    if out is None:
        return np.full(n_real, np.nan)
    E_ice, E_saddle = out

    sqrt_dt = np.sqrt(DT)
    taus_sec = np.full(n_real, np.nan)

    for i in range(n_real):
        E = float(E_ice)
        for step in range(N_STEPS):
            E = E + R(E, dF) * DT + sigma * sqrt_dt * np.random.randn()
            if E >= E_saddle:
                taus_sec[i] = step * DT
                break
    return taus_sec / SEC_PER_YEAR


def simulate_escape_times_ou(dF, sigma, tau_c_sec, n_real=N_REAL):
    """
    OU forcing: dη/dt = -η/τ_c + (2σ²/τ_c)^{1/2} ξ(t); dE/dt = R(E,dF) + η.
    Same marginal variance σ² for η. tau_c_sec in seconds.
    Returns escape times in years.
    """
    out = get_ice_min_and_saddle(dF)
    if out is None:
        return np.full(n_real, np.nan)
    E_ice, E_saddle = out

    sqrt_dt = np.sqrt(DT)
    ou_scale = np.sqrt(2.0 * sigma**2 / tau_c_sec)  # diffusion coefficient for η
    taus_sec = np.full(n_real, np.nan)

    for i in range(n_real):
        E = float(E_ice)
        eta = 0.0
        for step in range(N_STEPS):
            eta = eta - (eta / tau_c_sec) * DT + ou_scale * sqrt_dt * np.random.randn()
            E = E + R(E, dF) * DT + eta * DT
            if E >= E_saddle:
                taus_sec[i] = step * DT
                break
    return taus_sec / SEC_PER_YEAR


def main():
    df1_star, df2_star = find_bifurcation_points()
    print(f"Bistable range: ΔF₁* ≈ {df1_star:.2f}, ΔF₂* ≈ {df2_star:.2f} W m⁻²")

    dF_vals = np.linspace(df1_star + 0.3, df2_star - 0.3, N_DF)
    sigma = SIGMA

    tau_white_mean = []
    tau_white_se = []
    tau_ou_mean = {tc: [] for tc in TAU_C_SEC}
    tau_ou_se = {tc: [] for tc in TAU_C_SEC}

    for k, dF in enumerate(dF_vals):
        print(f"  ΔF = {dF:.2f} ({k+1}/{N_DF})")
        t_w = simulate_escape_times_white(dF, sigma)
        valid = np.isfinite(t_w)
        nv = valid.sum()
        if nv > 0:
            tau_white_mean.append(np.mean(t_w[valid]))
            tau_white_se.append(np.std(t_w[valid]) / np.sqrt(nv))
        else:
            tau_white_mean.append(np.nan)
            tau_white_se.append(np.nan)

        for tau_c in TAU_C_SEC:
            t_o = simulate_escape_times_ou(dF, sigma, tau_c)
            valid = np.isfinite(t_o)
            nv = valid.sum()
            if nv > 0:
                tau_ou_mean[tau_c].append(np.mean(t_o[valid]))
                tau_ou_se[tau_c].append(np.std(t_o[valid]) / np.sqrt(nv))
            else:
                tau_ou_mean[tau_c].append(np.nan)
                tau_ou_se[tau_c].append(np.nan)

    tau_white_mean = np.array(tau_white_mean)
    tau_white_se = np.array(tau_white_se)
    for tc in TAU_C_SEC:
        tau_ou_mean[tc] = np.array(tau_ou_mean[tc])
        tau_ou_se[tc] = np.array(tau_ou_se[tc])

    # ---- Plot: τ(ΔF) white vs OU ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_yscale("log")

    ax.plot(dF_vals, tau_white_mean, "k-o", ms=6, lw=2, label="White noise")
    if np.any(np.isfinite(tau_white_se)):
        y_lo = np.maximum(1e-6, tau_white_mean - tau_white_se)
        y_hi = tau_white_mean + tau_white_se
        ax.fill_between(dF_vals, y_lo, y_hi, color="gray", alpha=0.25)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for tau_c, col in zip(TAU_C_SEC, colors):
        weeks = tau_c / (7 * SEC_PER_DAY)
        tau_m = tau_ou_mean[tau_c]
        tau_s = tau_ou_se[tau_c]
        ax.plot(dF_vals, tau_m, "-o", color=col, ms=6, lw=2, label=rf"OU, $\tau_c = {weeks:.0f}$ week(s)")
        if np.any(np.isfinite(tau_s)):
            y_lo = np.maximum(1e-6, tau_m - tau_s)
            ax.fill_between(dF_vals, y_lo, tau_m + tau_s, color=col, alpha=0.2)

    ax.set_xlabel(r"$\Delta F$ (W m$^{-2}$)", fontsize=13)
    ax.set_ylabel(r"Mean escape time $\tau(\Delta F)$ (years)", fontsize=13)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    # Ensure visible range includes data (τ can be from days to hundreds of years)
    y_finite = np.concatenate([tau_white_mean] + [tau_ou_mean[tc] for tc in TAU_C_SEC])
    y_finite = y_finite[np.isfinite(y_finite)]
    if len(y_finite) > 0:
        y_min = max(1e-4, np.min(y_finite) * 0.5)
        y_max = np.max(y_finite) * 1.5
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(1e-4, 1e2)
    plt.tight_layout()

    out_path = "extended_escape_time_white_vs_OU.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
