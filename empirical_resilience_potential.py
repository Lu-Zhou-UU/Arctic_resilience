"""
Empirical Resilience Potential from Observed Sea Ice Data
==========================================================

Follows the same analysis and plotting approach as the user's workflow.

Core idea: U(E) = -(σ²/2) ln P(E) from observed E(t) via KDE.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")

# ---- Physical constants ----
RHO_I = 917.0
L_I = 3.34e5
KAPPA = RHO_I * L_I
AREF_KM2 = 25.0 * 25.0
AREF_M2 = AREF_KM2 * 1e6


def load_ice_volume_data(filename, dataset_name):
    """Load ice volume data from txt file."""
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Volume'])
        print(f"✓ Successfully loaded {dataset_name} ice volume data")
        print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"  Volume range: {df['Volume'].min():.1f} - {df['Volume'].max():.1f} thousand km³")
        return df
    except Exception as e:
        print(f"❌ Error loading {dataset_name} ice volume data: {e}")
        return None


def harmonize_volume(c3_df, PIOMAS_df, on='Year', overlap_years=None):
    """
    Return (c3, av, u, (a,b)) with bias-corrected PIOMAS.
    overlap_years: (y0, y1) to restrict regression; None = full overlap (matches original).
    """
    c3 = c3_df[[on, 'Volume']].rename(columns={'Volume': 'Vol_C3S'}).dropna().copy()
    av = PIOMAS_df[[on, 'Volume']].rename(columns={'Volume': 'Vol_PIOMAS'}).dropna().copy()
    df = pd.merge(c3, av, on=on, how='inner')

    if overlap_years is not None:
        df_reg = df[(df[on] >= overlap_years[0]) & (df[on] <= overlap_years[1])]
        df = df_reg if len(df_reg) >= 5 else df

    if len(df) >= 5:
        A = np.vstack([df['Vol_PIOMAS'].values, np.ones(len(df))]).T
        a, b = np.linalg.lstsq(A, df['Vol_C3S'].values, rcond=None)[0]
        print(f"  Bias correction: PIOMAS_bc = {a:.3f} × PIOMAS + {b:.3f}")
    else:
        a, b = 1.0, 0.0
        print(f"  ⚠️ Insufficient overlap ({len(df)} years); no bias correction applied")

    av['Vol_PIOMAS_bc'] = a * av['Vol_PIOMAS'] + b
    u = pd.merge(c3, av[[on, 'Vol_PIOMAS_bc']], on=on, how='outer').sort_values(on)
    u['Vol_Ensemble'] = u[['Vol_C3S', 'Vol_PIOMAS_bc']].mean(axis=1)
    return c3, av, u, (a, b)


def volume_to_state_E(df, time_col, vol_col, aref_m2=AREF_M2, kappa=KAPPA, vol_unit='km3'):
    """
    Convert sea-ice volume to energy per unit area (J m⁻²).
    vol_unit: 'km3' (default, matches original) or 'thousand_km3' (10³ km³)
    Original uses V*1e9 → assumes km³.
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.set_index(time_col).sort_index()

    V = out[vol_col].values
    if vol_unit == 'thousand_km3':
        V_m3 = V * 1e12   # 10³ km³ → m³
    else:
        V_m3 = V * 1e9    # km³ → m³ (original convention)

    h_m = V_m3 / aref_m2
    E_Jm2 = -kappa * h_m
    out['E'] = E_Jm2

    print(f"✓ Converted {len(out)} records to energy state variable")
    print(f"  E range: {out['E'].min():.2e} to {out['E'].max():.2e} J m⁻²")
    return out[['E']]


def monthly_anomalies(E_series):
    """Remove monthly climatology from E(t)."""
    df = E_series.to_frame('E')
    df['month'] = df.index.month
    clim = df.groupby('month')['E'].mean()
    df['E_anom'] = df.apply(lambda r: r['E'] - clim.loc[r['month']], axis=1)
    return df[['E', 'E_anom']]


def estimate_ar1_sigma(E_values, method='innovation'):
    """Estimate noise scale σ from time series."""
    x = np.asarray(E_values)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    if method == 'std':
        return np.std(x, ddof=1)
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    phi = np.clip(phi, -0.99, 0.99)
    var_x = np.var(x, ddof=1)
    if method == 'innovation':
        return np.sqrt(var_x * (1 - phi**2))
    elif method == 'residual':
        residuals = x[1:] - phi * x[:-1]
        return np.std(residuals, ddof=1)
    return np.nan


def kde_empirical_potential(E_values, sigma=1.0, grid=None, bw='scott'):
    """Build U_emp(E) = -(σ²/2) × ln P(E) via KDE."""
    x = np.asarray(E_values)
    x = x[~np.isnan(x)]
    if grid is None:
        lo, hi = np.percentile(x, [1, 99])
        grid = np.linspace(lo, hi, 500)
    kde = stats.gaussian_kde(x, bw_method=bw)
    p = np.maximum(kde(grid), 1e-300)
    U = -(sigma**2 / 2.0) * np.log(p)
    U -= np.min(U)
    spl = UnivariateSpline(grid, U, s=0, k=4)
    return grid, U, kde, spl


def extrema_from_potential(grid, U, spl, E_values=None, kde=None):
    """Identify minima and saddles in potential U(E). Matches original: prominence=0.01, width=2."""
    min_idx, _ = find_peaks(-U, prominence=0.01, width=2)
    max_idx, _ = find_peaks(U, prominence=0.01, width=2)
    d2U = spl.derivative(n=2)

    E_mode = None
    if E_values is not None:
        x = np.asarray(E_values)
        x = x[np.isfinite(x)]
        if len(x) >= 5:
            try:
                if kde is None:
                    kde = stats.gaussian_kde(x, bw_method='scott')
                p = kde(grid)
                E_mode = grid[np.argmax(p)]
            except Exception:
                E_mode = None

    minima = []
    for i in min_idx:
        Emin, Umin = grid[i], U[i]
        curv_min = d2U(Emin)
        minima.append((Emin, Umin, curv_min))

    saddles = []
    for i in max_idx:
        Esad, Usad = grid[i], U[i]
        curv_sad = d2U(Esad)
        saddles.append((Esad, Usad, curv_sad))

    if minima:
        if E_mode is not None:
            mins_ic = [m for m in minima if m[0] < 0]
            pool = mins_ic if mins_ic else minima
            pool.sort(key=lambda m: abs(m[0] - E_mode))
            chosen = pool[0]
            minima = [chosen] + [m for m in minima if m is not chosen]
        else:
            mins_ic = [m for m in minima if m[0] < 0]
            if mins_ic:
                chosen = sorted(mins_ic, key=lambda m: m[0])[-1]
                minima = [chosen] + [m for m in minima if m is not chosen]

    return minima, saddles


def bootstrap_metrics_month(epoch_df, month, nboot=1000, block_size=3,
                            sigma_method='innovation', bw_method='scott'):
    """
    Block bootstrap for uncertainty.
    Returns dict with:
      - draws: array (n_valid, 3) columns [ΔU, U''_min, ΔU/σ²]
      - n_attempt: attempted bootstrap samples
      - n_valid: draws with identifiable barrier and finite metrics
      - frac_identifiable_barrier: n_valid / n_attempt
    """
    month_df = epoch_df[epoch_df.index.month == month]
    years = sorted(month_df.index.year.unique())

    if len(years) < 8:
        return np.array([])

    draws = []
    n_attempt = 0
    for _ in range(nboot):
        n_attempt += 1
        samp_years = []
        while len(samp_years) < len(years):
            y0 = int(np.random.choice(years))
            block = list(range(y0, min(y0 + block_size, years[-1] + 1)))
            block = [y for y in block if y in years]
            samp_years.extend(block)
        samp_years = samp_years[:len(years)]

        # Match original: isin-based extraction (order/duplicates may differ from strict block bootstrap)
        sub = month_df[month_df.index.year.isin(samp_years)]['E'].values
        if len(sub) < 8:
            continue

        sig = estimate_ar1_sigma(sub, method=sigma_method)
        if not np.isfinite(sig) or sig <= 0:
            continue

        try:
            grid, U, kde, spl = kde_empirical_potential(sub, sigma=sig, bw=bw_method)
            mins, sads = extrema_from_potential(grid, U, spl, E_values=sub)

            if not mins or not sads:
                continue

            E_min, U_min, curv_min = mins[0]
            E_sad, U_sad, _ = sads[0]  # Match original: first saddle in list
            dU = U_sad - U_min

            draws.append([dU, curv_min, dU / (sig**2)])
        except Exception:
            continue

    draws_arr = np.array(draws) if draws else np.array([])
    n_valid = len(draws_arr)
    frac = (n_valid / n_attempt) if n_attempt > 0 else np.nan
    return {
        'draws': draws_arr,
        'n_attempt': n_attempt,
        'n_valid': n_valid,
        'frac_identifiable_barrier': frac
    }


def compute_sigma_sensitivity(E_values):
    """Compute σ using three methods."""
    sigmas = {}
    for method in ['innovation', 'std', 'residual']:
        sigmas[method] = estimate_ar1_sigma(E_values, method=method)
    return sigmas


def compute_bandwidth_sensitivity(E_values, sigma,
                                  methods=('scott', 'silverman', 'scott*0.8', 'scott*1.2')):
    """Build U(E) for several KDE bandwidth choices."""
    x = np.asarray(E_values)
    x = x[np.isfinite(x)]
    out = {}
    if x.size < 5 or np.allclose(np.var(x), 0.0):
        return out

    lo, hi = np.percentile(x, [1, 99])
    grid = np.linspace(lo, hi, 500)

    try:
        kde_scott = stats.gaussian_kde(x, bw_method='scott')
        scott_factor = kde_scott.factor
    except Exception:
        return out

    kdes = {
        'scott': kde_scott,
        'silverman': stats.gaussian_kde(x, bw_method='silverman'),
        'scott*0.8': stats.gaussian_kde(x, bw_method=scott_factor * 0.8),
        'scott*1.2': stats.gaussian_kde(x, bw_method=scott_factor * 1.2),
    }

    for label, kde in kdes.items():
        try:
            p = np.maximum(kde(grid), 1e-300)
            U = -(sigma**2 / 2.0) * np.log(p)
            U -= U.min()
            spl = UnivariateSpline(grid, U, s=0, k=4)
            out[label] = (grid, U, spl)
        except Exception:
            continue

    return out


def monthly_empirical_potentials_by_epoch(E_df, epochs, use_sigma_from_ar1=True,
                                          min_points=10, compute_bootstrap=True,
                                          nboot=1000):
    """Compute empirical potentials with bootstrap and sensitivity analyses."""
    results = {}

    df = E_df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month

    for (y0, y1) in epochs:
        epoch_label = f"{y0}-{y1}"
        results[epoch_label] = {}

        epoch_df = df[(df['year'] >= y0) & (df['year'] <= y1)]

        if len(epoch_df) < min_points:
            print(f"  ⚠️ Epoch {epoch_label}: insufficient data ({len(epoch_df)} points)")
            continue

        for m in epoch_df['month'].unique():
            month_data = epoch_df[epoch_df['month'] == m]['E'].values

            if len(month_data) < min_points:
                continue

            sigma_m = estimate_ar1_sigma(month_data, method='innovation')

            if not np.isfinite(sigma_m) or sigma_m <= 0:
                continue

            grid, U, kde, spl = kde_empirical_potential(month_data, sigma=sigma_m)
            minima, saddles = extrema_from_potential(grid, U, spl, E_values=month_data)

            if minima:
                E_s1, U_s1, curv_s1 = minima[0]
                s_right = [s for s in saddles if s[0] > E_s1]
                if s_right:
                    E_s2, U_s2, curv_s2 = sorted(s_right, key=lambda s: s[0])[0]
                else:
                    E_s2 = U_s2 = curv_s2 = np.nan
            else:
                E_s1 = U_s1 = curv_s1 = np.nan
                E_s2 = U_s2 = curv_s2 = np.nan

            dU = U_s2 - U_s1 if np.isfinite(U_s2) else np.nan

            result = {
                'E': month_data,
                'grid': grid,
                'U': U,
                'kde': kde,
                'spl': spl,
                'sigma_m': sigma_m,
                'minima': minima,
                'saddles': saddles,
                'E_s1': E_s1,
                'Upp_s1': U_s1,
                'Upp_prime_prime_s1': curv_s1,
                'E_s2': E_s2,
                'Upp_s2': U_s2,
                'Upp_prime_prime_s2': curv_s2,
                'dU': dU,
                'epoch_df': epoch_df,
            }

            if compute_bootstrap:
                print(f"    Computing bootstrap for {epoch_label}, month {m}...")
                boot_stats = bootstrap_metrics_month(epoch_df, m, nboot=nboot)
                boot_draws = boot_stats['draws']
                result['bootstrap_meta'] = {
                    'n_attempt': boot_stats['n_attempt'],
                    'n_valid': boot_stats['n_valid'],
                    'frac_identifiable_barrier': boot_stats['frac_identifiable_barrier']
                }
                if len(boot_draws) > 0:
                    pct = np.percentile(boot_draws, [2.5, 16, 50, 84, 97.5], axis=0)
                    result['bootstrap'] = {
                        'draws': boot_draws,
                        'dU': {'median': pct[2, 0], 'ci68': (pct[1, 0], pct[3, 0]), 'ci95': (pct[0, 0], pct[4, 0])},
                        'curv': {'median': pct[2, 1], 'ci68': (pct[1, 1], pct[3, 1]), 'ci95': (pct[0, 1], pct[4, 1])},
                        'dU_norm': {'median': pct[2, 2], 'ci68': (pct[1, 2], pct[3, 2]), 'ci95': (pct[0, 2], pct[4, 2])}
                    }

            sigmas = compute_sigma_sensitivity(month_data)
            result['sigma_sensitivity'] = sigmas
            dU_norm_sens = {}
            for method, sig in sigmas.items():
                if np.isfinite(sig) and sig > 0 and np.isfinite(dU):
                    dU_norm_sens[method] = dU / (sig**2)
            result['dU_norm_sensitivity'] = dU_norm_sens

            bw_results = compute_bandwidth_sensitivity(month_data, sigma_m)
            result['bandwidth_sensitivity'] = bw_results

            results[epoch_label][m] = result

    return results


from matplotlib.ticker import FuncFormatter


def plot_comprehensive_potential(results, month=10, figsize=(16, 12)):
    """4-panel: (a) U(E), (b) U''(E), (c) ΔU/σ² + U''(E_min), (d) bandwidth sensitivity."""
    month_name = {4: 'April', 10: 'October'}.get(month, f'Month {month}')

    scale_E = 1e12
    scale_U = 1e22
    fmt_E = FuncFormatter(lambda v, _: f"{v/scale_E:.1f}")
    fmt_U = FuncFormatter(lambda v, _: f"{v/scale_U:.1f}")

    epoch_data = {ep: months[month] for ep, months in results.items() if month in months}
    if not epoch_data:
        print(f"No data for {month_name}")
        return None

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(epoch_data)))

    xlim_E = [np.inf, -np.inf]
    for r in epoch_data.values():
        g = r['grid']
        xlim_E[0] = min(xlim_E[0], g.min())
        xlim_E[1] = max(xlim_E[1], g.max())

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.5], width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # ---------- Panel A: U(E), with minima/saddle and ΔU only when defined ----------
    for idx, (epoch_label, r) in enumerate(epoch_data.items()):
        grid, U = r['grid'], r['U']
        ax1.plot(grid, U, lw=2.5, alpha=0.9, color=colors[idx], label=epoch_label)

        E_s1, U_s1 = r['E_s1'], r['Upp_s1']
        if np.isfinite(E_s1) and np.isfinite(U_s1):
            ax1.plot(E_s1, U_s1, 'o', ms=11, color=colors[idx], mec='k', mew=1.8, zorder=10)
            ax1.annotate("ice-covered min", xy=(E_s1, U_s1), xytext=(8, -12),
                         textcoords='offset points', fontsize=10, color=colors[idx],
                         bbox=dict(fc='white', ec='none', alpha=0.8))

        if np.isfinite(r['E_s2']):
            E_s2, U_s2 = r['E_s2'], r['Upp_s2']
            ax1.plot(E_s2, U_s2, 's', ms=10, color=colors[idx], mec='k', mew=1.8, zorder=10)
            ax1.annotate("saddle", xy=(E_s2, U_s2), xytext=(8, 8),
                         textcoords='offset points', fontsize=10, color=colors[idx],
                         bbox=dict(fc='white', ec='none', alpha=0.8))

            ax1.annotate("", xy=(E_s2, U_s2), xytext=(E_s1, U_s1),
                         arrowprops=dict(arrowstyle='<->', lw=2.5, color=colors[idx], alpha=0.7))

            dU = r['dU']
            dU_norm = dU / (r['sigma_m']**2)
            xm, ym = 0.5 * (E_s1 + E_s2), 0.5 * (U_s1 + U_s2)
            dU_scaled = dU / 1e22
            ax1.text(xm, ym,
                     rf'$\Delta U={dU_scaled:.2f}\times 10^{{22}}$' + '\n' + rf'$\Delta U/\sigma^2={dU_norm:.2f}$',
                     ha='center', va='bottom', fontsize=15, fontweight='bold',
                     bbox=dict(fc='white', ec=colors[idx], boxstyle='round,pad=0.5', alpha=0.9))

    ax1.set_xlim(xlim_E)
    ax1.xaxis.set_major_formatter(fmt_E)
    ax1.yaxis.set_major_formatter(fmt_U)
    ax1.set_xlabel(r'$E\;(\times 10^{12}\ \mathrm{J\,m^{-2}})$', fontsize=15, fontweight='bold')
    ax1.set_ylabel(r'$U(E)\;(\times 10^{22}\ \mathrm{J^{2}\,m^{-4}})$', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, ls='--', alpha=0.3)
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    # ---------- Panel B: U''(E) ----------
    for idx, (epoch_label, r) in enumerate(epoch_data.items()):
        grid, spl = r['grid'], r['spl']
        curv_curve = spl.derivative(2)(grid)
        ax2.plot(grid, curv_curve, lw=2.5, alpha=0.85, color=colors[idx], label=epoch_label)
        ax2.plot(r['E_s1'], r['Upp_prime_prime_s1'], 'o', ms=10, color=colors[idx], mec='black', mew=1.5)
        if np.isfinite(r['E_s2']):
            ax2.axvline(r['E_s2'], ls='--', lw=1.5, color=colors[idx], alpha=0.6)

    ax2.set_xlim(xlim_E)
    ax2.xaxis.set_major_formatter(fmt_E)
    ax2.axhline(0, color='gray', ls=':', lw=1.5, alpha=0.6)
    ax2.set_xlabel(r'$E\;(\times 10^{12}\ \mathrm{J\,m^{-2}})$', fontsize=15, fontweight='bold')
    ax2.set_ylabel(r"$U^{\prime\prime}(E)\ \mathrm{(dimensionless)}$", fontsize=15, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    # ---------- Panel C: Bootstrap summary (clean style: blue bars + orange points) ----------
    epochs = list(epoch_data.keys())
    x_pos = np.arange(len(epochs))
    bar_width = 0.36

    dU_norm_vals, dU_norm_cis = [], []
    curv_vals, curv_cis = [], []
    sigma_vals = []

    for ep in epochs:
        r = epoch_data[ep]
        sigma_vals.append(r['sigma_m'])
        # U''(E_min): always use point estimate to match Panel B; bootstrap CI for uncertainty
        curv_vals.append(r['Upp_prime_prime_s1'])
        if 'bootstrap' in r:
            dU_norm_vals.append(r['bootstrap']['dU_norm']['median'])
            dU_norm_cis.append([*r['bootstrap']['dU_norm']['ci68']])
            curv_cis.append([*r['bootstrap']['curv']['ci68']])
        else:
            dU_norm = r['dU'] / (r['sigma_m']**2) if (np.isfinite(r['dU']) and r['sigma_m'] > 0) else np.nan
            dU_norm_vals.append(dU_norm)
            dU_norm_cis.append([dU_norm, dU_norm])
            curv_cis.append([r['Upp_prime_prime_s1'], r['Upp_prime_prime_s1']])

    curv_vals = np.asarray(curv_vals)
    dU_norm_vals = np.asarray(dU_norm_vals)
    curv_vals_array = np.array(curv_vals)
    curv_mean = np.nanmean(curv_vals_array)
    curv_min = np.nanmin(curv_vals_array)
    curv_max = np.nanmax(curv_vals_array)
    print(f"U''(Emin) statistics: mean = {curv_mean:.4f}, range = [{curv_min:.4f}, {curv_max:.4f}]")

    ax3_twin = ax3.twinx()

    # Blue bars: conditional ΔU/σ² (computed only when barrier is identifiable)
    dU_norm_plot = np.where(np.isfinite(dU_norm_vals), dU_norm_vals, 0.0)
    ax3.bar(x_pos, dU_norm_plot, width=0.55, color='steelblue', alpha=0.75,
            edgecolor='black', linewidth=1.4, label='ΔU/σ² (resilience)')
    ax3.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    if dU_norm_cis:
        yerr = np.array([[dU_norm_vals[i] - dU_norm_cis[i][0],
                         dU_norm_cis[i][1] - dU_norm_vals[i]] for i in range(len(epochs))]).T
        yerr = np.where(np.isfinite(yerr), yerr, 0)
        ax3.errorbar(x_pos, dU_norm_plot, yerr=yerr, fmt='none',
                     ecolor='black', capsize=5, capthick=2, alpha=0.8)

    valid_curv = np.isfinite(curv_vals)
    if np.any(valid_curv):
        ax3_twin.plot(x_pos[valid_curv], curv_vals[valid_curv], 'o', ms=12, color='darkorange',
                      mec='black', mew=2, label="U''(E_min)", zorder=6)
    if curv_cis and np.any(valid_curv):
        # CIs are bootstrap summaries and may not be centered on the plotted point estimate.
        # Force non-negative error lengths to satisfy matplotlib errorbar requirements.
        yerr_curv = np.array([[
            max(curv_vals[i] - curv_cis[i][0], 0.0),
            max(curv_cis[i][1] - curv_vals[i], 0.0)
        ] for i in range(len(epochs))]).T
        yerr_curv = np.where(np.isfinite(yerr_curv), yerr_curv, 0.0)
        ax3_twin.errorbar(
            x_pos[valid_curv], curv_vals[valid_curv],
            yerr=yerr_curv[:, valid_curv], fmt='none',
            ecolor='darkorange', capsize=5, capthick=2, alpha=0.8, zorder=5
        )

    # Fixed y-axis ranges for Panel C
    ax3.set_ylim(0, 0.6)
    ax3_twin.set_ylim(0.2, 0.75)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(epochs, fontsize=15)
    ax3.set_ylabel(r'$\Delta U/\sigma^{2}$ (unitless)', fontsize=15, fontweight='bold', color='steelblue')
    ax3_twin.set_ylabel(r'$U^{\prime\prime}(E_{\min})$ (unitless)', fontsize=15, fontweight='bold', color='darkorange')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3_twin.tick_params(axis='y', labelcolor='darkorange')
    ax3.grid(True, axis='y', ls='--', alpha=0.3)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)

    # σ annotation (normalization transparency)
    sigma_text = []
    for ep, sig in zip(epochs, sigma_vals):
        if np.isfinite(sig):
            sigma_text.append(f"{ep}: σ={sig/1e11:.2f}×10^11 J m^-2")
    if sigma_text:
        ax3.text(0.02, 0.03, "Noise scale (AR1 innovation): " + " | ".join(sigma_text),
                 transform=ax3.transAxes, fontsize=10,
                 bbox=dict(fc='white', ec='0.7', alpha=0.9, boxstyle='round,pad=0.35'))

    # # Barrier-detection fraction reported as compact text (instead of extra bars)
    # frac_text = []
    # for ep in epochs:
    #     f = epoch_data[ep].get('bootstrap_meta', {}).get('frac_identifiable_barrier', np.nan)
    #     if np.isfinite(f):
    #         frac_text.append(f"{ep}: {100*f:.0f}%")
    # if frac_text:
    #     ax3.text(0.02, 0.92, "Barrier detection fraction: " + " | ".join(frac_text),
    #              transform=ax3.transAxes, fontsize=10, color='dimgray',
    #              bbox=dict(fc='white', ec='0.7', alpha=0.9, boxstyle='round,pad=0.25'))

    # ---------- Panel D: Bandwidth sensitivity ----------
    # Show bandwidth sensitivity for each epoch explicitly.
    for idx, ep in enumerate(epochs):
        r = epoch_data[ep]
        if 'bandwidth_sensitivity' in r:
            for label, (grid, U, _spl) in r['bandwidth_sensitivity'].items():
                if label != 'scott':
                    continue
                ax4.plot(grid, U, lw=2.4, alpha=0.95, color=colors[idx], label=f'{ep} (Scott)')

            # Add envelope from non-Scott choices as shaded band to avoid clutter.
            curves = []
            for label, (grid, U, _spl) in r['bandwidth_sensitivity'].items():
                if label == 'scott':
                    continue
                curves.append(U)
            if curves:
                arr = np.vstack(curves)
                ax4.fill_between(grid, arr.min(axis=0), arr.max(axis=0), color=colors[idx], alpha=0.15,
                                 label=f'{ep} BW range')

    ax4.xaxis.set_major_formatter(fmt_E)
    ax4.yaxis.set_major_formatter(fmt_U)
    ax4.set_xlabel(r'$E\;(\times 10^{12}\ \mathrm{J\,m^{-2}})$', fontsize=15, fontweight='bold')
    ax4.set_ylabel(r'$U(E)\;(\times 10^{22}\ \mathrm{J^{2}\,m^{-4}})$', fontsize=15, fontweight='bold')
    ax4.legend(loc='best', framealpha=0.95, fontsize=10)
    ax4.grid(True, ls='--', alpha=0.3)
    ax4.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    plt.tight_layout()
    return fig


def export_comprehensive_metrics(results, filename='resilience_metrics_complete.csv'):
    """Export metrics table with bootstrap CIs."""
    rows = []
    for epoch_label, months in results.items():
        for m, r in months.items():
            row = {
                'epoch': epoch_label,
                'month': m,
                'E_min': r['E_s1'],
                'U_min': r['Upp_s1'],
                "U''_min": r['Upp_prime_prime_s1'],
                'E_saddle': r['E_s2'],
                'U_saddle': r['Upp_s2'],
                "U''_saddle": r.get('Upp_prime_prime_s2', np.nan),
                'ΔU': r['dU'],
                'σ_innovation': r['sigma_m']
            }
            if np.isfinite(r['dU']) and r['sigma_m'] > 0:
                row['ΔU/σ²'] = r['dU'] / (r['sigma_m']**2)
            else:
                row['ΔU/σ²'] = np.nan

            if 'bootstrap' in r:
                boot = r['bootstrap']
                row['ΔU_median'] = boot['dU']['median']
                row['ΔU_ci68_lo'], row['ΔU_ci68_hi'] = boot['dU']['ci68']
                row['ΔU_ci95_lo'], row['ΔU_ci95_hi'] = boot['dU']['ci95']
                row["U''_min_median"] = boot['curv']['median']
                row["U''_min_ci68_lo"], row["U''_min_ci68_hi"] = boot['curv']['ci68']
                row['ΔU/σ²_median'] = boot['dU_norm']['median']
                row['ΔU/σ²_ci68_lo'], row['ΔU/σ²_ci68_hi'] = boot['dU_norm']['ci68']

            if 'sigma_sensitivity' in r:
                for method, sig in r['sigma_sensitivity'].items():
                    row[f'σ_{method}'] = sig
            if 'dU_norm_sensitivity' in r:
                for method, val in r['dU_norm_sensitivity'].items():
                    row[f'ΔU/σ²_{method}'] = val

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, float_format='%.6e')
    print(f"\n✓ Saved comprehensive metrics to: {filename}")
    return df


def barrier_presence_fraction(boot_draws, threshold=0.05):
    """Fraction of bootstrap realizations with ΔU/σ² > threshold."""
    if boot_draws is None or len(boot_draws) == 0:
        return np.nan
    dU_norm = boot_draws[:, 2]
    valid = np.isfinite(dU_norm) & (dU_norm > threshold)
    return np.mean(valid)


def rolling_ews_annual(E_series, window_years):
    """Rolling variance and lag-1 AC on annual series."""
    df = E_series.to_frame('E').copy()
    df['year'] = df.index.year
    rows = []
    for end in range(window_years - 1, len(df)):
        sub = df.iloc[end - window_years + 1:end + 1]['E'].values
        yearsub = int(df.iloc[end]['year'])
        if len(sub) >= 5:
            var = np.var(sub, ddof=1)
            ac1 = np.corrcoef(sub[:-1], sub[1:])[0, 1] if len(sub) > 2 else np.nan
            rows.append({'year': yearsub, 'var': var, 'ac1': ac1})
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    OUT_DIR = '/Users/jay'
    PATH_C3 = '/Volumes/Yotta_1/C3_ice_volume_October.txt'
    PATH_PIOMAS = '/Volumes/Yotta_1/PIOMAS_ice_volume_October.txt'
    VOL_UNIT = 'km3'   # Matches original: V*1e9

    print("=" * 80)
    print("COMPREHENSIVE SEA ICE RESILIENCE ANALYSIS")
    print("=" * 80 + "\n")

    print("Loading ice volume data...\n")
    c3_df = load_ice_volume_data(PATH_C3, 'C3S')
    PIOMAS_df = load_ice_volume_data(PATH_PIOMAS, 'PIOMAS')

    if c3_df is None or PIOMAS_df is None:
        print("\nUsing synthetic data for demonstration...")
        np.random.seed(42)
        years = np.arange(1982, 2025)
        V = 15 - 0.05 * (years - 1982) + np.random.randn(len(years)) * 1.5
        c3_df = pd.DataFrame({'Year': years, 'Volume': np.maximum(V, 2)})
        PIOMAS_df = c3_df.copy()
        VOL_UNIT = 'km3'

    print("\nHarmonizing C3S and PIOMAS datasets...")
    c3, av, vol_ens, (a, b) = harmonize_volume(c3_df, PIOMAS_df, on='Year')  # Full overlap, matches original

    vol_ens = vol_ens.rename(columns={'Vol_Ensemble': 'Volume'})
    vol_ens = vol_ens.dropna(subset=['Volume'])
    vol_ens['Date'] = pd.to_datetime(vol_ens['Year'].astype(int).astype(str) + '-10-01')

    print("\nConverting volume to energy state variable...")
    E_oct_df = volume_to_state_E(vol_ens[['Date', 'Volume']], time_col='Date', vol_col='Volume',
                                 aref_m2=AREF_M2, vol_unit=VOL_UNIT)

    print("\n" + "=" * 80)
    print("RESILIENCE ANALYSIS WITH BOOTSTRAP UNCERTAINTY")
    print("=" * 80 + "\n")

    epochs = [(1982, 2005), (2006, 2024)]
    print(f"Analysis epochs: {epochs}\n")

    print("Computing empirical potentials with bootstrap...")
    results_oct = monthly_empirical_potentials_by_epoch(
        E_oct_df,
        epochs=epochs,
        use_sigma_from_ar1=True,
        min_points=8,
        compute_bootstrap=True,
        nboot=500
    )

    try:
        metrics_df = export_comprehensive_metrics(
            results_oct,
            filename=f'{OUT_DIR}/oct_resilience_metrics_comprehensive.csv'
        )
    except Exception as e:
        print(f"⚠️ export_comprehensive_metrics failed: {e}")
        metrics_df = pd.DataFrame()

    print("\n=== Summary of October Resilience Metrics ===")
    if metrics_df is not None and not metrics_df.empty:
        wanted = ['epoch', 'month', 'E_min', 'ΔU', "U''_min", 'ΔU/σ²',
                  'ΔU_median', 'ΔU_ci68_lo', 'ΔU_ci68_hi',
                  "U''_min_median", "U''_min_ci68_lo", "U''_min_ci68_hi",
                  'ΔU/σ²_median', 'ΔU/σ²_ci68_lo', 'ΔU/σ²_ci68_hi']
        cols = [c for c in wanted if c in metrics_df.columns]
        to_show = metrics_df[metrics_df['month'] == 10] if 'month' in metrics_df.columns else metrics_df
        print(to_show[cols].to_string(index=False))

    if results_oct and 10 in results_oct.get('1982-2005', {}) and 10 in results_oct.get('2006-2024', {}):
        early_boot = results_oct['1982-2005'][10].get('bootstrap', {}).get('draws')
        late_boot = results_oct['2006-2024'][10].get('bootstrap', {}).get('draws')
        f_early = barrier_presence_fraction(early_boot)
        f_late = barrier_presence_fraction(late_boot)
        print(f"\nBarrier present (bootstrap): early={f_early:.2f}, late={f_late:.2f}")

    print("\nCreating comprehensive visualization...")
    fig = plot_comprehensive_potential(results_oct, month=10)
    if fig is not None:
        out_png = f'{OUT_DIR}/oct_comprehensive_resilience_analysis.png'
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {out_png}")
        plt.close(fig)
    else:
        print("⚠️ No figure produced (no data for October)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
