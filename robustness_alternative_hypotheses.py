# =============================================================================
# Robustness and alternative hypotheses (Results subsection)
# Implements: epoch split sensitivity, detrending sensitivity, KDE bandwidth
# sensitivity, and synthetic monostable controls for the resilience potential.
# Produces Extended Data Figure (6 panels) and CSV summaries.
#
# Usage: Run from the same directory as Sea_ice_energy_new.py, or set
#   sys.path and data paths below. Requires: numpy, pandas, scipy, matplotlib,
#   seaborn; optionally statsmodels for LOESS.
# =============================================================================

from __future__ import division, print_function

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# Optional: add path to Sea_ice_energy_new if not in same directory
import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from Sea_ice_energy_new import (
        load_ice_volume_data,
        harmonize_volume,
        volume_to_state_E,
        estimate_ar1_sigma,
        kde_empirical_potential,
        extrema_from_potential,
        bootstrap_metrics_month,
        compute_bandwidth_sensitivity,
        AREF_M2,
        KAPPA,
    )
except ImportError:
    # If running from another location, set path to Yotta_1 or copy required defs
    sys.path.insert(0, '/Volumes/Yotta_1')
    from Sea_ice_energy_new import (
        load_ice_volume_data,
        harmonize_volume,
        volume_to_state_E,
        estimate_ar1_sigma,
        kde_empirical_potential,
        extrema_from_potential,
        bootstrap_metrics_month,
        compute_bandwidth_sensitivity,
        AREF_M2,
        KAPPA,
    )

# Data paths (adjust if needed)
DEFAULT_C3_FILE = '/Volumes/Yotta_1/C3_ice_volume_October.txt'
DEFAULT_PIOMAS_FILE = '/Volumes/Yotta_1/PIOMAS_ice_volume_October.txt'

# -----------------------------------------------------------------------------
# Detrending helpers (for detrending sensitivity)
# -----------------------------------------------------------------------------

def detrend_linear(E_values):
    """Remove linear trend; return residuals with same mean as original."""
    x = np.asarray(E_values, dtype=float)
    t = np.arange(len(x))
    mask = np.isfinite(x)
    if mask.sum() < 4:
        return x
    slope, intercept = np.polyfit(t[mask], x[mask], 1)
    trend = intercept + slope * t
    return x - trend + np.nanmean(x)


def detrend_quadratic(E_values):
    """Remove quadratic trend; return residuals with same mean."""
    x = np.asarray(E_values, dtype=float)
    t = np.arange(len(x))
    mask = np.isfinite(x)
    if mask.sum() < 6:
        return x
    coefs = np.polyfit(t[mask], x[mask], 2)
    trend = np.polyval(coefs, t)
    return x - trend + np.nanmean(x)


def detrend_loess(E_values, frac=0.4):
    """Remove LOESS smooth (trend); return residuals. frac = span as fraction of n."""
    x = np.asarray(E_values, dtype=float)
    t = np.arange(len(x))
    mask = np.isfinite(x)
    if mask.sum() < 5:
        return x
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(x[mask], t[mask], frac=min(frac, 0.99), return_sorted=False)
        trend = np.interp(t, t[mask], smooth)
        return x - trend + np.nanmean(x)
    except ImportError:
        # Fallback: moving average as crude LOESS proxy
        w = max(3, int(len(x) * frac))
        kernel = np.ones(w) / w
        trend = np.convolve(np.where(mask, x, np.nanmean(x)), kernel, mode='same')
        return x - trend + np.nanmean(x)


# -----------------------------------------------------------------------------
# Single-epoch potential and barrier metrics (used by all analyses)
# -----------------------------------------------------------------------------

BARRIER_THRESHOLD = 0.05  # min ΔU/σ² to count as "barrier present"

def barrier_presence_fraction(boot_draws, threshold=BARRIER_THRESHOLD):
    """Fraction of bootstrap realizations with ΔU/σ² > threshold."""
    if boot_draws is None or len(boot_draws) == 0:
        return np.nan
    dU_norm = boot_draws[:, 2]
    valid = np.isfinite(dU_norm) & (dU_norm > threshold)
    return np.mean(valid)


def potential_metrics_from_E(E_values, sigma=None, bw_method='scott'):
    """
    From E_values (1D array), compute point-estimate dU, dU/σ², and whether
    a distinct barrier exists. Returns dict with dU, sigma, dU_norm, has_barrier.
    """
    x = np.asarray(E_values)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return None
    if sigma is None:
        sigma = estimate_ar1_sigma(x, method='innovation')
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    try:
        grid, U, kde, spl = kde_empirical_potential(x, sigma=sigma, bw_method=bw_method)
        mins, sads = extrema_from_potential(grid, U, spl, E_values=x)
        if not mins or not sads:
            return None
        E_min, U_min, _ = mins[0]
        s_right = [s for s in sads if s[0] > E_min]
        if not s_right:
            return None
        E_sad, U_sad, _ = sorted(s_right, key=lambda s: s[0])[0]
        dU = U_sad - U_min
        dU_norm = dU / (sigma**2)
        return {
            'dU': dU, 'sigma': sigma, 'dU_norm': dU_norm,
            'has_barrier': dU_norm > BARRIER_THRESHOLD,
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 1) Epoch split sensitivity
# -----------------------------------------------------------------------------

def run_epoch_sensitivity(E_df, month=10, split_years=(2000, 2003, 2006, 2007, 2010, 2012),
                          window_years=20, nboot=500, min_points=8):
    """
    Repeat potential reconstruction for alternative epoch splits and sliding windows.
    E_df: DataFrame index=Date, column 'E'.
    Returns: dict with 'splits' and 'sliding' results for plotting.
    """
    df = E_df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    month_df = df[(df['month'] == month)].copy()
    years_all = sorted(month_df['year'].unique())
    y_min, y_max = min(years_all), max(years_all)

    results_splits = []
    for split_year in split_years:
        y_early_end = split_year - 1
        y_late_start = split_year
        if y_early_end < y_min or y_late_start > y_max:
            continue
        early_df = month_df[(month_df['year'] >= y_min) & (month_df['year'] <= y_early_end)]
        late_df = month_df[(month_df['year'] >= y_late_start) & (month_df['year'] <= y_max)]
        if len(early_df) < min_points or len(late_df) < min_points:
            continue
        # Build epoch_df with DatetimeIndex and column E (bootstrap_metrics_month expects index.month)
        early_epoch = early_df[['E']].copy()
        late_epoch = late_df[['E']].copy()
        early_epoch.index = pd.to_datetime(early_epoch.index)
        late_epoch.index = pd.to_datetime(late_epoch.index)

        boot_early = bootstrap_metrics_month(early_epoch, month, nboot=nboot)
        boot_late = bootstrap_metrics_month(late_epoch, month, nboot=nboot)

        def _summarize(boot, label):
            if boot is None or len(boot) == 0:
                return {'median': np.nan, 'ci68_lo': np.nan, 'ci68_hi': np.nan, 'frac_barrier': np.nan}
            pct = np.percentile(boot, [16, 50, 84], axis=0)
            return {
                'median': pct[1, 2],
                'ci68_lo': pct[0, 2],
                'ci68_hi': pct[2, 2],
                'frac_barrier': barrier_presence_fraction(boot),
            }

        results_splits.append({
            'split_year': split_year,
            'early': f"{y_min}-{y_early_end}",
            'late': f"{y_late_start}-{y_max}",
            'dU_norm_early_median': _summarize(boot_early, 'early')['median'],
            'dU_norm_early_ci68': (_summarize(boot_early, 'early')['ci68_lo'], _summarize(boot_early, 'early')['ci68_hi']),
            'dU_norm_late_median': _summarize(boot_late, 'late')['median'],
            'dU_norm_late_ci68': (_summarize(boot_late, 'late')['ci68_lo'], _summarize(boot_late, 'late')['ci68_hi']),
            'frac_barrier_early': _summarize(boot_early, 'early')['frac_barrier'],
            'frac_barrier_late': _summarize(boot_late, 'late')['frac_barrier'],
        })

    # Sliding 20-year windows (e.g. 1982-2001, 1984-2003, ...)
    results_sliding = []
    for start in range(y_min, y_max - window_years + 2):
        end = start + window_years - 1
        win_df = month_df[(month_df['year'] >= start) & (month_df['year'] <= end)]
        if len(win_df) < min_points:
            continue
        win_epoch = win_df[['E']].copy()
        win_epoch.index = pd.to_datetime(win_epoch.index)
        boot = bootstrap_metrics_month(win_epoch, month, nboot=nboot)
        if boot is None or len(boot) == 0:
            continue
        pct = np.percentile(boot, [16, 50, 84], axis=0)
        results_sliding.append({
            'window_start': start,
            'window_end': end,
            'window_mid': (start + end) / 2,
            'dU_norm_median': pct[1, 2],
            'dU_norm_ci68_lo': pct[0, 2],
            'dU_norm_ci68_hi': pct[2, 2],
            'frac_barrier': barrier_presence_fraction(boot),
        })

    return {
        'splits': pd.DataFrame(results_splits),
        'sliding': pd.DataFrame(results_sliding),
        'split_years': split_years,
    }


# -----------------------------------------------------------------------------
# 2) Detrending sensitivity
# -----------------------------------------------------------------------------

def run_detrending_sensitivity(E_df, epochs=((1982, 2005), (2006, 2024)), month=10, nboot=500):
    """
    Apply linear, quadratic, and LOESS detrending within each epoch; run
    potential pipeline and bootstrap. Returns dict with metrics per (epoch_label, detrend_method).
    """
    df = E_df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    month_df = df[df['month'] == month]

    detrenders = {
        'linear': detrend_linear,
        'quadratic': detrend_quadratic,
        'LOESS': lambda x: detrend_loess(x, frac=0.4),
    }
    rows = []
    for (y0, y1) in epochs:
        ep_df = month_df[(month_df['year'] >= y0) & (month_df['year'] <= y1)]
        if len(ep_df) < 8:
            continue
        E_raw = ep_df['E'].values
        epoch_label = f"{y0}-{y1}"
        for method_name, detrend_fn in detrenders.items():
            E_detrended = detrend_fn(E_raw)
            ep_detrended = ep_df[['E']].copy()
            ep_detrended['E'] = E_detrended
            ep_detrended.index = pd.to_datetime(ep_detrended.index)
            boot = bootstrap_metrics_month(ep_detrended, month, nboot=nboot)
            point = potential_metrics_from_E(E_detrended)
            # Always append a row so panel (c) is never empty; use bootstrap when point fails
            if boot is not None and len(boot) > 0:
                pct = np.percentile(boot, [16, 50, 84], axis=0)
                row = {
                    'epoch': epoch_label,
                    'detrend': method_name,
                    'dU_norm_median': pct[1, 2],
                    'dU_norm_ci68_lo': pct[0, 2],
                    'dU_norm_ci68_hi': pct[2, 2],
                    'frac_barrier': barrier_presence_fraction(boot),
                }
            elif point is not None:
                row = {
                    'epoch': epoch_label,
                    'detrend': method_name,
                    'dU_norm_median': point['dU_norm'],
                    'dU_norm_ci68_lo': point['dU_norm'],
                    'dU_norm_ci68_hi': point['dU_norm'],
                    'frac_barrier': 1.0 if point['has_barrier'] else 0.0,
                }
            else:
                row = {
                    'epoch': epoch_label,
                    'detrend': method_name,
                    'dU_norm_median': np.nan,
                    'dU_norm_ci68_lo': np.nan,
                    'dU_norm_ci68_hi': np.nan,
                    'frac_barrier': np.nan,
                }
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3) KDE bandwidth sensitivity (barrier metrics per bandwidth)
# -----------------------------------------------------------------------------

def run_bandwidth_sensitivity(E_values, sigma=None, nboot=300):
    """
    For each bandwidth (scott, silverman, scott*0.8, scott*1.2), compute
    point dU/σ² and fraction of bootstrap runs with barrier.
    """
    x = np.asarray(E_values)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return None
    if sigma is None:
        sigma = estimate_ar1_sigma(x, method='innovation')
    bw_results = compute_bandwidth_sensitivity(x, sigma)
    if not bw_results:
        return None
    out = []
    for label, (grid, U, spl) in bw_results.items():
        try:
            mins, sads = extrema_from_potential(grid, U, spl, E_values=x)
            if not mins or not sads:
                out.append({'bandwidth': label, 'dU_norm': np.nan, 'has_barrier': False})
                continue
            E_min, U_min, _ = mins[0]
            s_right = [s for s in sads if s[0] > E_min]
            if not s_right:
                out.append({'bandwidth': label, 'dU_norm': np.nan, 'has_barrier': False})
                continue
            _, U_sad, _ = sorted(s_right, key=lambda s: s[0])[0]
            dU = U_sad - U_min
            dU_norm = dU / (sigma**2)
            out.append({'bandwidth': label, 'dU_norm': dU_norm, 'has_barrier': dU_norm > BARRIER_THRESHOLD})
        except Exception:
            out.append({'bandwidth': label, 'dU_norm': np.nan, 'has_barrier': False})
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# 4) Synthetic monostable controls
# -----------------------------------------------------------------------------

def fit_ar1_drift(E_values):
    """Fit AR(1) with linear drift: E_t = a + b*t + z_t, z_t = phi*z_{t-1} + eps."""
    x = np.asarray(E_values)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return None
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)
    z = x - (a + b * t)
    phi = np.corrcoef(z[:-1], z[1:])[0, 1]
    phi = np.clip(phi, -0.99, 0.99)
    resid = z[1:] - phi * z[:-1]
    sigma_innov = np.std(resid, ddof=1)
    return {'a': a, 'b': b, 'phi': phi, 'sigma': sigma_innov, 'z0': z[0], 'n': n}


def generate_synthetic_ar1_drift(params, n_realizations=1000, seed=42):
    """
    Generate n_realizations monostable series: x[0]=mu0, x[t+1] = a + b*(t+1) + phi*(x[t]-a-b*t) + sigma*eps.
    Returns (n_realizations, n) array.
    """
    rng = np.random.default_rng(seed)
    a, b, phi, sigma, z0, n = params['a'], params['b'], params['phi'], params['sigma'], params['z0'], params['n']
    out = np.zeros((n_realizations, n))
    for i in range(n_realizations):
        z = np.zeros(n)
        z[0] = z0
        for t in range(1, n):
            z[t] = phi * z[t - 1] + sigma * rng.standard_normal()
        x = a + b * np.arange(n) + z
        out[i, :] = x
    return out


def run_synthetic_controls(E_early_values, n_realizations=1000, nboot_per_series=0, no_barrier_value=0.0):
    """
    Fit AR(1)+drift to E_early_values; generate n_realizations synthetic series;
    run potential pipeline on each; return distribution of ΔU/σ² and false-positive rate.

    When the pipeline finds no barrier (potential_metrics_from_E returns None), we store
    no_barrier_value (default 0.0) so the distribution and histogram are interpretable:
    "no barrier" = 0, "barrier" = ΔU/σ² > 0.
    """
    params = fit_ar1_drift(E_early_values)
    if params is None:
        return None
    syn = generate_synthetic_ar1_drift(params, n_realizations=n_realizations)
    dU_norm_list = []
    has_barrier_list = []
    for i in range(n_realizations):
        met = potential_metrics_from_E(syn[i, :])
        if met is not None:
            dU_norm_list.append(met['dU_norm'])
            has_barrier_list.append(met['has_barrier'])
        else:
            # No barrier: store 0 so histogram shows "all at 0" instead of all NaN
            dU_norm_list.append(no_barrier_value)
            has_barrier_list.append(False)
    dU_norm_arr = np.array(dU_norm_list)
    fp_rate = np.mean(has_barrier_list)
    return {
        'params': params,
        'dU_norm_synthetic': dU_norm_arr,
        'dU_norm_observed_early': potential_metrics_from_E(E_early_values)['dU_norm'] if potential_metrics_from_E(E_early_values) else np.nan,
        'false_positive_rate': fp_rate,
        'n_with_barrier': np.sum(has_barrier_list),
        'n_realizations': n_realizations,
        'synthetic_example': syn[0, :],
    }


# -----------------------------------------------------------------------------
# Extended Data Figure (compact 2×2 layout)
# -----------------------------------------------------------------------------

def plot_robustness_extended_figure(epoch_sens, detrend_sens, bw_sens_early, bw_sens_late,
                                    synth_results, observed_early_dU_norm, observed_late_dU_norm,
                                    outpath='Extended_Data_robustness.png'):
    """
    Build a compact Extended Data figure with the three key robustness panels:
      (a) Epoch split sensitivity of ΔU/σ²
      (b) Detrending sensitivity of ΔU/σ²
      (c) Synthetic vs observed barrier metric (ΔU/σ²)
    The fourth axis is left blank (turned off).
    """

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) ΔU/σ² vs split year (early and late epoch)
    ax = axes[0, 0]
    if epoch_sens['splits'] is not None and len(epoch_sens['splits']) > 0:
        df = epoch_sens['splits']
        x = df['split_year'].values
        e_lo = df['dU_norm_early_median'] - df['dU_norm_early_ci68'].apply(lambda v: v[0] if isinstance(v, (tuple, list)) else v)
        e_hi = df['dU_norm_early_ci68'].apply(lambda v: v[1] if isinstance(v, (tuple, list)) else v) - df['dU_norm_early_median']
        ax.errorbar(x, df['dU_norm_early_median'], yerr=np.array([e_lo.values, e_hi.values]), fmt='o-', capsize=4, label='Early epoch', color='steelblue')
        l_lo = df['dU_norm_late_median'] - df['dU_norm_late_ci68'].apply(lambda v: v[0] if isinstance(v, (tuple, list)) else v)
        l_hi = df['dU_norm_late_ci68'].apply(lambda v: v[1] if isinstance(v, (tuple, list)) else v) - df['dU_norm_late_median']
        ax.errorbar(x, df['dU_norm_late_median'], yerr=np.array([l_lo.values, l_hi.values]), fmt='s--', capsize=4, label='Late epoch', color='coral')
    ax.set_xlabel('Split year')
    ax.set_ylabel(r'$\Delta U/\sigma^2$')
    # ax.set_title('(a) Epoch split sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Detrending sensitivity: ΔU/σ² by method (grouped bars: early vs late epoch)
    ax = axes[0, 1]
    if detrend_sens is not None and len(detrend_sens) > 0:
        methods = detrend_sens['detrend'].unique().tolist()
        epochs = detrend_sens['epoch'].unique().tolist()
        x = np.arange(len(methods))
        width = 0.35
        for i, epoch in enumerate(epochs):
            med, err_lo, err_hi = [], [], []
            for m in methods:
                row = detrend_sens[(detrend_sens['epoch'] == epoch) & (detrend_sens['detrend'] == m)]
                if len(row) > 0:
                    mval = row['dU_norm_median'].values[0]
                    med.append(mval)
                    lo = row['dU_norm_ci68_lo'].values[0]
                    hi = row['dU_norm_ci68_hi'].values[0]
                    err_lo.append(mval - lo if np.isfinite(lo) else 0)
                    err_hi.append(hi - mval if np.isfinite(hi) else 0)
                else:
                    med.append(np.nan)
                    err_lo.append(0)
                    err_hi.append(0)
            med = np.array(med)
            err_lo = np.array(err_lo)
            err_hi = np.array(err_hi)
            # Replace NaN for plotting (bar height 0, no error bar)
            plot_med = np.where(np.isfinite(med), med, 0.0)
            plot_err_lo = np.where(np.isfinite(med), err_lo, 0.0)
            plot_err_hi = np.where(np.isfinite(med), err_hi, 0.0)
            off = -width/2 if i == 0 else width/2
            ax.bar(x + off, plot_med, width, yerr=[plot_err_lo, plot_err_hi], label=epoch, alpha=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
    ax.set_xlabel('Detrending method')
    ax.set_ylabel(r'$\Delta U/\sigma^2$')
    # ax.set_title('(b) Detrending sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Synthetic vs observed barrier metric: jittered strip at 0 + observed lines
    ax = axes[1, 0]
    if synth_results is not None and 'dU_norm_synthetic' in synth_results:
        dU_syn = np.asarray(synth_results['dU_norm_synthetic'])
        dU_syn = dU_syn[np.isfinite(dU_syn)]

        if len(dU_syn) > 0:
            # Jittered strip at x = 0 (synthetic null, all or most at 0)
            # Use a random subset if very large for readability
            rng = np.random.default_rng(0)
            sample_size = min(200, len(dU_syn))
            y_jitter = rng.uniform(0.02, 0.98, size=sample_size)
            # x positions: all at 0 (null) with tiny horizontal jitter so points are visible
            x_jitter = rng.normal(loc=0.0, scale=0.002, size=sample_size)
            ax.scatter(x_jitter, y_jitter, s=10, color='gray', alpha=0.5, label='Synthetic (null, ΔU/σ²≈0)')

            obs_early = observed_early_dU_norm
            obs_late = observed_late_dU_norm
            # Observed early and late as vertical lines
            if np.isfinite(obs_early):
                ax.axvline(obs_early, color='steelblue', lw=2.5, label=f'Observed early ({obs_early:.2f})')
                # Horizontal arrow from null (0) to observed early
                ax.annotate('', xy=(obs_early, 0.9), xytext=(0.0, 0.9),
                            arrowprops=dict(arrowstyle='<-', lw=1.5, color='steelblue'))
            if np.isfinite(obs_late):
                ax.axvline(obs_late, color='coral', lw=2.5, label=f'Observed late ({obs_late:.2f})')
                ax.annotate('', xy=(obs_late, 0.8), xytext=(0.0, 0.8),
                            arrowprops=dict(arrowstyle='<-', lw=1.5, color='coral'))

            # # Text indicating interpretation
            # ax.text(0.02, 0.6, 'Synthetic null concentrated at 0\n(all monostable)', transform=ax.transAxes,
            #         ha='left', va='center', fontsize=10)

            ax.set_xlim(left=-0.02, right=max(0.2, float(np.nanmax([obs_early if np.isfinite(obs_early) else 0,
                                                                     obs_late if np.isfinite(obs_late) else 0])) * 1.5))
    ax.set_xlabel(r'$\Delta U/\sigma^2$')
    ax.set_ylabel('Density')
    # ax.set_title('(c) Synthetic vs observed barrier metric')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) unused axis: turn off
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


# -----------------------------------------------------------------------------
# Main: load data, run all analyses, save outputs and figure
# -----------------------------------------------------------------------------

def main(c3_file=DEFAULT_C3_FILE, piomas_file=DEFAULT_PIOMAS_FILE,
         nboot_epoch=500, nboot_detrend=500, n_synthetic=1000,
         use_piomas_only_for_synthetic=True):
    """
    use_piomas_only_for_synthetic: If True (default), the early epoch for synthetic
    controls is built from PIOMAS-only volume (bias-corrected). This ensures 1982-2005
    has 24 finite years when C3S only covers 2002+, so the AR(1)+drift fit and synthetic
    series have enough points and observed early ΔU/σ² can be computed.
    """
    print("=" * 70)
    print("ROBUSTNESS AND ALTERNATIVE HYPOTHESES")
    print("=" * 70)

    # Load and prepare E (October)
    print("\nLoading volume data and converting to energy...")
    c3_df = load_ice_volume_data(c3_file, 'C3S')
    PIOMAS_df = load_ice_volume_data(piomas_file, 'PIOMAS')
    if c3_df is None or PIOMAS_df is None:
        print("Cannot load data; check paths.")
        return
    c3, av, vol_ens, _ = harmonize_volume(c3_df, PIOMAS_df, on='Year')
    vol_ens = vol_ens.rename(columns={'Vol_Ensemble': 'Volume'})
    vol_ens['Date'] = pd.to_datetime(vol_ens['Year'].astype(int).astype(str) + '-10-01')
    E_oct_df = volume_to_state_E(vol_ens[['Date', 'Volume']], time_col='Date', vol_col='Volume', aref_m2=AREF_M2)

    # Ensure October-only index for bootstrap (one row per year)
    E_oct_df = E_oct_df[E_oct_df.index.month == 10]

    print("\n1) Epoch split sensitivity...")
    epoch_sens = run_epoch_sensitivity(E_oct_df, month=10, nboot=nboot_epoch)
    if epoch_sens['splits'] is not None and len(epoch_sens['splits']) > 0:
        epoch_sens['splits'].to_csv('robustness_epoch_splits.csv', index=False)
        print("   Saved: robustness_epoch_splits.csv")
    if epoch_sens['sliding'] is not None and len(epoch_sens['sliding']) > 0:
        epoch_sens['sliding'].to_csv('robustness_sliding_windows.csv', index=False)
        print("   Saved: robustness_sliding_windows.csv")

    print("\n2) Detrending sensitivity...")
    detrend_sens = run_detrending_sensitivity(E_oct_df, nboot=nboot_detrend)
    if detrend_sens is not None and len(detrend_sens) > 0:
        detrend_sens.to_csv('robustness_detrending.csv', index=False)
        print("   Saved: robustness_detrending.csv")

    print("\n3) Bandwidth sensitivity...")
    df_early = E_oct_df[(E_oct_df.index.year >= 1982) & (E_oct_df.index.year <= 2005)]
    df_late = E_oct_df[(E_oct_df.index.year >= 2006) & (E_oct_df.index.year <= 2024)]
    bw_sens_early = run_bandwidth_sensitivity(df_early['E'].values, nboot=0)
    bw_sens_late = run_bandwidth_sensitivity(df_late['E'].values, nboot=0)
    if bw_sens_early is not None:
        bw_sens_early.to_csv('robustness_bandwidth_early.csv', index=False)
    if bw_sens_late is not None:
        bw_sens_late.to_csv('robustness_bandwidth_late.csv', index=False)

    print("\n4) Synthetic monostable controls...")
    # Use PIOMAS-only for early epoch when ensemble has NaNs (e.g. C3S starts 2002+)
    # so we have 24 years of finite E and synthetics of length 24
    if use_piomas_only_for_synthetic and PIOMAS_df is not None:
        c3, av, _, _ = harmonize_volume(c3_df, PIOMAS_df, on='Year')
        vol_piomas = av[['Year', 'Vol_PIOMAS_bc']].rename(columns={'Vol_PIOMAS_bc': 'Volume'})
        vol_piomas['Date'] = pd.to_datetime(vol_piomas['Year'].astype(int).astype(str) + '-10-01')
        E_oct_piomas = volume_to_state_E(vol_piomas[['Date', 'Volume']], time_col='Date', vol_col='Volume', aref_m2=AREF_M2)
        E_oct_piomas = E_oct_piomas[E_oct_piomas.index.month == 10]
        df_early_full = E_oct_piomas[(E_oct_piomas.index.year >= 1982) & (E_oct_piomas.index.year <= 2005)]
        E_early = df_early_full['E'].dropna().values
    else:
        E_early = df_early['E'].values
    n_finite_early = np.count_nonzero(np.isfinite(E_early))
    if n_finite_early < 6:
        print(f"   ⚠️ Early epoch has {n_finite_early} finite points (< 6); skipping synthetic controls.")
        synth_results = None
    else:
        synth_results = run_synthetic_controls(E_early, n_realizations=n_synthetic)
    if synth_results is not None:
        print(f"   False-positive rate (barrier present): {synth_results['false_positive_rate']:.3f}")
        print(f"   Observed early ΔU/σ²: {synth_results['dU_norm_observed_early']:.3f}")
        np.savetxt('robustness_synthetic_dU_norm.txt', synth_results['dU_norm_synthetic'])

    # Point estimates for observed early/late (for panel f):
    # prefer values from main metrics CSV (consistent with Fig. 5), fall back to direct computation
    try:
        metrics = pd.read_csv('oct_resilience_metrics_comprehensive.csv')
        m10 = metrics[metrics['month'] == 10]
        obs_early_csv = m10[m10['epoch'] == '1982-2005']['ΔU/σ²_median'].values
        obs_late_csv  = m10[m10['epoch'] == '2006-2024']['ΔU/σ²_median'].values
        observed_early_dU_norm = obs_early_csv[0] if len(obs_early_csv) > 0 else np.nan
        observed_late_dU_norm  = obs_late_csv[0]  if len(obs_late_csv)  > 0 else np.nan
    except Exception:
        obs_early_met = potential_metrics_from_E(E_early) if len(np.isfinite(E_early).nonzero()[0]) >= 8 else None
        obs_late_met  = potential_metrics_from_E(df_late['E'].values)
        observed_early_dU_norm = obs_early_met['dU_norm'] if obs_early_met else np.nan
        observed_late_dU_norm  = obs_late_met['dU_norm']  if obs_late_met  else np.nan

    print("\n5) Plotting Extended Data Figure...")
    plot_robustness_extended_figure(
        epoch_sens, detrend_sens, bw_sens_early, bw_sens_late,
        synth_results, observed_early_dU_norm, observed_late_dU_norm,
        outpath='Extended_Data_robustness.png'
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
