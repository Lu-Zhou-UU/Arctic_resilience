import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")

# ---- Physical constants ----
RHO_I = 917.0                # kg m^-3, density of ice
L_I   = 3.34e5               # J kg^-1, latent heat of fusion
KAPPA = RHO_I * L_I          # J m^-3, latent energy density
print(f"Physical constants: ρ_i = {RHO_I} kg/m³, L_i = {L_I:.2e} J/kg")
print(f"Latent energy density κ = ρ_i × L_i = {KAPPA:.2e} J/m³")

# ---- Reference area ----
AREF_KM2 = 25.0 * 25.0       # 625 km^2
AREF_M2  = AREF_KM2 * 1e6    # = 6.25e8 m^2
print(f"Reference area A_ref = {AREF_KM2} km² = {AREF_M2:.2e} m²\n")


def load_ice_volume_data(filename, dataset_name):
    """Load ice volume data from txt file"""
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Volume'])
        print(f"✓ Successfully loaded {dataset_name} ice volume data")
        print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"  Volume range: {df['Volume'].min():.1f} - {df['Volume'].max():.1f} thousand km³")
        return df
    except Exception as e:
        print(f"❌ Error loading {dataset_name} ice volume data: {e}")
        return None


def harmonize_volume(c3_df, PIOMAS_df, on='Year'):
    """
    Return (c3, PIOMAS_bc, ensemble) dataframes with aligned index and bias-corrected PIOMAS.
    Performs linear bias correction: PIOMAS_bc = a × PIOMAS + b to match C3S
    """
    c3 = c3_df[[on, 'Volume']].rename(columns={'Volume':'Vol_C3S'}).dropna().copy()
    av = PIOMAS_df[[on, 'Volume']].rename(columns={'Volume':'Vol_PIOMAS'}).dropna().copy()
    df = pd.merge(c3, av, on=on, how='inner')

    if len(df) >= 5:
        A = np.vstack([df['Vol_PIOMAS'].values, np.ones(len(df))]).T
        a, b = np.linalg.lstsq(A, df['Vol_C3S'].values, rcond=None)[0]
        print(f"  Bias correction: PIOMAS_bc = {a:.3f} × PIOMAS + {b:.3f}")
    else:
        a, b = 1.0, 0.0
        print(f"  ⚠️ Insufficient overlap ({len(df)} years); no bias correction applied")

    av['Vol_PIOMAS_bc'] = a * av['Vol_PIOMAS'] + b
    u = pd.merge(c3, av[[on,'Vol_PIOMAS_bc']], on=on, how='outer').sort_values(on)
    u['Vol_Ensemble'] = u[['Vol_C3S','Vol_PIOMAS_bc']].mean(axis=1)

    return c3, av, u, (a, b)


def volume_to_state_E(df, time_col, vol_col, aref_m2=AREF_M2, kappa=KAPPA):
    """
    Convert sea-ice volume (km³) to energy per unit area (J m⁻²)
    using E = -ρ_i L_i × (V / A_ref) = -κ × h
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.set_index(time_col).sort_index()

    V_m3 = out[vol_col].values * 1e9
    h_m  = V_m3 / aref_m2
    E_Jm2 = -kappa * h_m
    out['E'] = E_Jm2

    print(f"✓ Converted {len(out)} records to energy state variable")
    print(f"  E range: {out['E'].min():.2e} to {out['E'].max():.2e} J m⁻²")
    return out[['E']]


def monthly_anomalies(E_series):
    """Remove monthly climatology from E(t) to obtain E'(t)"""
    df = E_series.to_frame('E')
    df['month'] = df.index.month
    clim = df.groupby('month')['E'].mean()
    df['E_anom'] = df.apply(lambda r: r['E'] - clim.loc[r['month']], axis=1)
    return df[['E','E_anom']]


def estimate_ar1_sigma(E_values, method='innovation'):
    """
    Estimate noise scale σ from time series.
    
    Methods:
    - 'innovation': AR(1) innovation std (default)
    - 'std': simple std dev
    - 'residual': std of AR(1) residuals
    """
    x = np.asarray(E_values)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    
    if method == 'std':
        return np.std(x, ddof=1)
    
    # AR(1) fit
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    phi = np.clip(phi, -0.99, 0.99)
    var_x = np.var(x, ddof=1)
    
    if method == 'innovation':
        # Innovation variance: σ² = Var(X) × (1 - φ²)
        return np.sqrt(var_x * (1 - phi**2))
    
    elif method == 'residual':
        # Residuals from AR(1) fit
        residuals = x[1:] - phi * x[:-1]
        return np.std(residuals, ddof=1)
    
    return np.nan


def kde_empirical_potential(E_values, sigma=1.0, grid=None, bw='scott'):
    """
    Build U_emp(E) = -(σ²/2) × ln P(E) via KDE.
    Returns grid, U, kde, and spline interpolator.
    """
    x = np.asarray(E_values)
    x = x[~np.isnan(x)]
    if grid is None:
        lo, hi = np.percentile(x, [1, 99])
        grid = np.linspace(lo, hi, 500)
    kde = stats.gaussian_kde(x, bw_method=bw)
    p = np.maximum(kde(grid), 1e-300)
    U = -(sigma**2 / 2.0) * np.log(p)
    U -= np.min(U)  # Normalize to min=0
    
    spl = UnivariateSpline(grid, U, s=0, k=4)
    
    return grid, U, kde, spl


def extrema_from_potential(grid, U, spl, E_values=None, kde=None):
    """Identify minima and saddles in potential U(E)."""
    # Adaptive prominence based on potential range
    U_range = np.max(U) - np.min(U)
    prominence = max(0.001, U_range * 0.001)  # 0.1% of range
    width = max(1, len(grid) // 250)  # Adaptive width
    
    min_idx, _ = find_peaks(-U, prominence=prominence, width=width)
    max_idx, _ = find_peaks(U, prominence=prominence, width=width)

    d2U = spl.derivative(n=2)

    # 2) Compute data mode on the same grid (if data provided)
    E_mode = None
    if E_values is not None:
        x = np.asarray(E_values)
        x = x[np.isfinite(x)]
        if len(x) >= 5:
            try:
                if kde is None:
                    kde = stats.gaussian_kde(x, bw_method='scott')
                p = kde(grid)                         # density on the plotting grid
                E_mode = grid[np.argmax(p)]          # mode in E-units
            except Exception:
                E_mode = None

    # 3) Build lists with curvature
    minima = []
    for i in min_idx:
        Emin = grid[i]; Umin = U[i]; curv_min = d2U(Emin)
        minima.append((Emin, Umin, curv_min))

    saddles = []
    for i in max_idx:
        Esad = grid[i]; Usad = U[i]; curv_sad = d2U(Esad)
        saddles.append((Esad, Usad, curv_sad))

    # 4) Prefer ice-covered minimum near the data mode (if available)
    if minima:
        if E_mode is not None:
            # keep only E<0 if present; otherwise fall back to all minima
            mins_ic = [m for m in minima if m[0] < 0]
            pool = mins_ic if mins_ic else minima
            pool.sort(key=lambda m: abs(m[0] - E_mode))
            # Put the chosen minimum first in the list
            chosen = pool[0]
            # reorder minima so chosen is first
            minima = [chosen] + [m for m in minima if m is not chosen]

        else:
            # still prefer an ice-covered well if any exist
            mins_ic = [m for m in minima if m[0] < 0]
            if mins_ic:
                chosen = sorted(mins_ic, key=lambda m: m[0])[-1]  # closest to 0 from negative side
                minima = [chosen] + [m for m in minima if m is not chosen]

    # 5) No change for saddles list; selection (first to the right of Emin)
    return minima, saddles


def bootstrap_metrics_month(epoch_df, month, nboot=1000, block_size=3, 
                           sigma_method='innovation', bw_method='scott'):
    """
    Block bootstrap for uncertainty quantification.
    
    Returns array of shape (nboot, 3): [ΔU, U''_min, ΔU/σ²]
    """
    # Extract month data
    month_df = epoch_df[epoch_df.index.month == month]
    years = sorted(month_df.index.year.unique())
    
    if len(years) < 8:
        return np.array([])
    
    draws = []
    for _ in range(nboot):
        # Block resampling
        samp_years = []
        while len(samp_years) < len(years):
            y0 = np.random.choice(years)
            block = list(range(y0, min(y0 + block_size, years[-1] + 1)))
            samp_years.extend(block)
        samp_years = samp_years[:len(years)]
        
        # Extract data
        sub = month_df[month_df.index.year.isin(samp_years)]['E'].values
        if len(sub) < 8:
            continue
        
        # Compute potential
        sig = estimate_ar1_sigma(sub, method=sigma_method)
        if not np.isfinite(sig) or sig <= 0:
            continue
            
        try:
            grid, U, kde, spl = kde_empirical_potential(sub, sigma=sig, bw=bw_method)
            mins, sads = extrema_from_potential(grid, U, spl, E_values=sub)
            
            if not mins or not sads:
                continue
            
            E_min, U_min, curv_min = mins[0]
            E_sad, U_sad, _ = sads[0]
            dU = U_sad - U_min
            
            draws.append([dU, curv_min, dU / (sig**2)])
        except:
            continue
    
    return np.array(draws)


def compute_sigma_sensitivity(E_values):
    """
    Compute σ using three methods for sensitivity analysis.
    Returns dict with keys: 'innovation', 'std', 'residual'
    """
    sigmas = {}
    for method in ['innovation', 'std', 'residual']:
        sigmas[method] = estimate_ar1_sigma(E_values, method=method)
    return sigmas


def compute_bandwidth_sensitivity(E_values, sigma,
                                  methods=('scott', 'silverman', 'scott*0.8', 'scott*1.2')):
    """
    Build U(E) for several KDE bandwidth choices.
    Returns: dict {label: (grid, U, spline)}
    """
    x = np.asarray(E_values)
    x = x[np.isfinite(x)]
    out = {}
    if x.size < 5 or np.allclose(np.var(x), 0.0):
        return out  # not enough spread for KDE

    # common energy grid
    lo, hi = np.percentile(x, [1, 99])
    grid = np.linspace(lo, hi, 500)

    try:
        kde_scott = stats.gaussian_kde(x, bw_method='scott')
        scott_factor = kde_scott.factor  # baseline scalar
    except Exception:
        return out

    # assemble KDEs
    kdes = {
        'scott'     : kde_scott,
        'silverman' : stats.gaussian_kde(x, bw_method='silverman'),
        'scott*0.8' : stats.gaussian_kde(x, bw_method=scott_factor * 0.8),
        'scott*1.2' : stats.gaussian_kde(x, bw_method=scott_factor * 1.2),
    }

    # build potentials
    for label, kde in kdes.items():
        try:
            p = np.maximum(kde(grid), 1e-300)
            U = -(sigma**2 / 2.0) * np.log(p)
            U -= U.min()  # normalize so min(U)=0
            spl = UnivariateSpline(grid, U, s=0, k=4)
            out[label] = (grid, U, spl)
        except Exception:
            continue

    return out



def monthly_empirical_potentials_by_epoch(E_df, epochs, use_sigma_from_ar1=True, 
                                         min_points=10, compute_bootstrap=True,
                                         nboot=1000):
    """
    Compute empirical potentials with bootstrap uncertainty and sensitivity analyses.
    """
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
            
            # Primary analysis with AR(1) innovation σ
            sigma_m = estimate_ar1_sigma(month_data, method='innovation')
            
            if not np.isfinite(sigma_m) or sigma_m <= 0:
                continue
            
            # Main potential
            grid, U, kde, spl = kde_empirical_potential(month_data, sigma=sigma_m)
            minima, saddles = extrema_from_potential(grid, U, spl, E_values=month_data)
            
            # After calling extrema_from_potential(...)
            if minima:
                E_s1, U_s1, curv_s1 = minima[0]  # preferred ice-covered min
                # choose first saddle to the right of Emin
                s_right = [s for s in saddles if s[0] > E_s1]
                if s_right:
                    E_s2, U_s2, curv_s2 = sorted(s_right, key=lambda s: s[0])[0]
                else:
                    E_s2 = U_s2 = curv_s2 = np.nan
            else:
                E_s1 = U_s1 = curv_s1 = np.nan
                E_s2 = U_s2 = curv_s2 = np.nan
            
            dU = U_s2 - U_s1 if np.isfinite(U_s2) else np.nan
            
            # Store main results
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
                'dU': dU
            }
            
            # Bootstrap uncertainty
            if compute_bootstrap:
                print(f"    Computing bootstrap for {epoch_label}, month {m}...")
                boot_draws = bootstrap_metrics_month(epoch_df, m, nboot=nboot)
                if len(boot_draws) > 0:
                    # Compute percentiles: 16%, 50%, 84% (68% CI) and 2.5%, 97.5% (95% CI)
                    pct = np.percentile(boot_draws, [2.5, 16, 50, 84, 97.5], axis=0)
                    result['bootstrap'] = {
                        'draws': boot_draws,
                        'dU': {'median': pct[2,0], 'ci68': (pct[1,0], pct[3,0]), 'ci95': (pct[0,0], pct[4,0])},
                        'curv': {'median': pct[2,1], 'ci68': (pct[1,1], pct[3,1]), 'ci95': (pct[0,1], pct[4,1])},
                        'dU_norm': {'median': pct[2,2], 'ci68': (pct[1,2], pct[3,2]), 'ci95': (pct[0,2], pct[4,2])}
                    }
            
            # Sigma sensitivity
            sigmas = compute_sigma_sensitivity(month_data)
            result['sigma_sensitivity'] = sigmas
            dU_norm_sens = {}
            for method, sig in sigmas.items():
                if np.isfinite(sig) and sig > 0 and np.isfinite(dU):
                    dU_norm_sens[method] = dU / (sig**2)
            result['dU_norm_sensitivity'] = dU_norm_sens
            
            # Bandwidth sensitivity
            bw_results = compute_bandwidth_sensitivity(month_data, sigma_m)
            result['bandwidth_sensitivity'] = bw_results
            
            results[epoch_label][m] = result
    
    return results


from matplotlib.ticker import FuncFormatter

def plot_comprehensive_potential(results, month=10, figsize=(16, 12)):
    """
    Comprehensive multi-panel plot showing:
    - Panel A: Empirical potentials U(E)
    - Panel B: Curvature U''(E)
    - Panel C: ΔU/σ² (bars) and U''(E_min) (dots) with CIs
    - Panel D: Bandwidth sensitivity for U(E)
    """
    month_name = {4:'April', 10:'October'}.get(month, f'Month {month}')

    # --- helpers for unit formatting ---
    scale_E = 1e12          # show E in ×10^12 J m^-2
    scale_U = 1e22          # show U in ×10^22 J^2 m^-4
    fmt_E = FuncFormatter(lambda v, _: f"{v/scale_E:.1f}")
    fmt_U = FuncFormatter(lambda v, _: f"{v/scale_U:.1f}")

    # Filter data for this month
    epoch_data = {ep: months[month] for ep, months in results.items() if month in months}
    if not epoch_data:
        print(f"No data for {month_name}")
        return None

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(epoch_data)))

    # Shared x-limits in E
    xlim_E = [np.inf, -np.inf]
    for r in epoch_data.values():
        g = r['grid']
        xlim_E[0] = min(xlim_E[0], g.min())
        xlim_E[1] = max(xlim_E[1], g.max())

    # Figure layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.5], width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :])  # U(E)
    ax2 = fig.add_subplot(gs[1, :])  # U''(E)
    ax3 = fig.add_subplot(gs[2, 0])  # ΔU/σ² + U''(E_min)
    ax4 = fig.add_subplot(gs[2, 1])  # bandwidth sensitivity

    # ---------------- Panel A: U(E) ----------------
    for idx, (epoch_label, r) in enumerate(epoch_data.items()):
        grid, U = r['grid'], r['U']
        ax1.plot(grid, U, lw=2.5, alpha=0.9, color=colors[idx], label=epoch_label)

        # minima & saddle
        E_s1, U_s1 = r['E_s1'], r['Upp_s1']
        ax1.plot(E_s1, U_s1, 'o', ms=12, color=colors[idx], mec='k', mew=2, zorder=10)

        if np.isfinite(r['E_s2']):
            E_s2, U_s2 = r['E_s2'], r['Upp_s2']
            ax1.plot(E_s2, U_s2, 's', ms=12, color=colors[idx], mec='k', mew=2, zorder=10)

            # barrier glyph
            ax1.annotate("", xy=(E_s2, U_s2), xytext=(E_s1, U_s1),
                         arrowprops=dict(arrowstyle='<->', lw=2.5, color=colors[idx], alpha=0.7))

            # annotation (kept compact)
            dU = r['dU']; dU_norm = dU / (r['sigma_m']**2)
            xm, ym = 0.5*(E_s1+E_s2), 0.5*(U_s1+U_s2)

            # You already have: dU (J^2 m^-4) and dU_norm (unitless)
            dU_scaled = dU / 1e22  # match y-axis scale
            ax1.text(xm, ym,
                rf'$\Delta U={dU_scaled:.2f}\times 10^{{22}}$'+'\n'+rf'$\Delta U/\sigma^2={dU_norm:.2f}$',
                ha='center', va='bottom', fontsize=15, fontweight='bold',
                bbox=dict(fc='white', ec=colors[idx], boxstyle='round,pad=0.5', alpha=0.9))


            
            # ax1.text(xm, ym, f"ΔU={dU:.2e}\nΔU/σ²={dU_norm:.2f}",
            #          ha='center', va='bottom', fontsize=15, fontweight='bold',
            #          bbox=dict(fc='white', ec=colors[idx], boxstyle='round,pad=0.5', alpha=0.9))

    ax1.set_xlim(xlim_E)
    ax1.xaxis.set_major_formatter(fmt_E)
    ax1.yaxis.set_major_formatter(fmt_U)
    # Panel A labels (U vs E)
    ax1.set_xlabel(r'$E\;(\times 10^{12}\ \mathrm{J\,m^{-2}})$', fontsize=15, fontweight='bold')
    ax1.set_ylabel(r'$U(E)\;(\times 10^{22}\ \mathrm{J^{2}\,m^{-4}})$', fontsize=15, fontweight='bold')
    # ax1.set_title(f'(a) {month_name} Empirical Potential by Epoch', fontsize=15, fontweight='bold', loc='left')

    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, ls='--', alpha=0.3)
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    # ---------------- Panel B: U''(E) ----------------
    for idx, (epoch_label, r) in enumerate(epoch_data.items()):
        grid, spl = r['grid'], r['spl']
        curv_curve = spl.derivative(2)(grid)
        ax2.plot(grid, curv_curve, lw=2.5, alpha=0.85, color=colors[idx], label=epoch_label)

        # marker at Emin and vline at Esad
        ax2.plot(r['E_s1'], r['Upp_prime_prime_s1'], 'o', ms=10, color=colors[idx], mec='black', mew=1.5)
        if np.isfinite(r['E_s2']):
            ax2.axvline(r['E_s2'], ls='--', lw=1.5, color=colors[idx], alpha=0.6)

    ax2.set_xlim(xlim_E)
    ax2.xaxis.set_major_formatter(fmt_E)
    ax2.axhline(0, color='gray', ls=':', lw=1.5, alpha=0.6)
    ax2.set_xlabel(r'$E\;(\times 10^{12}\ \mathrm{J\,m^{-2}})$', fontsize=15, fontweight='bold')
    ax2.set_ylabel(r'$U^{\prime\prime}(E)\ \mathrm{(dimensionless)}$', fontsize=15, fontweight='bold')
    # ax2.set_title(f'(b) {month_name} Potential Curvature', fontsize=15, fontweight='bold', loc='left')

    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    # ---------------- Panel C: ΔU/σ² + U″(Emin) ----------------
    epochs = list(epoch_data.keys())
    x_pos = np.arange(len(epochs))

    dU_norm_vals, dU_norm_cis = [], []
    curv_vals, curv_cis = [], []

    for ep in epochs:
        r = epoch_data[ep]
        if 'bootstrap' in r:
            dU_norm_vals.append(r['bootstrap']['dU_norm']['median'])
            dU_norm_cis.append([*r['bootstrap']['dU_norm']['ci68']])
            curv_vals.append(r['bootstrap']['curv']['median'])
            curv_cis.append([*r['bootstrap']['curv']['ci68']])
        else:
            dU_norm = r['dU']/(r['sigma_m']**2) if (np.isfinite(r['dU']) and r['sigma_m']>0) else np.nan
            dU_norm_vals.append(dU_norm); dU_norm_cis.append([dU_norm, dU_norm])
            curv_vals.append(r['Upp_prime_prime_s1']); curv_cis.append([r['Upp_prime_prime_s1'], r['Upp_prime_prime_s1']])

    # Print mean of U''(Emin) and range
    curv_vals_array = np.array(curv_vals)
    curv_mean = np.nanmean(curv_vals_array)
    curv_min = np.nanmin(curv_vals_array)
    curv_max = np.nanmax(curv_vals_array)
    print(f"U''(Emin) statistics: mean = {curv_mean:.4f}, range = [{curv_min:.4f}, {curv_max:.4f}]")
    print(f"The curvature at the ice-covered minimum changes from {curv_min:.4f} to {curv_max:.4f}")

    ax3_twin = ax3.twinx()

    # bars: ΔU/σ² (unitless)
    bars = ax3.bar(x_pos, dU_norm_vals, color='steelblue', alpha=0.7,
                   edgecolor='black', linewidth=1.5, label='ΔU/σ² (resilience)')
    if dU_norm_cis:
        yerr = np.array([[dU_norm_vals[i] - dU_norm_cis[i][0],
                          dU_norm_cis[i][1] - dU_norm_vals[i]] for i in range(len(epochs))]).T
        ax3.errorbar(x_pos, dU_norm_vals, yerr=yerr, fmt='none',
                     ecolor='black', capsize=5, capthick=2, alpha=0.8)

    # dots: U″(E_min) (dimensionless)
    ax3_twin.plot(x_pos, curv_vals, 'o', ms=12, color='darkorange',
                  mec='black', mew=2, label='U″(E_min)')
    if curv_cis:
        yerr = np.array([[curv_vals[i] - curv_cis[i][0],
                          curv_cis[i][1] - curv_vals[i]] for i in range(len(epochs))]).T
        ax3_twin.errorbar(x_pos, curv_vals, yerr=yerr, fmt='none',
                          ecolor='darkorange', capsize=5, capthick=2, alpha=0.8)

    ax3.set_xticks(x_pos); ax3.set_xticklabels(epochs, fontsize=15)
    # ax3.set_ylabel('ΔU/σ² (unitless)', fontsize=15, fontweight='bold', color='steelblue')
    # ax3_twin.set_ylabel('U″(E_min) (dimensionless)', fontsize=15, fontweight='bold', color='darkorange')

    ax3.set_ylabel(r'$\Delta U/\sigma^{2}$ (unitless)', fontsize=15, fontweight='bold', color='steelblue')
    ax3_twin.set_ylabel(r'$U^{\prime\prime}(E_{\min})$ (dimensionless)', fontsize=15, fontweight='bold', color='darkorange')
    # ax3.set_title(f'(c) {month_name} Resilience Metrics by Epoch', fontsize=15, fontweight='bold', loc='left')

    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3_twin.tick_params(axis='y', labelcolor='darkorange')
    ax3.grid(True, axis='y', ls='--', alpha=0.3)
    # Panel C labels (summary)
    


    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labels1+labels2, loc='upper right', framealpha=0.95)

    # ---------------- Panel D: Bandwidth sensitivity ----------------
    first_epoch = epochs[0]
    r = epoch_data[first_epoch]
    if 'bandwidth_sensitivity' in r:
        for label, (grid, U, _spl) in r['bandwidth_sensitivity'].items():
            lw = 2.5 if label == 'scott' else 1.5
            alpha = 0.9 if label == 'scott' else 0.35
            ax4.plot(grid, U, lw=lw, alpha=alpha, label=label)

    ax4.xaxis.set_major_formatter(fmt_E)
    ax4.yaxis.set_major_formatter(fmt_U)
    ax4.set_xlabel(r'$E\;(\times 10^{12}\ \mathrm{J\,m^{-2}})$', fontsize=15, fontweight='bold')
    ax4.set_ylabel(r'$U(E)\;(\times 10^{22}\ \mathrm{J^{2}\,m^{-4}})$', fontsize=15, fontweight='bold')
    # ax4.set_title(f'(d) Bandwidth Sensitivity\n({first_epoch})', fontsize=15, fontweight='bold', loc='left')
    ax4.legend(loc='best', framealpha=0.95)
    ax4.grid(True, ls='--', alpha=0.3)
    ax4.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

    plt.tight_layout()
    return fig

def export_sigma_consistency_table(results, month=10,
                                   output_path='sigma_consistency_october.csv'):
    """
    Export a small table comparing observationally estimated σ (AR(1) innovation)
    per epoch to the published basin-mean reference (~7 W m⁻²) for the Methods
    'consistency check' sentence. σ here is in E units (J m⁻²).
    """
    rows = []
    for epoch_label, months in results.items():
        if month not in months:
            continue
        r = months[month]
        sig = r.get('sigma_m', np.nan)
        if not np.isfinite(sig):
            continue
        rows.append({
            'epoch': epoch_label,
            'month': int(month),
            'sigma_innovation_Jm2': sig,
            'sigma_innovation_1e12Jm2': sig / 1e12,
        })

    if not rows:
        print("  ⚠️ No sigma values to export for consistency table.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.6e')

    # Simple per-epoch summary for direct use in Methods
    by_epoch = df.groupby('epoch')['sigma_innovation_1e12Jm2'].agg(['mean', 'std', 'count'])
    print(f"\n✓ σ (AR(1) innovation) consistency check → {output_path}")
    print("  October σ_E (mean ± 1 s.d.) in units of 10¹² J m⁻²:")
    for ep, row in by_epoch.iterrows():
        print(f"    {ep}: {row['mean']:.2f} ± {row['std']:.2f} (N={int(row['count'])})")
    print("  Note: these σ are in energy units E (J m⁻²);")
    print("        7 W m⁻² is the stochastic forcing amplitude in the EBM.")
    return df

def export_comprehensive_metrics(results, filename='resilience_metrics_complete.csv'):
    """
    Export comprehensive metrics table with bootstrap CIs.
    """
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
            
            # Add normalized barrier
            if np.isfinite(r['dU']) and r['sigma_m'] > 0:
                row['ΔU/σ²'] = r['dU'] / (r['sigma_m']**2)
            else:
                row['ΔU/σ²'] = np.nan
            
            # Add bootstrap CIs if available
            if 'bootstrap' in r:
                boot = r['bootstrap']
                row['ΔU_median'] = boot['dU']['median']
                row['ΔU_ci68_lo'] = boot['dU']['ci68'][0]
                row['ΔU_ci68_hi'] = boot['dU']['ci68'][1]
                row['ΔU_ci95_lo'] = boot['dU']['ci95'][0]
                row['ΔU_ci95_hi'] = boot['dU']['ci95'][1]
                
                row["U''_min_median"] = boot['curv']['median']
                row["U''_min_ci68_lo"] = boot['curv']['ci68'][0]
                row["U''_min_ci68_hi"] = boot['curv']['ci68'][1]
                
                row['ΔU/σ²_median'] = boot['dU_norm']['median']
                row['ΔU/σ²_ci68_lo'] = boot['dU_norm']['ci68'][0]
                row['ΔU/σ²_ci68_hi'] = boot['dU_norm']['ci68'][1]
            
            # Add sigma sensitivity
            if 'sigma_sensitivity' in r:
                for method, sig in r['sigma_sensitivity'].items():
                    row[f'σ_{method}'] = sig
            
            # Add ΔU/σ² sensitivity
            if 'dU_norm_sensitivity' in r:
                for method, val in r['dU_norm_sensitivity'].items():
                    row[f'ΔU/σ²_{method}'] = val
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, float_format='%.6e')
    print(f"\n✓ Saved comprehensive metrics to: {filename}")
    return df


def rolling_ews_annual(E_series, window_years):
    """Rolling variance and lag-1 AC on annual series"""
    df = E_series.to_frame('E').copy()
    df['year'] = df.index.year
    rows = []
    for end in range(window_years-1, len(df)):
        sub = df.iloc[end-window_years+1 : end+1]['E'].values
        yearsub = int(df.iloc[end]['year'])
        if len(sub) >= 5:
            var = np.var(sub, ddof=1)
            ac1 = np.corrcoef(sub[:-1], sub[1:])[0,1] if len(sub) > 2 else np.nan
            rows.append({'year': yearsub, 'var': var, 'ac1': ac1})
    return pd.DataFrame(rows)


# ==========================================
# MAIN ANALYSIS SCRIPT
# ==========================================

print("=" * 80)
print("COMPREHENSIVE SEA ICE RESILIENCE ANALYSIS")
print("With Bootstrap Uncertainty & Sensitivity Tests")
print("=" * 80 + "\n")

# Load data
print("Loading ice volume data...\n")
c3_df = load_ice_volume_data('/Volumes/Yotta_1/C3_ice_volume_October.txt', 'C3S')
PIOMAS_df = load_ice_volume_data('/Volumes/Yotta_1/PIOMAS_ice_volume_October.txt', 'PIOMAS')

# Harmonize volumes
print("\nHarmonizing C3S and PIOMAS datasets...")
c3, av, vol_ens, (a, b) = harmonize_volume(c3_df, PIOMAS_df, on='Year')

# Prepare ensemble
vol_ens = vol_ens.rename(columns={'Vol_Ensemble':'Volume'})
vol_ens['Date'] = pd.to_datetime(vol_ens['Year'].astype(int).astype(str) + '-10-01')

# Convert to energy state variable
print("\nConverting volume to energy state variable...")
E_oct_df = volume_to_state_E(vol_ens[['Date','Volume']], time_col='Date', vol_col='Volume',
                              aref_m2=AREF_M2)

# Early Warning Signals
print("\n" + "=" * 80)
print("COMPUTING EARLY WARNING SIGNALS")
print("=" * 80 + "\n")

E_oct_anom = monthly_anomalies(E_oct_df['E'])
ews_oct = rolling_ews_annual(E_oct_anom['E_anom'], window_years=11)

if not ews_oct.empty:
    print(f"✓ Computed EWS for {len(ews_oct)} rolling windows\n")

# Resilience Analysis with Bootstrap
print("=" * 80)
print("RESILIENCE ANALYSIS WITH BOOTSTRAP UNCERTAINTY")
print("=" * 80 + "\n")

epochs = [(1982, 2005), (2006, 2024)]
print(f"Analysis epochs: {epochs}\n")

print("Computing empirical potentials with bootstrap (this may take a few minutes)...")
results_oct = monthly_empirical_potentials_by_epoch(
    E_oct_df,
    epochs=epochs,
    use_sigma_from_ar1=True,
    min_points=8,
    compute_bootstrap=True,
    nboot=1000
)

# --- Export metrics safely
try:
    metrics_df = export_comprehensive_metrics(
        results_oct,
        filename='oct_resilience_metrics_comprehensive.csv'
    )
except Exception as e:
    print(f"⚠️ export_comprehensive_metrics failed: {e}")
    metrics_df = pd.DataFrame()  # ensure variable exists


print("\n=== Summary of October Resilience Metrics ===")

if metrics_df is None or metrics_df.empty:
    print("No metrics to display (empty DataFrame).")
else:
    # Columns you want to show (filter to the ones that actually exist)
    wanted = ['epoch', 'month', 'E_min', 'ΔU', "U''_min", 'ΔU/σ²',
              'ΔU_median', 'ΔU_ci68_lo', 'ΔU_ci68_hi',
              "U''_min_median", "U''_min_ci68_lo", "U''_min_ci68_hi",
              'ΔU/σ²_median', 'ΔU/σ²_ci68_lo', 'ΔU/σ²_ci68_hi']
    cols = [c for c in wanted if c in metrics_df.columns]
    # Show only October rows if multiple months exist
    to_show = metrics_df[metrics_df['month'] == 10] if 'month' in metrics_df.columns else metrics_df
    print(to_show[cols].to_string(index=False))

# --- Sigma consistency check (October)
sigma_consistency_df = export_sigma_consistency_table(
    results_oct,
    month=10,
    output_path='sigma_consistency_october.csv'
)


# # --- After: results_oct = monthly_empirical_potentials_by_epoch(...)

# def prob_metric_decrease(results_oct, month=10, early='1982-2005', late='2006-2024'):
#     """Return probabilities that early > late for ΔU/σ² and U″(Emin)."""
#     def _get_draws(ep):
#         boot = results_oct.get(ep, {}).get(month, {}).get('bootstrap', None)
#         if boot is None or 'draws' not in boot:
#             return None
#         return boot['draws']  # columns: [ΔU, U″(Emin), ΔU/σ²]

#     d_early = _get_draws(early)
#     d_late  = _get_draws(late)

#     # If draws are missing, recompute them quickly
#     if d_early is None:
#         ep_df = results_oct[early][month]['epoch_df']
#         d_early = bootstrap_metrics_month(ep_df, month, nboot=2000, block_size=3)
#     if d_late is None:
#         ep_df = results_oct[late][month]['epoch_df']
#         d_late  = bootstrap_metrics_month(ep_df, month, nboot=2000, block_size=3)

#     # Align lengths
#     n = min(len(d_early), len(d_late))
#     if n == 0:
#         return np.nan, np.nan

#     d_early = d_early[:n]
#     d_late  = d_late[:n]

#     # Columns: 0=ΔU, 1=U″(Emin), 2=ΔU/σ²
#     p_barrier = np.mean(d_early[:,2] > d_late[:,2])  # P(ΔU/σ²_early > ΔU/σ²_late)
#     p_curv    = np.mean(d_early[:,1] > d_late[:,1])  # P(U″_early > U″_late)
#     return p_barrier, p_curv

# p_barrier, p_curv = prob_metric_decrease(results_oct, month=10)

# print(f"Probability resilience decreased (ΔU/σ² early > late): {p_barrier:.3f}")
# print(f"Probability curvature decreased (U″(Emin) early > late): {p_curv:.3f}")

# # Optionally append to your comprehensive CSV
# with open('oct_resilience_metrics_comprehensive.csv', 'a') as f:
#     f.write(f"# Probabilities (October): P[ΔU/σ²_early>late]={p_barrier:.3f}, "
#             f"P[U''_early>late]={p_curv:.3f}\n")

# print("\n=== Summary of October Resilience Metrics ===")
# display_cols = ['epoch', 'month', 'E_min', 'ΔU', "U''_min", 'ΔU/σ²']
# if 'ΔU_median' in metrics_df.columns:
#     display_cols.extend(['ΔU_ci68_lo', 'ΔU_ci68_hi'])
# print(metrics_df[display_cols].to_string(index=False))

# Create comprehensive visualization

def barrier_presence_fraction(boot_draws, threshold=0.05):
    """
    Fraction of bootstrap realizations with a meaningful barrier.
    threshold : minimum normalized barrier ΔU/σ² to count as 'present'
    """
    if boot_draws is None or len(boot_draws) == 0:
        return np.nan
    dU_norm = boot_draws[:, 2]  # ΔU/σ²
    valid = np.isfinite(dU_norm) & (dU_norm > threshold)
    return np.mean(valid)

f_early = barrier_presence_fraction(results_oct['1982-2005'][10]['bootstrap']['draws'])
f_late  = barrier_presence_fraction(results_oct['2006-2024'][10]['bootstrap']['draws'])
print(f"Barrier present (bootstrap): early={f_early:.2f}, late={f_late:.2f}")



print("\nCreating comprehensive visualization...")
fig = plot_comprehensive_potential(results_oct, month=10)
if fig is not None:
    fig.savefig('oct_comprehensive_resilience_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: oct_comprehensive_resilience_analysis.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nKey Features:")
print("  ✓ Block bootstrap uncertainty quantification (68% & 95% CIs)")
print("  ✓ Sigma sensitivity analysis (innovation, std, residual)")
print("  ✓ Bandwidth sensitivity analysis (Scott, Silverman, ±20%)")
print("  ✓ Comprehensive 4-panel visualization")
print("  ✓ Complete metrics export with all CIs")
plt.show()