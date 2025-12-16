import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---- Physical constants ----
RHO_I = 917.0            # kg m^-3, density of ice
L_I   = 3.34e5           # J kg^-1, latent heat of fusion
KAPPA = RHO_I * L_I      # J m^-3, latent energy density (rho_i * L_i)

# ---- Reference area (Using 10M km² for pan-Arctic mean thickness base) ----
AREF_KM2 = 25.0 * 25.0       # 625 km^2
AREF_M2  = AREF_KM2 * 1e6    # = 6.25e8 m^2
print(f"Reference Area Aref = {AREF_M2:.2e} m²")

# ==================== DATA LOADING AND STATE CONVERSION ====================

def load_monthly_siv(filepath, start_year=1982, end_year=2024):
    """
    Loads monthly SIV data (Jan 1982 - Dec 2024) from a text file.
    Assumption: File contains monthly volume data (km^3) in time-order.
    """
    try:
        # Assuming the file is monthly data, organized by month/year order
        df = pd.read_csv(filepath, sep=' ', header=None, names=['Volume'])
        
        # Create a date range from start to end year, monthly frequency
        months_total = (end_year - start_year + 1) * 12
        if len(df) < months_total:
            print(f"⚠️ Warning: File length ({len(df)}) less than expected ({months_total} months). Truncating date range.")
            dates = pd.date_range(start=f'{start_year}-01-01', periods=len(df), freq='MS')
        else:
            dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-01', freq='MS')
            
        df['Date'] = dates[:len(df)]
        df = df[['Date', 'Volume']].set_index('Date').sort_index()
        
        print(f"✓ Loaded {len(df)} monthly SIV records.")
        print(f"  Range: {df.index.min().date()} to {df.index.max().date()}")
        return df
    except Exception as e:
        print(f"❌ Error loading monthly SIV data: {e}")
        return None

def volume_to_state_E(df, vol_col='Volume', aref_m2=AREF_M2, kappa=KAPPA):
    """
    Convert SIV (km³) to model energy state E (J m⁻²) using E = -ρ_i L_i * h_i.
    """
    out = df.copy()
    # Convert km³ → m³
    V_m3 = out[vol_col].values * 1e9
    # Calculate mean thickness (m)
    h_m = V_m3 / aref_m2
    # Energy per m² (negative for ice-covered state)
    E_Jm2 = -kappa * h_m
    out['E'] = E_Jm2
    
    print(f"✓ Converted SIV to state E. Range: {out['E'].min():.2e} to {out['E'].max():.2e} J m⁻²")
    return out[['E']]

# --- Generic helper functions (re-used/simplified from original code) ---

def monthly_anomalies(E_series):
    """ Remove monthly climatology from E(t). """
    df = E_series.to_frame('E')
    df['month'] = df.index.month
    clim = df.groupby('month')['E'].mean()
    df['E_anom'] = df.apply(lambda r: r['E'] - clim.loc[r['month']], axis=1)
    return df[['E', 'E_anom']]

def ar1_fit(x):
    """ AR(1) phi and noise variance sig2. """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 3: return np.nan, np.nan
    x0, x1 = x[:-1], x[1:]
    phi = np.dot(x0, x1) / np.dot(x0, x0) if np.dot(x0,x0) != 0 else np.nan
    eps = x1 - phi * x0
    sig2 = np.var(eps, ddof=1) if len(eps) > 1 else np.nan
    return phi, sig2

def monthly_sigma_lambda(E_anom_df):
    """ Per-month noise amplitude sigma_m and recovery rate lambda_m. """
    res = []
    for m in range(1,13):
        xm = E_anom_df.loc[E_anom_df.index.month==m, 'E_anom'].values
        phi, sig2 = ar1_fit(xm)
        sigma = np.sqrt(sig2) if np.isfinite(sig2) else np.nan
        res.append({'month': m, 'sigma': sigma})
    return pd.DataFrame(res)

# --- Full Monthly EWS (Rolling) ---

def rolling_ews_monthly(E_anom_df, window_years=10, min_points=8):
    """
    Compute rolling variance and lag-1 AC by month using a trailing window.
    """
    df = E_anom_df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    out = []
    anom_col = 'E_anom' # Column name for anomaly (detrended E)
    win = window_years * 12
    
    for end in range(win, len(df)):
        sub = df.iloc[end-win:end]
        end_time = sub.index[-1]
        
        for m in range(1, 13):
            xm = sub.loc[sub['month'] == m, anom_col].values
            xm_clean = xm[~np.isnan(xm)]
            
            if len(xm_clean) >= min_points:
                var = np.var(xm_clean, ddof=1)
                ac1 = np.corrcoef(xm_clean[:-1], xm_clean[1:])[0, 1] if len(xm_clean) >= 3 and np.std(xm_clean) > 1e-8 else np.nan
                
                out.append({
                    'time': end_time, 'month': m, 'var': var, 'ac1': ac1, 'n_points': len(xm_clean)
                })
    
    out_df = pd.DataFrame(out).set_index('time').sort_index()
    print(f"✓ Computed Monthly EWS ({window_years}-yr rolling): {len(out_df)} points.")
    return out_df

# --- Potential Analysis Functions --- (Assuming these are from your original code)

def kde_empirical_potential(E_values, sigma=1.0, grid=None, bw='scott'):
    """ Build U_emp(E) = -(sigma^2/2) * ln P(E) via KDE. """
    # ... (Implementation remains the same) ...
    x = np.asarray(E_values); x = x[~np.isnan(x)]
    if len(x) < 2: return None, None, None
    if grid is None: lo, hi = np.percentile(x, [1, 99]); grid = np.linspace(lo, hi, 400)
    kde = stats.gaussian_kde(x, bw_method=bw)
    p = np.maximum(kde(grid), 1e-300)
    U = -(sigma**2 / 2.0) * np.log(p)
    U -= np.min(U)
    return grid, U, kde

def extrema_from_potential(grid, U):
    """ Find minima (stable) and maxima (saddles). """
    # ... (Implementation remains the same) ...
    if grid is None or U is None: return []
    sp = UnivariateSpline(grid, U, k=3, s=0)
    d1 = sp.derivative(n=1)(grid); d2 = sp.derivative(n=2)(grid)
    zc_idx = np.where(np.diff(np.sign(d1)) != 0)[0]; extrema = []
    for i in zc_idx:
        e = grid[i]; u2 = d2[i]
        kind = 'min' if u2 > 0 else 'max'
        extrema.append({'E': e, 'Upp': u2, 'type': kind})
    return sorted(extrema, key=lambda r: r['E'])

def epoch_slices(df, epochs):
    """ Returns dict {epoch_label: df_slice} """
    # ... (Implementation remains the same) ...
    out = {}
    for (y0,y1) in epochs:
        lab = f"{y0}-{y1}"
        sl = df[(df.index.year>=y0) & (df.index.year<=y1)]
        if not sl.empty:
            out[lab] = sl.copy()
    return out

def monthly_empirical_potentials_by_epoch(E_df, epochs, use_sigma_from_ar1=True, min_points=8):
    """ Compute U_emp(E) and metrics for each month/epoch. """
    results = {}
    for (y0, y1) in epochs:
        lab = f"{y0}-{y1}"
        sub = E_df[(E_df.index.year >= y0) & (E_df.index.year <= y1)]
        if sub.empty: continue
        
        anom = monthly_anomalies(sub['E']) 
        siglam = monthly_sigma_lambda(anom)
        results[lab] = {}
        
        for m in range(1, 13):
            Em = sub.loc[sub.index.month == m, 'E'].dropna().values
            if len(Em) < min_points: continue
            
            sigma_m = float(siglam.loc[siglam['month'] == m, 'sigma']) if use_sigma_from_ar1 else 1.0
            if not np.isfinite(sigma_m) or sigma_m <= 0: sigma_m = 1.0

            grid, U, _ = kde_empirical_potential(Em, sigma=sigma_m)
            extrema = extrema_from_potential(grid, U)
            if not extrema: continue

            U_spline = UnivariateSpline(grid, U, k=3, s=0)
            mins = [e for e in extrema if e['type'] == 'min']
            if not mins: continue
            
            e_min_ice = sorted(mins, key=lambda z: z['E'])[0] # Lowest E state is ice-covered
            E_s1 = e_min_ice['E']; U_s1 = float(U_spline(E_s1)); Upp_s1 = float(e_min_ice['Upp'])

            maxs = [e for e in extrema if e['type'] == 'max']
            saddles_r = [e for e in maxs if e['E'] > E_s1]
            dU = np.nan
            if saddles_r:
                e_sad = sorted(saddles_r, key=lambda z: z['E'])[0]
                E_s2 = e_sad['E']; U_s2 = float(U_spline(E_s2))
                dU = U_s2 - U_s1

            results[lab][m] = dict(grid=grid, U=U, E_s1=E_s1, Upp_s1=Upp_s1, dU=dU, sigma_m=sigma_m)
    return results

# --- Plotting Functions --- (These are fine and plot the correct variables)

def plot_ews_heatmaps(ews_df, title_prefix='SIV Proxy', vmin_var=None, vmax_var=None, vmin_ac1=-1, vmax_ac1=1):
    """ Plots heatmaps for Variance and AC1. """
    # ... (Function body from original code) ...
    if ews_df.empty:
        print("❌ No EWS data to plot.")
        return None, None
    V = ews_df.pivot_table(index='month', columns=ews_df.index, values='var')
    A = ews_df.pivot_table(index='month', columns=ews_df.index, values='ac1')
    if V.dropna(how='all').empty or A.dropna(how='all').empty:
        print("❌ All variance or AC1 values are NaN. Cannot plot.")
        return None, None
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    valid_cols_V = V.columns[V.notna().any()]
    extent_V = [mdates.date2num(valid_cols_V.min()), mdates.date2num(valid_cols_V.max()), 0.5, 12.5]
    im1 = axs[0].imshow(V.values, aspect='auto', origin='lower', extent=extent_V, cmap='YlOrRd', vmin=vmin_var, vmax=vmax_var, interpolation='nearest')
    axs[0].set_ylabel('Month'); axs[0].set_title(f'{title_prefix} Rolling Variance', fontweight='bold')
    axs[0].set_yticks(range(1, 13)); plt.colorbar(im1, ax=axs[0], label='Variance')

    valid_cols_A = A.columns[A.notna().any()]
    extent_A = [mdates.date2num(valid_cols_A.min()), mdates.date2num(valid_cols_A.max()), 0.5, 12.5]
    im2 = axs[1].imshow(A.values, aspect='auto', origin='lower', extent=extent_A, cmap='coolwarm', vmin=vmin_ac1, vmax=vmax_ac1, interpolation='nearest')
    axs[1].set_ylabel('Month'); axs[1].set_xlabel('Year'); axs[1].set_title(f'{title_prefix} Lag-1 AC', fontweight='bold')
    axs[1].set_yticks(range(1, 13)); plt.colorbar(im2, ax=axs[1], label='AC(1)')

    axs[1].xaxis_date(); axs[1].xaxis.set_major_locator(mdates.YearLocator(base=5)); axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha='right'); plt.tight_layout()
    return fig, axs

def plot_empirical_potentials(results, months=(3, 6, 9)):
    """ Plots small multiples of the empirical potential U_emp. """
    # ... (Function body from original code) ...
    if not isinstance(results, dict) or len(results) == 0: return None, None
    epoch_labels = list(results.keys()); nrow = len(epoch_labels); ncol = len(months); month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), squeeze=False)
    for i, lab in enumerate(epoch_labels):
        for j, m in enumerate(months):
            ax = axes[i, j]; r = results.get(lab, {}).get(m, None)
            if r is None: ax.set_title(f"{lab} {month_names[m-1]} (no data)"); ax.axis('off'); continue
            ax.plot(r['grid'], r['U'], lw=2)
            if np.isfinite(r.get('E_s1', np.nan)): ax.axvline(r['E_s1'], ls='--', alpha=0.6)
            if np.isfinite(r.get('E_s2', np.nan)): ax.axvline(r['E_s2'], ls='--', color='tab:red', alpha=0.6)
            
            ax.set_title(f"{lab} {month_names[m-1]}")
            ax.set_xlabel(r'$E$ (J m$^{-2}$)')
            ax.set_ylabel(r'$U_m^{\rm emp}(E)$ (scaled)')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig, axes

def plot_curvature_barrier_summary(results):
    """ Plots seasonal trends in Curvature (U'') and normalized Barrier (ΔU/σ^2) by epoch. """
    # ... (Function body from original code) ...
    epoch_labels = list(results.keys()); months = range(1,13); rows = []
    for lab in epoch_labels:
        for m in months:
            if m in results[lab]:
                r = results[lab][m]; rows.append({'epoch': lab, 'month': m, 'Upp': r['Upp_s1'], 'dU': r['dU'], 'sigma': r['sigma_m']})
    df = pd.DataFrame(rows)
    if df.empty: print("No curvature/barrier data."); return None, None

    # Curvature (Recovery Rate Proxy)
    fig1, ax1 = plt.subplots(1,1, figsize=(12,4))
    for lab in epoch_labels:
        sub = df[df['epoch']==lab]
        ax1.plot(sub['month'], sub['Upp'], marker='o', label=lab)
    ax1.set_xlabel('Month'); ax1.set_ylabel(r'Curvature $U^{\prime\prime}$ (ice-covered min)')
    ax1.set_xticks(range(1, 13)); ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_title('Monthly Curvature by Epoch (Recovery Rate)')

    # Normalized Barrier (Resistance Proxy)
    fig2, ax2 = plt.subplots(1,1, figsize=(12,4))
    for lab in epoch_labels:
        sub = df[df['epoch']==lab].copy()
        sub['dU_norm'] = sub['dU'] / (sub['sigma']**2) if 'sigma' in sub.columns else sub['dU']
        ax2.plot(sub['month'], sub['dU_norm'], marker='s', label=lab)
    ax2.set_xlabel('Month'); ax2.set_ylabel(r'Normalized Barrier $\Delta U / \sigma^2$')
    ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_title('Monthly Barrier Height by Epoch (Resistance)')

    plt.tight_layout()
    return fig1, fig2

# --- Helpers to do Sep-style EWS for any calendar month ----------------------
from scipy.stats import kendalltau

MONTH_NAME = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def _subset_month_E(E_df, month):
    """Return a monthly series E_m with one value per year for the given month."""
    s = E_df[E_df.index.month == month]['E'].copy()
    s.index = pd.to_datetime(s.index)  # ensure datetime index
    return s.sort_index()

def rolling_ews_annual(E_series, window_years=10):
    """
    Rolling variance and lag-1 AC on an annual series (e.g., a single month per year).
    Returns DataFrame with: ['year','var','ac1'].
    """
    df = E_series.to_frame('E').copy()
    df['year'] = df.index.year
    rows = []
    for end in range(window_years-1, len(df)):
        sub = df.iloc[end-window_years+1:end+1]['E'].values
        yr  = int(df.iloc[end]['year'])
        if len(sub) >= 5:
            var = np.var(sub, ddof=1)
            ac1 = np.corrcoef(sub[:-1], sub[1:])[0,1] if len(sub) > 2 else np.nan
            rows.append({'year': yr, 'var': var, 'ac1': ac1})
    return pd.DataFrame(rows)

def plot_month_ews(E_df, month, window_years=10, events=(2007, 2012, 2020),
                   savepath=None):
    """
    Make the Sep-style two-panel figure for a given calendar month.
    - Removes that month's climatology (via monthly_anomalies) before EWS
    - Shows Kendall τ and p for variance and AC(1)
    """
    mname = MONTH_NAME[month-1]
    # 1) subset to one value per year for this month
    Em = _subset_month_E(E_df, month)
    if Em.empty:
        print(f"❌ No data for month {mname}.")
        return None

    # 2) anomalies (this function handles a single-month series fine)
    Em_anom = monthly_anomalies(Em)['E_anom']

    # 3) rolling EWS on the annual (per-year) series
    ews = rolling_ews_annual(Em_anom, window_years=window_years)
    if ews.empty:
        print(f"❌ Not enough years for rolling EWS in {mname}.")
        return None

    # 4) standardizations and Kendall tests
    ews_plot = ews.copy()
    ews_plot['var_z'] = (ews_plot['var'] - ews_plot['var'].mean()) / (ews_plot['var'].std(ddof=1) or 1)
    r = ews_plot['ac1'].astype(float).clip(-0.999, 0.999)
    ews_plot['ac1_fisher'] = np.arctanh(r)

    try:
        tau_var, p_var = kendalltau(ews_plot['year'].values, ews_plot['var'].values)
        tau_ac,  p_ac  = kendalltau(ews_plot['year'].values, ews_plot['ac1'].values)
    except Exception:
        tau_var = p_var = tau_ac = p_ac = np.nan

    # 5) plot (same style as your September figure)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # Variance
    ax[0].plot(ews_plot['year'], ews_plot['var'], marker='o', lw=1.8, label='Variance (raw)')
    ax[0].plot(ews_plot['year'], ews_plot['var_z'], marker='s', lw=1.4, alpha=0.85, label='Variance (z-score)')
    ax[0].set_ylabel(f"Var[ E'({mname}) ]")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc='best', framealpha=0.9)
    ax[0].set_title(f"{mname} early-warning signals ({window_years}-yr rolling)  —  Var: Kendall τ={tau_var:.2f}, p={p_var:.3f}")

    # AC(1)
    ax[1].plot(ews_plot['year'], ews_plot['ac1'], marker='o', lw=1.8, label='AC(1) (raw)')
    ax[1].plot(ews_plot['year'], ews_plot['ac1_fisher'], marker='s', lw=1.4, alpha=0.85, label='AC(1) (Fisher z)')
    ax[1].set_ylabel("AC(1)")
    ax[1].set_xlabel('Year')
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc='best', framealpha=0.9)
    ax[1].set_title(f"AC(1): Kendall τ={tau_ac:.2f}, p={p_ac:.3f}")

    # Optional event markers (e.g., 2007/2012/2020)
    y0 = ews_plot['year'].min(); y1 = ews_plot['year'].max()
    for yr in events or ():
        if y0 <= yr <= y1:
            for a in ax:
                a.axvline(yr, ls='--', lw=1.2, color='gray', alpha=0.6)
                a.text(yr, a.get_ylim()[1]*0.97, str(yr), ha='center', va='top',
                       fontsize=9, color='dimgray')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {savepath}")
    return fig



# ==================== MAIN EXECUTION FOR MONTHLY SIV (E) ====================

if __name__ == "__main__":
    print("=" * 70)
    print("SIV-BASED SEASONAL RESILIENCE ANALYSIS (The Definitive E)")
    print("=" * 70)
    
    # 1. Load the full monthly SIV data
    print("\n[1/5] Loading Monthly SIV data...")
    # NOTE: You need to replace 'your_monthly_siv_file.txt' with the actual path
    # Example: SIV data starting Jan 1982
    siv_monthly = load_monthly_siv('SIV_monthly_198201_202412.txt', start_year=1982, end_year=2024)
    if siv_monthly is None or siv_monthly.empty: exit()

    # 2. Convert to Model State E (Physically Correct)
    print("\n[2/5] Converting SIV to state variable E (J m⁻²)...")
    E_df = volume_to_state_E(siv_monthly) 
    
    # 3. Define Epochs
    years = E_df.index.year.unique()
    ymin, ymax = int(years.min()), int(years.max())
    bounds = np.linspace(ymin, ymax + 1, 4, dtype=int) 
    # epochs = [(int(bounds[i]), int(bounds[i+1] - 1)) for i in range(3)]
    epochs = [(1982, 2005), (2006, 2024)]
    print(f"  Analysis Epochs: {epochs}")

    # 4. Compute Monthly EWS
    print("\n[3/5] Computing Monthly EWS (Rolling Variance/AC1)...")
    E_anom = monthly_anomalies(E_df['E']) # Deseasonalized residuals
    
    # NOTE: Use min_points=8 (8 years in 10-year window)
    ews = rolling_ews_monthly(E_anom, window_years=10, min_points=8)
    
    # 5. Compute Monthly Empirical Potentials and Metrics
    print("\n[4/5] Computing Epochal Potentials and Metrics (Curvature/Barrier)...")
    # NOTE: Use min_points=8 (8 months/points per month over ~14 years per epoch)
    results_monthly = monthly_empirical_potentials_by_epoch(
        E_df,
        epochs=epochs,
        use_sigma_from_ar1=True,
        min_points=8 
    )

    # 6. Generate Figures
    print("\n[5/5] Generating Visualizations...")

    # Figure 1: EWS Heatmaps (Seasonal Volatility)
    fig1, axs1 = plot_ews_heatmaps(ews, title_prefix='SIV (E) Monthly')
    if fig1: fig1.savefig('SIV_EWS_heatmaps.png', dpi=300, bbox_inches='tight'); print("  ✓ Saved: SIV_EWS_heatmaps.png")
    
    # Figure 2 & 3: Curvature and Barrier Summary (Seasonal Resilience Loss)
    fig3, fig4 = plot_curvature_barrier_summary(results_monthly)
    if fig3: fig3.savefig('SIV_Curvature_by_epoch.png', dpi=300, bbox_inches='tight'); print("  ✓ Saved: SIV_Curvature_by_epoch.png")
    if fig4: fig4.savefig('SIV_Barrier_norm_by_epoch.png', dpi=300, bbox_inches='tight'); print("  ✓ Saved: SIV_Barrier_norm_by_epoch.png")
    
    # Figure 4: Potential Landscape Snapshots (Visual proof of flattening)
    fig5, axes5 = plot_empirical_potentials(results_monthly, months=(3, 6, 9))
    if fig5: fig5.savefig('SIV_Potential_Snapshots_M369.png', dpi=300, bbox_inches='tight'); print("  ✓ Saved: SIV_Potential_Snapshots_M369.png")

    print("\n" + "=" * 70)
    print("SIV ANALYSIS SETUP COMPLETE. RUN CODE WITH YOUR DATA FILE.")
    print("=" * 70)
    plt.show()

    # After E_df is created in Script B
    fig_jun = plot_month_ews(E_df, month=6,  window_years=10, savepath='EWS_June_variance_ac1.png')
    fig_oct = plot_month_ews(E_df, month=10, window_years=10, savepath='EWS_October_variance_ac1.png')



