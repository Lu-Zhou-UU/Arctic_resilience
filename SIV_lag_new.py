import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# ---- Physical constants ----
RHO_I = 917.0                # kg m^-3, density of ice
L_I   = 3.34e5               # J kg^-1, latent heat of fusion
KAPPA = RHO_I * L_I          # J m^-3, latent energy density

# ---- Reference area ----
# 25 km × 25 km grid cell
AREF_KM2 = 25.0 * 25.0       # 625 km^2
AREF_M2  = AREF_KM2 * 1e6    # = 6.25e8 m^2
print(f"Reference area Aref = {AREF_M2:.2e} m²")


def load_ice_volume_data(filename, dataset_name):
    """
    Load ice volume data from txt file
    
    Parameters
    ----------
    filename : str
        Path to the txt file
    dataset_name : str
        Name of the dataset for display purposes
        
    Returns
    -------
    pd.DataFrame
        Ice volume data with Year and Volume columns
    """
    try:
        # Read the txt file
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Volume'])
        print(f"✓ Successfully loaded {dataset_name} ice volume data from {filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"  Volume range: {df['Volume'].min():.1f} - {df['Volume'].max():.1f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name} ice volume data: {e}")
        print(f"Tip: Check if the file {filename} exists and has the correct format (year volume)")
        return None


def harmonize_volume(c3_df, avhrr_df, on='Year'):
    """Return (c3, avhrr_bc, ensemble) dataframes with aligned index and bias-corrected AVHRR."""
    c3 = c3_df[[on, 'Volume']].rename(columns={'Volume':'Vol_C3S'}).dropna().copy()
    av = avhrr_df[[on, 'Volume']].rename(columns={'Volume':'Vol_AVHRR'}).dropna().copy()
    df = pd.merge(c3, av, on=on, how='inner')  # overlapping years only for fit

    # Linear bias correction: AVHRR_bc = a * AVHRR + b to match C3S
    if len(df) >= 5:
        A = np.vstack([df['Vol_AVHRR'].values, np.ones(len(df))]).T
        a, b = np.linalg.lstsq(A, df['Vol_C3S'].values, rcond=None)[0]
    else:
        a, b = 1.0, 0.0

    # Apply to full AVHRR
    av['Vol_AVHRR_bc'] = a * av['Vol_AVHRR'] + b

    # Union years; outer join and ensemble mean where both exist
    u = pd.merge(c3, av[[on,'Vol_AVHRR_bc']], on=on, how='outer').sort_values(on)
    u['Vol_Ensemble'] = u[['Vol_C3S','Vol_AVHRR_bc']].mean(axis=1)

    return c3, av, u, (a, b)


def volume_to_state_E(df, time_col, vol_col, aref_m2=AREF_M2, kappa=KAPPA):
    """
    Convert sea-ice volume (km³) to energy per unit area (J m⁻²)
    using E = -ρ_i L_i * (V / Aref)
    -----------------------------------------------------------
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [time_col, vol_col]
    time_col : str
        Datetime or year column
    vol_col : str
        Sea-ice volume in km³
    aref_m2 : float
        Reference area in m² (default = 25x25 km cell)
    kappa : float
        ρ_i * L_i (latent energy density, J m⁻³)

    Returns
    -------
    pd.DataFrame with Date index and column ['E'] (J m⁻²)
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.set_index(time_col).sort_index()

    # Convert km³ → m³, then to mean thickness (m)
    V_m3 = out[vol_col].values * 1e9
    h_m  = V_m3 / aref_m2

    # Energy per m² (negative for ice-covered state)
    E_Jm2 = -kappa * h_m
    out['E'] = E_Jm2

    print(f"Converted {len(out)} records: "
          f"E range {out['E'].min():.2e} to {out['E'].max():.2e} J m⁻²")
    return out[['E']]


def monthly_anomalies(E_series):
    """
    Remove monthly climatology from E(t) to obtain E'(t) for early-warning metrics.
    """
    df = E_series.to_frame('E')
    df['month'] = df.index.month
    clim = df.groupby('month')['E'].mean()
    df['E_anom'] = df.apply(lambda r: r['E'] - clim.loc[r['month']], axis=1)
    return df[['E','E_anom']]


def rolling_ews_annual(E_series, window_years=8):
    """
    Rolling variance and lag-1 AC on an annual series (e.g., October or April only).
    Returns a DataFrame with columns: ['year','var','ac1'].
    """
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


def process_month_data(c3_file, avhrr_file, month_num, month_name):
    """
    Process ice volume data for a specific month and return EWS dataframe
    
    Parameters
    ----------
    c3_file : str
        Path to C3S data file
    avhrr_file : str
        Path to AVHRR data file
    month_num : int
        Month number (4 for April, 10 for October)
    month_name : str
        Month name for display
    
    Returns
    -------
    pd.DataFrame
        EWS dataframe with AC(1) Fisher z-transform
    """
    print(f"\n{'='*60}")
    print(f"Processing {month_name.upper()} data")
    print(f"{'='*60}")
    
    # Load data
    c3_df = load_ice_volume_data(c3_file, f'C3S {month_name}')
    avhrr_df = load_ice_volume_data(avhrr_file, f'AVHRR {month_name}')
    
    # Harmonize volumes
    c3, av, vol_ens, (a, b) = harmonize_volume(c3_df, avhrr_df, on='Year')
    
    # Prepare for conversion
    vol_ens = vol_ens.rename(columns={'Vol_Ensemble': 'Volume'})
    
    # Build timestamp per year (use month_num)
    vol_ens['Date'] = pd.to_datetime(
        vol_ens['Year'].astype(int).astype(str) + f'-{month_num:02d}-01'
    )
    
    # Convert to model state E (J m^-2)
    E_df = volume_to_state_E(
        vol_ens[['Date', 'Volume']], 
        time_col='Date', 
        vol_col='Volume',
        aref_m2=AREF_M2
    )
    
    # Use anomalies for EWS
    E_anom = monthly_anomalies(E_df['E'])
    ews = rolling_ews_annual(E_anom['E_anom'], window_years=8)
    
    if ews is None or ews.empty:
        print(f"⚠️ No EWS data computed for {month_name}")
        return None
    
    # Standardize and transform
    ews_plot = ews.copy()
    
    # Guard against constant series
    if np.std(ews_plot['var'].values, ddof=1) > 0:
        ews_plot['var_z'] = (ews_plot['var'] - ews_plot['var'].mean()) / ews_plot['var'].std(ddof=1)
    else:
        ews_plot['var_z'] = np.zeros_like(ews_plot['var'].values)
    
    # Fisher z for AC1: only for |r|<1; clamp slightly inside [-1,1] to be safe
    r = ews_plot['ac1'].astype(float).clip(-0.999, 0.999)
    ews_plot['ac1_fisher'] = np.arctanh(r)
    
    # Kendall trend test
    try:
        from scipy.stats import kendalltau
        tau_ac, p_ac = kendalltau(ews_plot['year'].values, ews_plot['ac1'].values)
        print(f"{month_name} AC(1): Kendall τ={tau_ac:.2f}, p={p_ac:.3f}")
    except Exception:
        tau_ac, p_ac = (np.nan, np.nan)
    
    # Store Kendall results
    ews_plot['tau_ac'] = tau_ac
    ews_plot['p_ac'] = p_ac
    
    return ews_plot


# ---------------------------
# MAIN PROCESSING
# ---------------------------

# Process October data
ews_october = process_month_data(
    c3_file='/Volumes/Yotta_1/C3_ice_volume_October.txt',
    avhrr_file='/Volumes/Yotta_1/PIOMAS_ice_volume_October.txt',
    month_num=10,
    month_name='October'
)

# Process April data
ews_april = process_month_data(
    c3_file='/Volumes/Yotta_1/C3_ice_volume.txt',
    avhrr_file='/Volumes/Yotta_1/PIOMAS_ice_volume_April.txt',
    month_num=4,
    month_name='April'
)

# ---------------------------
# COMBINED PLOT: AC(1) Fisher z only
# ---------------------------

if ews_october is not None and ews_april is not None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # Plot October
    ax.plot(ews_october['year'], ews_october['ac1_fisher'], 
            marker='s', lw=2.0, alpha=0.85, 
            color='tab:orange', label='October',
            markersize=6)
    
    # Plot April
    ax.plot(ews_april['year'], ews_april['ac1_fisher'], 
            marker='o', lw=2.0, alpha=0.85, 
            color='tab:blue', label='April',
            markersize=6)
    
    # Formatting
    ax.set_ylabel('AC(1) (Fisher z-transform)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.95, fontsize=11)
    
    # Title with Kendall tau values
    tau_oct = ews_october['tau_ac'].iloc[0] if 'tau_ac' in ews_october.columns else np.nan
    p_oct = ews_october['p_ac'].iloc[0] if 'p_ac' in ews_october.columns else np.nan
    tau_apr = ews_april['tau_ac'].iloc[0] if 'tau_ac' in ews_april.columns else np.nan
    p_apr = ews_april['p_ac'].iloc[0] if 'p_ac' in ews_april.columns else np.nan
    
    title = f"Sea Ice Early Warning Signal: AC(1) Fisher z-transform (8-yr rolling window)\n"
    title += f"October: Kendall τ={tau_oct:.2f}, p={p_oct:.3f}  |  "
    title += f"April: Kendall τ={tau_apr:.2f}, p={p_apr:.3f}"
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    
    # Optional: Add zero reference line
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('combined_AC1_Fisher_October_April.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: combined_AC1_Fisher_October_April.png")
    plt.show()
    
else:
    print("\n❌ Could not create combined plot - missing data for one or both months")
    if ews_october is None:
        print("   Missing: October data")
    if ews_april is None:
        print("   Missing: April data")