# ============================================
# COMBINED TWO-PANEL PLOT: SIE and SIV Early Warning Signals
# ALIGNED VERSION - Both panels use year-based x-axis
# ============================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import kendalltau

# ---- Physical constants for SIV ----
RHO_I = 917.0                # kg m^-3, density of ice
L_I   = 3.34e5               # J kg^-1, latent heat of fusion
KAPPA = RHO_I * L_I          # J m^-3, latent energy density

# ---- Reference area ----
AREF_KM2 = 25.0 * 25.0       # 625 km^2
AREF_M2  = AREF_KM2 * 1e6    # = 6.25e8 m^2

# ==========================================
# SECTION 1: SEA ICE EXTENT (SIE) FUNCTIONS
# ==========================================


# --- NEW: yearly rolling EWS for SIE (per target month) ---
def sie_monthly_to_yearly_ews(X_df, month, window_years=10):
    """
    X_df: DataFrame indexed by Date with column 'X' (anomaly or detrended state).
    month: 4 (April) or 10 (October)
    Returns: DataFrame with columns ['year','var','ac1','ac1_fisher']
    """
    # 1) pick one value per year (the chosen month)
    s = X_df.loc[X_df.index.month == month, 'X'].copy()
    # ensure one sample per year
    s.index = pd.to_datetime(s.index.year.astype(str))  # normalize to Jan-01 of each year
    s = s[~s.index.duplicated(keep='last')].sort_index()

    years = s.index.year.values
    vals  = s.values

    rows = []
    for i in range(window_years-1, len(vals)):
        sub = vals[i-window_years+1:i+1]
        y   = int(years[i])
        if np.isfinite(sub).sum() >= 5:
            var = np.var(sub, ddof=1)
            ac1 = np.corrcoef(sub[:-1], sub[1:])[0,1] if len(sub) > 2 else np.nan
            rows.append({'year': y, 'var': var, 'ac1': ac1})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Fisher z for plotting
    r = out['ac1'].astype(float).clip(-0.999, 0.999)
    out['ac1_fisher'] = np.arctanh(r)

    # Kendall trend on the raw AC(1) (yearly)
    try:
        tau, p = kendalltau(out['year'].values, out['ac1'].values)
    except Exception:
        tau, p = (np.nan, np.nan)
    out['tau_ac'] = tau
    out['p_ac']   = p
    return out



def load_monthly_extent(filepath, start_date="1979-01-01"):
    df = pd.read_csv(filepath, header=None, names=['Extent_km2'])
    dates = pd.date_range(start=pd.Timestamp(start_date), periods=len(df), freq='MS')
    df['Date'] = dates
    # NOTE: This file has a known calendar discontinuity around 1986-04 where
    # the monthly phase shifts abruptly. We correct by shifting timestamps
    # forward by 4 months from the breakpoint onward before month extraction.
    # This preserves the physical seasonality used in April/October diagnostics.
    breakpoint = pd.Timestamp("1986-04-01")
    mask = df['Date'] >= breakpoint
    if mask.any():
        df.loc[mask, 'Date'] = df.loc[mask, 'Date'] + pd.DateOffset(months=4)
        # Keep one value per timestamp after correction.
        df = df.sort_values('Date').drop_duplicates(subset='Date', keep='first')
    df['Extent'] = df['Extent_km2'] / 1e6  # million km²
    out = df[['Date', 'Extent']].sort_values('Date').reset_index(drop=True)
    print(f"✓ Loaded {len(out)} monthly SIE records "
          f"from {out['Date'].min().date()} to {out['Date'].max().date()}")
    return out

def extent_to_state(sie_df):
    """Map SIE to state variable X = -Extent (so more ice → more negative)."""
    out = sie_df.copy()
    out['X'] = -out['Extent'].astype(float)
    out = out.set_index('Date').sort_index()
    return out[['X']]

def detrend_within_window(x):
    """Linear detrend within rolling window."""
    if len(x) < 3:
        return x - np.mean(x)
    t = np.arange(len(x))
    slope, intercept, *_ = stats.linregress(t, x)
    return x - (intercept + slope * t)

def rolling_ews_detrended_sie(E_df, window_years=10, min_points=8, normalize='demean'):
    """Compute rolling variance and lag-1 AC by month after detrending for SIE."""
    df = E_df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    win = window_years * 12
    out = []

    for end in range(win, len(df)):
        sub = df.iloc[end - win:end]
        end_time = sub.index[-1]
        # end_year = end_time.year  # Extract year for consistent x-axis
        end_year = end_time.year + (end_time.month - 0.5) / 12

        for m in (4, 10):  # only April & October
            xm = sub.loc[sub['month'] == m, 'X'].dropna().values
            if len(xm) < min_points:
                continue

            x_d = detrend_within_window(xm)
            if normalize == 'demean':
                x_w = x_d - np.mean(x_d)
            elif normalize == 'zscore':
                s = np.std(x_d, ddof=1)
                x_w = (x_d - np.mean(x_d)) / s if s > 1e-8 else x_d
            else:
                x_w = x_d

            var = np.var(x_w, ddof=1)
            ac1 = np.corrcoef(x_w[:-1], x_w[1:])[0, 1] if len(x_w) > 2 else np.nan

            # Store year instead of datetime for consistent x-axis
            out.append({'year': end_year, 'month': m, 'var': var, 'ac1': ac1})

    out_df = pd.DataFrame(out)
    if not out_df.empty:
        print(f"✓ Computed detrended SIE EWS for {len(out_df)} points.")
    else:
        print("⚠️ No SIE EWS data computed.")
    return out_df

# ==========================================
# SECTION 2: SEA ICE VOLUME (SIV) FUNCTIONS
# ==========================================

def load_ice_volume_data(filename, dataset_name):
    """Load ice volume data from txt file"""
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Volume'])
        print(f"✓ Successfully loaded {dataset_name} ice volume data from {filename}")
        return df
    except Exception as e:
        print(f"❌ Error loading {dataset_name} ice volume data: {e}")
        return None

def harmonize_volume(c3_df, avhrr_df, on='Year'):
    """Return ensemble dataframes with bias-corrected AVHRR."""
    c3 = c3_df[[on, 'Volume']].rename(columns={'Volume':'Vol_C3S'}).dropna().copy()
    av = avhrr_df[[on, 'Volume']].rename(columns={'Volume':'Vol_AVHRR'}).dropna().copy()
    df = pd.merge(c3, av, on=on, how='inner')

    if len(df) >= 5:
        A = np.vstack([df['Vol_AVHRR'].values, np.ones(len(df))]).T
        a, b = np.linalg.lstsq(A, df['Vol_C3S'].values, rcond=None)[0]
    else:
        a, b = 1.0, 0.0

    av['Vol_AVHRR_bc'] = a * av['Vol_AVHRR'] + b
    u = pd.merge(c3, av[[on,'Vol_AVHRR_bc']], on=on, how='outer').sort_values(on)
    u['Vol_Ensemble'] = u[['Vol_C3S','Vol_AVHRR_bc']].mean(axis=1)

    return c3, av, u, (a, b)

def volume_to_state_E(df, time_col, vol_col, aref_m2=AREF_M2, kappa=KAPPA):
    """Convert sea-ice volume (km³) to energy per unit area (J m⁻²)"""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.set_index(time_col).sort_index()

    V_m3 = out[vol_col].values * 1e9
    h_m  = V_m3 / aref_m2
    E_Jm2 = -kappa * h_m
    out['E'] = E_Jm2

    print(f"Converted {len(out)} records: E range {out['E'].min():.2e} to {out['E'].max():.2e} J m⁻²")
    return out[['E']]

def monthly_anomalies(E_series):
    """Remove monthly climatology from E(t)"""
    df = E_series.to_frame('E')
    df['month'] = df.index.month
    clim = df.groupby('month')['E'].mean()
    df['E_anom'] = df.apply(lambda r: r['E'] - clim.loc[r['month']], axis=1)
    return df[['E','E_anom']]

def rolling_ews_annual(E_series, window_years=8):
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

def process_siv_month_data(c3_file, avhrr_file, month_num, month_name):
    """Process ice volume data for a specific month"""
    print(f"\n{'='*60}")
    print(f"Processing SIV {month_name.upper()} data")
    print(f"{'='*60}")
    
    c3_df = load_ice_volume_data(c3_file, f'C3S {month_name}')
    avhrr_df = load_ice_volume_data(avhrr_file, f'AVHRR {month_name}')
    
    c3, av, vol_ens, (a, b) = harmonize_volume(c3_df, avhrr_df, on='Year')
    vol_ens = vol_ens.rename(columns={'Vol_Ensemble': 'Volume'})
    vol_ens['Date'] = pd.to_datetime(
        vol_ens['Year'].astype(int).astype(str) + f'-{month_num:02d}-01'
    )
    
    E_df = volume_to_state_E(
        vol_ens[['Date', 'Volume']], 
        time_col='Date', 
        vol_col='Volume',
        aref_m2=AREF_M2
    )
    
    E_anom = monthly_anomalies(E_df['E'])
    ews = rolling_ews_annual(E_anom['E_anom'], window_years=11)
    
    if ews is None or ews.empty:
        print(f"⚠️ No SIV EWS data computed for {month_name}")
        return None
    
    ews_plot = ews.copy()
    
    if np.std(ews_plot['var'].values, ddof=1) > 0:
        ews_plot['var_z'] = (ews_plot['var'] - ews_plot['var'].mean()) / ews_plot['var'].std(ddof=1)
    else:
        ews_plot['var_z'] = np.zeros_like(ews_plot['var'].values)
    
    r = ews_plot['ac1'].astype(float).clip(-0.999, 0.999)
    ews_plot['ac1_fisher'] = np.arctanh(r)
    
    try:
        tau_ac, p_ac = kendalltau(ews_plot['year'].values, ews_plot['ac1'].values)
        print(f"SIV {month_name} AC(1): Kendall τ={tau_ac:.2f}, p={p_ac:.3f}")
    except Exception:
        tau_ac, p_ac = (np.nan, np.nan)
    
    ews_plot['tau_ac'] = tau_ac
    ews_plot['p_ac'] = p_ac
    
    return ews_plot

# ==========================================
# MAIN PROCESSING
# ==========================================

print("=" * 70)
print("COMBINED SEA ICE EXTENT & VOLUME EARLY WARNING ANALYSIS")
print("=" * 70)

# ========== PROCESS SIE DATA ==========
print("\n" + "=" * 70)
print("SECTION 1: SEA ICE EXTENT (SIE)")
print("=" * 70)

sie = load_monthly_extent("SIE_monthly_197901_202506.txt", start_date="1979-01-01")
X_df = extent_to_state(sie)
X_df['X_anom'] = X_df['X'] - X_df['X'].rolling(12*10, center=True, min_periods=6).mean()

# X_for_ews = X_df[['X_anom']].copy()
# X_for_ews.columns = ['X']

# ews_sie = rolling_ews_detrended_sie(X_for_ews, window_years=10, min_points=8, normalize='demean')

# # Calculate Fisher z for SIE
# sie_april = ews_sie[ews_sie['month'] == 4].copy()
# sie_october = ews_sie[ews_sie['month'] == 10].copy()

# Build an anomaly/detrended state first (you already do this)
X_for_ews = X_df[['X_anom']].rename(columns={'X_anom':'X'})

sie_april   = sie_monthly_to_yearly_ews(X_for_ews, month=4,  window_years=10)
sie_october = sie_monthly_to_yearly_ews(X_for_ews, month=10, window_years=10)

tau_sie_apr, p_sie_apr = sie_april['tau_ac'].iloc[0], sie_april['p_ac'].iloc[0] if not sie_april.empty else (np.nan, np.nan)
tau_sie_oct, p_sie_oct = sie_october['tau_ac'].iloc[0], sie_october['p_ac'].iloc[0] if not sie_october.empty else (np.nan, np.nan)

# if not sie_april.empty:
#     sie_april['ac1_fisher'] = np.arctanh(sie_april['ac1'].clip(-0.999, 0.999))
#     tau_sie_apr, p_sie_apr = kendalltau(sie_april['year'].values, sie_april['ac1'].values)
#     print(f"SIE April AC(1): Kendall τ={tau_sie_apr:.2f}, p={p_sie_apr:.3f}")
# else:
#     tau_sie_apr, p_sie_apr = np.nan, np.nan

# if not sie_october.empty:
#     sie_october['ac1_fisher'] = np.arctanh(sie_october['ac1'].clip(-0.999, 0.999))
#     tau_sie_oct, p_sie_oct = kendalltau(sie_october['year'].values, sie_october['ac1'].values)
#     print(f"SIE October AC(1): Kendall τ={tau_sie_oct:.2f}, p={p_sie_oct:.3f}")
# else:
#     tau_sie_oct, p_sie_oct = np.nan, np.nan

# ========== PROCESS SIV DATA ==========
print("\n" + "=" * 70)
print("SECTION 2: SEA ICE VOLUME (SIV)")
print("=" * 70)

ews_siv_october = process_siv_month_data(
    c3_file='/Volumes/Yotta_1/C3_ice_volume_October.txt',
    avhrr_file='/Volumes/Yotta_1/PIOMAS_ice_volume_October.txt',
    month_num=10,
    month_name='October'
)

ews_siv_april = process_siv_month_data(
    c3_file='/Volumes/Yotta_1/C3_ice_volume.txt',
    avhrr_file='/Volumes/Yotta_1/PIOMAS_ice_volume_April.txt',
    month_num=4,
    month_name='April'
)

# ==========================================
# TWO-PANEL COMBINED PLOT - ALIGNED VERSION
# ==========================================
record_years = [2007, 2012, 2020]

if (not sie_april.empty and not sie_october.empty and 
    ews_siv_april is not None and ews_siv_october is not None):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # ========== PANEL 1: SEA ICE EXTENT AC(1) ==========
    # Now SIE data has 'year' column directly, matching SIV format
    
    ax1.plot(sie_october['year'], sie_october['ac1_fisher'], 
             marker='s', lw=2.5, alpha=0.85, 
             color='#D4AF37', label='October',
             markersize=7)
    
    ax1.plot(sie_april['year'], sie_april['ac1_fisher'], 
             marker='o', lw=2.5, alpha=0.85, 
             color='#6A4C93', label='April',
             markersize=7)
    
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.set_ylabel('Normalized lag-1 autocorrelation', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    # ax1.legend(loc='lower left', framealpha=0.95, fontsize=11)
    
    # Add vertical lines for record years
    for year in record_years:
        ax1.axvline(x=year, color='darkslategray', linestyle='--', 
                   alpha=0.6, linewidth=2, zorder=1)
        # Add year label at top of panel
        ax1.text(year, ax1.get_ylim()[1]*0.97, str(year),
                rotation=0, fontsize=11, ha='center', va='top',
                fontweight='bold', color='darkslategray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Title with Kendall statistics
    title1 = f"(a) Sea Ice Extent\n"
    # title1 += f"October: τ={tau_sie_oct:.2f}, p={p_sie_oct:.3f}  |  April: τ={tau_sie_apr:.2f}, p={p_sie_apr:.3f}"
    # ax1.set_title(title1, fontsize=12, fontweight='bold', loc='left', pad=10)
    
    ax1.set_xlim(1988, 2026)

    # ========== PANEL 2: SEA ICE VOLUME AC(1) ==========
    ax2.plot(ews_siv_october['year'], ews_siv_october['ac1_fisher'], 
             marker='s', lw=2.5, alpha=0.85, 
             color='#D4AF37', label='October',
             markersize=7)
    
    ax2.plot(ews_siv_april['year'], ews_siv_april['ac1_fisher'], 
             marker='o', lw=2.5, alpha=0.85, 
             color='#6A4C93', label='April',
             markersize=7)
    
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_ylabel('Normalized lag-1 autocorrelation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='lower left', framealpha=0.95, fontsize=11)
    
    # Add vertical lines for record years
    for year in record_years:
        ax2.axvline(x=year, color='darkslategray', linestyle='--',
                   alpha=0.6, linewidth=2, zorder=1)
        # # Add year label at top of panel
        # ax2.text(year, ax2.get_ylim()[1]*0.97, str(year),
        #         rotation=0, fontsize=11, ha='center', va='top',
        #         fontweight='bold', color='darkslategray',
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_xlim(1988, 2026)
    
    # Extract Kendall statistics for SIV
    tau_siv_oct = ews_siv_october['tau_ac'].iloc[0]
    p_siv_oct = ews_siv_october['p_ac'].iloc[0]
    tau_siv_apr = ews_siv_april['tau_ac'].iloc[0]
    p_siv_apr = ews_siv_april['p_ac'].iloc[0]
    
    # Title with Kendall statistics
    title2 = f"(b) Sea Ice Volume\n"
    # title2 += f"October: τ={tau_siv_oct:.2f}, p={p_siv_oct:.3f}  |  April: τ={tau_siv_apr:.2f}, p={p_siv_apr:.3f}"
    # ax2.set_title(title2, fontsize=12, fontweight='bold', loc='left', pad=10)
    
    plt.tight_layout()
    plt.savefig('Combined_SIE_SIV_AC1_Apr_Oct_aligned.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("✓ Saved: Combined_SIE_SIV_AC1_Apr_Oct_aligned.png")
    print("=" * 70)
    plt.show()
    
else:
    print("\n❌ Could not create combined plot - missing data")
    if sie_april.empty:
        print("   Missing: SIE April data")
    if sie_october.empty:
        print("   Missing: SIE October data")
    if ews_siv_april is None:
        print("   Missing: SIV April data")
    if ews_siv_october is None:
        print("   Missing: SIV October data")