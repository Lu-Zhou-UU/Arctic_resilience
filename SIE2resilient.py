# ============================================
# SEA ICE EXTENT (SIE) RESILIENCE ANALYSIS — Detrended + Monthwise
# ============================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import kendalltau
from scipy.signal import detrend

# ---------------------------
# Loaders & transforms
# ---------------------------
def load_monthly_extent(filepath, start_date="1979-01-01"):
    """
    Load a single-column monthly sea-ice extent time series (km^2) and
    return a DataFrame with ['Date','Extent'] where Extent is in million km^2.
    """
    df = pd.read_csv(filepath, header=None, names=['Extent_km2'])
    dates = pd.date_range(start=pd.Timestamp(start_date), periods=len(df), freq='MS')
    df['Date'] = dates
    df['Extent'] = df['Extent_km2'] / 1e6  # million km^2
    out = df[['Date', 'Extent']].sort_values('Date').reset_index(drop=True)
    print(f"✓ Loaded {len(out)} monthly SIE records "
          f"from {out['Date'].min().date()} to {out['Date'].max().date()}")
    print(f"  Range: {out['Extent'].min():.2f}–{out['Extent'].max():.2f} million km²")
    return out

def extent_to_state(sie_df):
    """
    Map SIE to a state variable X for resilience analysis.
    X = -Extent (million km^2), so more ice => more negative (analogous to E<0).
    """
    out = sie_df.copy()
    out['X'] = -out['Extent'].astype(float)
    out = out.set_index('Date').sort_index()
    return out[['X']]

# ---------------------------
# Helper: detrend within window
# ---------------------------
def detrend_within_window(x, method='linear'):
    """
    Detrend a time series (within each window) using linear or LOESS method.
    """
    if len(x) < 3:
        return x - np.mean(x)
    t = np.arange(len(x))
    if method == 'linear':
        slope, intercept, *_ = stats.linregress(t, x)
        return x - (intercept + slope * t)
    else:
        # Simple LOESS-like smoothing (if statsmodels unavailable)
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(x, size=max(3, len(x)//4), mode='nearest')
        return x - smooth

# ---------------------------
# Rolling EWS (with detrending)
# ---------------------------
def rolling_ews_detrended(E_df, window_years=10, min_points=8,
                          detrend_method='linear', normalize='demean', debug=False):
    """
    Compute rolling variance and lag-1 AC by month after *window detrending*.
    normalize:
      'demean' (default) -> subtract mean only (preserves variance)
      'zscore'          -> subtract & divide by std (flattens variance)
      'none'            -> no centering/scaling after detrending
    """
    df = E_df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month

    # choose anomaly column if present, otherwise raw X
    anom_cols = [c for c in df.columns if 'anom' in c.lower()]
    anom_col = anom_cols[0] if anom_cols else 'X'

    win = window_years * 12
    out = []

    for end in range(win, len(df)):
        sub = df.iloc[end - win:end]
        end_time = sub.index[-1]

        for m in range(1, 13):
            xm = sub.loc[sub['month'] == m, anom_col].dropna().values
            if len(xm) < min_points:
                continue

            # 1) detrend inside the window
            x_d = detrend_within_window(xm, method=detrend_method)

            # 2) optional normalization
            if normalize == 'zscore':
                s = np.std(x_d, ddof=1)
                x_w = (x_d - np.mean(x_d)) / s if s > 1e-8 else x_d * 0.0
            elif normalize == 'demean':
                x_w = x_d - np.mean(x_d)       # <-- preserves variance
            else:  # 'none'
                x_w = x_d

            # 3) EWS metrics
            var = np.var(x_w, ddof=1)
            ac1 = np.corrcoef(x_w[:-1], x_w[1:])[0, 1] if len(x_w) > 2 else np.nan

            out.append({'time': end_time, 'month': m, 'var': var, 'ac1': ac1})

    out_df = pd.DataFrame(out)
    if not out_df.empty:
        out_df = out_df.set_index('time').sort_index()
        print(f"✓ Detrended EWS (normalize='{normalize}') computed: {len(out_df)} records")
    else:
        print("⚠️ No EWS data computed. Check window size or data coverage.")
    return out_df


# ---------------------------
# Per-month Kendall τ summary
# ---------------------------
def kendall_summary(ews_df, target_months=(6, 7, 8, 9)):
    """
    Compute Kendall τ and p-value for each month’s var and ac1 time series.
    """
    print("\n===== Kendall τ Trend Tests (per month) =====")
    rows = []
    for m in target_months:
        sub = ews_df[ews_df['month'] == m]
        if len(sub) < 5: 
            continue
        τv, pv = kendalltau(sub.index.year, sub['var'])
        τa, pa = kendalltau(sub.index.year, sub['ac1'])
        rows.append({'month': m, 'τ_var': τv, 'p_var': pv, 'τ_ac1': τa, 'p_ac1': pa})
        print(f"Month {m:02d}: Var τ={τv:+.2f} (p={pv:.3f}), AC1 τ={τa:+.2f} (p={pa:.3f})")
    return pd.DataFrame(rows)

# ---------------------------
# Plots (same as before)
# ---------------------------
def plot_ews_heatmaps(ews_df, title_prefix='', vmin_var=None, vmax_var=None, vmin_ac1=-1, vmax_ac1=1):
    if ews_df is None or ews_df.empty:
        print("❌ No EWS data to plot.")
        return None, None

    V = ews_df.pivot_table(index='month', columns=ews_df.index, values='var')
    A = ews_df.pivot_table(index='month', columns=ews_df.index, values='ac1')

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    extent_V = [mdates.date2num(V.columns.min()), mdates.date2num(V.columns.max()), 0.5, 12.5]
    im1 = axs[0].imshow(V.values, aspect='auto', origin='lower', extent=extent_V,
                        cmap='YlOrRd', vmin=vmin_var, vmax=vmax_var, interpolation='nearest')
    axs[0].set_ylabel('Month')
    axs[0].set_title(f'{title_prefix} Detrended Rolling Variance', fontweight='bold')
    plt.colorbar(im1, ax=axs[0], label='Variance')

    extent_A = [mdates.date2num(A.columns.min()), mdates.date2num(A.columns.max()), 0.5, 12.5]
    im2 = axs[1].imshow(A.values, aspect='auto', origin='lower', extent=extent_A,
                        cmap='coolwarm', vmin=vmin_ac1, vmax=vmax_ac1, interpolation='nearest')
    axs[1].set_ylabel('Month'); axs[1].set_xlabel('Year')
    axs[1].set_title(f'{title_prefix} Detrended Lag-1 AC', fontweight='bold')
    axs[1].xaxis_date()
    axs[1].xaxis.set_major_locator(mdates.YearLocator(base=5))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.colorbar(im2, ax=axs[1], label='AC(1)')
    plt.tight_layout()
    return fig, axs

def plot_ews_time_series(ews_df, target_months=(3, 6, 9), figsize=(14, 8)):
    if ews_df.empty:
        print("❌ No EWS data to plot."); return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors = plt.cm.tab10(np.linspace(0, 1, 12))
    for m in target_months:
        sub = ews_df[ews_df['month'] == m]
        if sub.empty: continue
        ax1.plot(sub.index, sub['var'], label=month_names[m-1],
                 color=colors[m-1], lw=2, marker='o', alpha=0.8)
        sub2 = sub.dropna(subset=['ac1'])
        ax2.plot(sub2.index, sub2['ac1'], label=month_names[m-1],
                 color=colors[m-1], lw=2, marker='o', alpha=0.8)
    ax1.set_ylabel('Variance (detrended, z-normalized)')
    ax1.set_title('Rolling Variance by Month (detrended)', fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9); ax1.grid(alpha=0.3)
    ax2.axhline(0, color='gray', ls='--', lw=1)
    ax2.set_ylabel('Lag-1 AC'); ax2.set_xlabel('Year')
    ax2.set_title('Rolling AC(1) by Month (detrended)', fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9); ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("SEA ICE EXTENT RESILIENCE ANALYSIS — Detrended + Monthwise")
    print("=" * 70)

    # 1) Load monthly SIE
    sie = load_monthly_extent("SIE_monthly_197901_202506.txt", start_date="1979-01-01")

    # 2) State variable X = -SIE
    X_df = extent_to_state(sie)
    print(f"State X range: [{X_df['X'].min():.2f}, {X_df['X'].max():.2f}]")

    # 3) Compute anomalies (initial step)
    X_anom = X_df.copy()
    X_anom['X_anom'] = X_anom['X'] - X_anom['X'].rolling(12*5, center=True, min_periods=6).mean()  # smooth 5-yr anomaly

    # 4) Rolling EWS (detrended & standardized)
    ews = rolling_ews_detrended(X_anom, window_years=10, min_points=8, detrend_method='linear')

    # 5) Trend diagnostics (Kendall τ)
    kendall_results = kendall_summary(ews, target_months=(6,9))
    kendall_results.to_csv("SIE_EWS_Kendall_summary.csv", index=False)
    print("\nSaved Kendall τ summary → SIE_EWS_Kendall_summary.csv")

    # 6) Visualizations
    fig1, axs1 = plot_ews_heatmaps(ews, title_prefix='SIE-based X (detrended)')
    if fig1:
        fig1.savefig('SIE_EWS_heatmaps_detrended.png', dpi=300, bbox_inches='tight')
    fig2 = plot_ews_time_series(ews, target_months=(6,9))
    if fig2:
        fig2.savefig('SIE_EWS_timeseries_detrended.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nAnalysis complete ✅ — detrended & monthwise standardized EWS computed.")
