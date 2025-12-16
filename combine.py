#!/usr/bin/env python
"""
Arctic Climate Analysis: Combined Sea Ice and Albedo Trends
Panel 1: Sea-ice extent & volume anomalies with regime shift analysis
Panel 2: Surface albedo trends from AVHRR & CERES
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
import seaborn as sns
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import matplotlib.dates as mdates # Required for mdates.YearLocator()
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_sea_ice_data(filename):
    """Load sea ice extent data from Excel file"""
    try:
        df = pd.read_excel(filename, sheet_name='September-NH', skiprows=9)
        print(f"✓ Loaded sea ice extent data: {df.shape}")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"❌ Error loading sea ice data: {e}")
        return None

def load_volume_data(filename, dataset_name):
    """Load ice volume data from txt file"""
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Volume'])
        print(f"✓ Loaded {dataset_name} volume data: {len(df)} years")
        return df
    except Exception as e:
        print(f"❌ Error loading {dataset_name} volume: {e}")
        return None

def load_albedo_data(filename, dataset_name):
    """Load albedo data from txt file"""
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Albedo'])
        print(f"✓ Loaded {dataset_name} albedo data: {len(df)} years")
        return df
    except Exception as e:
        print(f"❌ Error loading {dataset_name} albedo: {e}")
        return None
        
# Helper function to ensure consistent date creation
def year_to_start_of_day_timestamp(year):
    """Converts a year (int) to a Timestamp object at Jan 1st, 00:00:00."""
    # pd.to_datetime(year, format='%Y') already does this, but this wrapper
    # makes the intent explicit and is clear that only the day-level is relevant.
    return pd.to_datetime(f'{year}-01-01', format='%Y-%m-%d')
    
def prepare_extent_data(df):
    """Prepare sea ice extent data for plotting"""
    # Auto-detect time and anomaly columns
    time_col = None
    anomaly_col = None
    
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in ['time', 'date', 'year']):
            time_col = col
            break
    
    for col in df.columns:
        if 'anomaly' in str(col).lower():
            anomaly_col = col
            break
    
    if time_col and anomaly_col:
        plot_df = df[[time_col, anomaly_col]].copy().dropna()
        
        # Convert time to numeric year
        if not pd.api.types.is_numeric_dtype(plot_df[time_col]):
            try:
                if pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                    plot_df['Year'] = plot_df[time_col].dt.year
                else:
                    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
                    plot_df['Year'] = plot_df[time_col].dt.year
            except:
                plot_df['Year'] = plot_df[time_col]
        else:
            plot_df['Year'] = plot_df[time_col]
        
        plot_df['Anomaly'] = plot_df[anomaly_col]
        return plot_df[['Year', 'Anomaly']].sort_values('Year')
    
    return None

def compute_loess(x, y, frac=0.3):
    """Compute LOESS smoothing"""
    # Remove NaN values
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return np.full_like(y, np.nan)
    
    try:
        smoothed = lowess(y[mask], x[mask], frac=frac, return_sorted=False)
        result = np.full_like(y, np.nan)
        result[mask] = smoothed
        return result
    except:
        return np.full_like(y, np.nan)

def normalize_to_zscore(data):
    """Convert to z-scores (standardized anomalies)"""
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std > 0:
        return (data - mean) / std
    return data - mean

def create_panel1_sea_ice(extent_df, c3_volume_df, avhrr_volume_df, save_fig=True):
    """
    Panel 1: Sea-ice extent & volume anomalies with regime shift analysis
    """
    print("\n=== CREATING PANEL 1: SEA ICE EXTENT & VOLUME ===")
    
    # Create figure with main plot and inset
    # fig = plt.figure(figsize=(16, 10))
    
    # Main plot
    # ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=2, rowspan=2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # Prepare data
    years = extent_df['Year'].values
    extent_anomaly = extent_df['Anomaly'].values
    
    # Normalize to z-scores
    extent_zscore = normalize_to_zscore(extent_anomaly)
    
    # # REGIME SHIFT BACKGROUND (pre/post 1995)
    # regime_shift_year = 1995
    # ax1.axvspan(years.min(), regime_shift_year, alpha=0.1, color='blue', label='Pre-1995 regime')
    # ax1.axvspan(regime_shift_year, years.max(), alpha=0.1, color='red', label='Post-1995 regime')
    
    # # Add regime shift annotation
    # ax1.annotate('Regime shift ≈ 1995', xy=(1995, 0), xytext=(1995, 1.5),
    #             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    #             fontsize=11, fontweight='bold', ha='center')
    
    # RECORD MINIMA MARKERS
    record_years = [2007, 2012, 2020]
    for year in record_years:
        if year <= years.max():
            ax1.axvline(x=year, color='gray', linestyle='--', alpha=0.5, lw=1)
            ax1.text(year, ax1.get_ylim()[1]*0.95, str(year), 
                    rotation=0, fontsize=12, ha='center', va='top')
    
    # Plot extent anomaly (z-scores)
    color1 = '#2E86AB'
    line1 = ax1.plot(years, extent_zscore, linewidth=1.5, color=color1, 
                     alpha=0.7, marker='o', markersize=4, label='Extent anomaly (z-score)')
    
    # Add LOESS smoothing for extent
    extent_loess = compute_loess(years, extent_zscore, frac=0.3)  # ~10-year window
    ax1.plot(years, extent_loess, linewidth=2.5, color='darkblue', 
             linestyle='--', label='Extent LOESS smooth', alpha=0.8)
    
    # Add LINEAR TREND for extent
    z_extent = np.polyfit(years, extent_zscore, 1)
    p_extent = np.poly1d(z_extent)
    ax1.plot(years, p_extent(years), 
            linewidth=2, color=color1, linestyle=':', 
            alpha=0.5)
    
    # # Zero line
    # ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # First y-axis settings
    ax1.set_xlabel('Year', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Sea Ice Extent Anomaly (z-score)', color=color1, fontsize=18, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2.4,2.2)
    ax1.set_xlim(1978,2026)
    
    # Add volume on second y-axis if available
    if c3_volume_df is not None or avhrr_volume_df is not None:
        ax2 = ax1.twinx()
        
        # Use C3 volume as primary, AVHRR as secondary
        if c3_volume_df is not None:
            # Normalize volume to z-scores
            volume_zscore = normalize_to_zscore(c3_volume_df['Volume'].values)
            
            color2 = '#A23B72'
            ax2.plot(c3_volume_df['Year'], volume_zscore, 
                    linewidth=2, color=color2, alpha=0.8,
                    marker='o', markersize=4, label='Volume (z-score)')
            
            # Add LOESS for volume
            volume_loess = compute_loess(c3_volume_df['Year'].values, volume_zscore, frac=0.3)
            ax2.plot(c3_volume_df['Year'], volume_loess, 
                    linewidth=2.5, color='darkred', linestyle='--',
                    label='Volume LOESS smooth', alpha=0.8)
            
            # Add LINEAR TREND for volume
            z_volume = np.polyfit(c3_volume_df['Year'].values, volume_zscore, 1)
            p_volume = np.poly1d(z_volume)
            ax2.plot(c3_volume_df['Year'], p_volume(c3_volume_df['Year'].values),
                    linewidth=2, color=color2, linestyle=':',
                    alpha=0.5)
            
            ax2.set_ylabel('Sea Ice Volume Anomaly (z-score)', color=color2, 
                          fontsize=18, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=15)
            ax2.set_ylim(-2.4,2.2)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if 'ax2' in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1.extend(lines2)
        labels1.extend(labels2)
    
    ax1.legend(lines1, labels1, loc='lower left', framealpha=0.9, fontsize=15, ncol=2)
    
    # ax1.set_title('Panel 1: Arctic Sea Ice Extent & Volume Anomalies (1979-2024)\nNormalized to z-scores with regime shift analysis',
                 # fontsize=14, fontweight='bold', pad=20)
    
    # # VARIANCE INSET (resilience indicator)
    # ax_inset = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    
    # # Calculate rolling variance (5-year window)
    # window = 5
    # rolling_var = pd.Series(extent_zscore).rolling(window=window, center=True).std()
    
    # ax_inset.fill_between(years, 0, rolling_var, color='gray', alpha=0.3)
    # ax_inset.plot(years, rolling_var, linewidth=2, color='darkgray', label=f'{window}-yr rolling σ')
    
    # # Add LINEAR TREND for variance
    # # Remove NaN values for trend calculation
    # valid_idx = ~np.isnan(rolling_var)
    # if np.sum(valid_idx) > 2:
    #     z_var = np.polyfit(years[valid_idx], rolling_var[valid_idx], 1)
    #     p_var = np.poly1d(z_var)
    #     ax_inset.plot(years, p_var(years), 
    #                  linewidth=1.5, color='red', linestyle=':', 
    #                  alpha=0.6, label=f'Trend ({z_var[0]:.5f}/yr)')
    
    # # # Mark regime shift
    # # ax_inset.axvline(x=regime_shift_year, color='red', linestyle='--', alpha=0.5)
    
    # ax_inset.set_xlabel('Year', fontsize=10)
    # ax_inset.set_ylabel('Variance (σ)', fontsize=10)
    # ax_inset.set_title('Resilience Indicator (variance)', fontsize=10, fontweight='bold')
    # ax_inset.grid(True, alpha=0.3)
    # ax_inset.legend(loc='upper right', fontsize=8)
    
    # # Print trend summary
    # print("\n=== TREND ANALYSIS ===")
    # print(f"Extent trend (z-score): {z_extent[0]:.4f} σ/year")
    # if 'z_volume' in locals():
    #     print(f"Volume trend (z-score): {z_volume[0]:.4f} σ/year")
    # if 'z_var' in locals():
    #     print(f"Variance trend: {z_var[0]:.5f} σ/year")
    #     if z_var[0] > 0:
    #         print("  → Increasing variance indicates declining resilience")
    #     else:
    #         print("  → Decreasing variance indicates stabilizing system")
    
    # # Calculate trends in original units too
    # z_extent_orig = np.polyfit(years, extent_anomaly, 1)
    # print(f"\nExtent trend (original): {z_extent_orig[0]:.4f} million km²/year")
    # if c3_volume_df is not None:
    #     z_volume_orig = np.polyfit(c3_volume_df['Year'].values, c3_volume_df['Volume'].values, 1)
    #     print(f"Volume trend (original): {z_volume_orig[0]:.1f} km³/year")
    
    plt.tight_layout()
    
    if save_fig:
        filename = 'panel1_sea_ice_enhanced.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Panel 1 saved as: {filename}")
    
    plt.show()
    return fig

def format_x_axis_ticks(ax, time_data):
    """
    Format x-axis ticks to show years appropriately, aligning the ticks
    with the start of the year (Jan 1st) for better grid alignment.
    """
    # Determine the time span
    if pd.api.types.is_datetime64_any_dtype(time_data):
        # For datetime data
        min_year = time_data.min().year
        max_year = time_data.max().year
        time_span = max_year - min_year
        
        # Choose tick frequency based on data span
        if time_span <= 20:
            # Every year for <= 20 years
            years = range(min_year, max_year + 1, 1)
            # 🎯 FIX: Anchor to January (month=1) instead of September (month=9)
            tick_positions = [pd.Timestamp(year=year, month=1, day=1) for year in years]
            rotation = 0
        elif time_span <= 50:
            # Every 2 years for 21-50 years
            years = range(min_year, max_year + 1, 2)
            # 🎯 FIX: Anchor to January (month=1) instead of September (month=9)
            tick_positions = [pd.Timestamp(year=year, month=1, day=1) for year in years]
            rotation = 0
        else:
            # Every 5 years for > 50 years
            start_year = (min_year // 5) * 5  # Round down to nearest 5
            years = range(start_year, max_year + 1, 5)
            # 🎯 FIX: Anchor to January (month=1) instead of September (month=9)
            tick_positions = [pd.Timestamp(year=year, month=1, day=1) for year in years]
            rotation = 0
            
        tick_labels = [str(year) for year in years]
            
    else:
        # For numeric data (assuming years) - Logic remains the same
        min_year = int(time_data.min())
        max_year = int(time_data.max())
        time_span = max_year - min_year
        
        if time_span <= 20:
            years = range(min_year-1, max_year, 1)
            rotation = 45
        elif time_span <= 50:
            years = range(min_year-1, max_year, 2)
            rotation = 45
        else:
            start_year = (min_year // 5) * 5
            years = range(start_year, max_year + 1, 5)
            rotation = 0
            
        tick_positions = list(years)
        tick_labels = [str(year) for year in years]
    
    # Set the ticks
    ax.set_xticks(tick_positions)
    # The 'ha' (horizontal alignment) setting is crucial here to ensure the label
    # is positioned correctly relative to the tick mark.
    ax.set_xticklabels(tick_labels, rotation=rotation, ha='right' if rotation > 0 else 'center')
    
    # Add minor ticks for better visual appeal
    if time_span > 20:
        ax.tick_params(axis='x', which='minor', length=3)
        if pd.api.types.is_datetime64_any_dtype(time_data):
            ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    step_size = len(years) // max(1, len(years) // 10) if time_span > 50 else (2 if time_span > 20 else 1)
    # print(f"✓ X-axis formatted: {time_span} year span, showing every {step_size} year(s)")
    
    return time_span

def compute_loess(x, y, frac):
    """
    MOCK implementation of LOESS smoothing for demonstration.
    NOTE: A proper LOESS implementation (like from statsmodels) 
    is typically required, but this uses a simple Savitzky-Golay filter 
    as a placeholder for a smooth trend line.
    """
    # Convert frac to window size (must be odd and > 1)
    window_length = int(len(x) * frac)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(3, window_length) # Ensure minimum window size
    
    try:
        # Use a Savitzky-Golay filter as a smoothing proxy
        smoothed = signal.savgol_filter(y, window_length, 3) # window_length, polynomial order 3
    except ValueError:
        # Fallback for short data sequences
        smoothed = np.full_like(y, np.mean(y))
    
    return smoothed

# ==================== ALBEDO PLOTTING FUNCTION (REQUESTED) ====================

def create_panel2_albedo(avhrr_df, ceres_df, save_fig=True):
    """
    Panel 2: Arctic Surface albedo trends with dual-axis consistent scaling 
    and process linkage.

    Parameters
    ----------
    avhrr_df : pd.DataFrame
        DataFrame with 'Year' and 'Albedo' for AVHRR.
    ceres_df : pd.DataFrame
        DataFrame with 'Year' and 'Albedo' for CERES.
    save_fig : bool
        Whether to save the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    print("\n=== CREATING PANEL 2: SURFACE ALBEDO TRENDS (DUAL-AXIS) ===")
    
    # Check for valid data
    if avhrr_df is None and ceres_df is None:
        print("❌ Cannot create plot: Both AVHRR and CERES data are missing.")
        return None
        
    # Standardize data columns for plotting
    # Convert 'Year' columns to datetime objects for consistent x-axis formatting
    if avhrr_df is not None:
        avhrr_df['Year_dt'] = pd.to_datetime(avhrr_df['Year'], format='%Y')
    if ceres_df is not None:
        ceres_df['Year_dt'] = pd.to_datetime(ceres_df['Year'], format='%Y')

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8)) # Use a single subplot for the main plot as per dual-axis design

    # --- AVHRR Data (Primary Axis - Left) ---
    if avhrr_df is not None:
        color_avhrr = '#1F77B4' # Blue
        
        # 1. Raw Data
        ax1.plot(avhrr_df['Year_dt'], avhrr_df['Albedo'], 
                 linewidth=3, color=color_avhrr, alpha=0.8,
                 marker='o', markersize=4, linestyle='-',
                 label='AVHRR Albedo')
        
        # # 2. LOESS trend (LOESS 'frac' parameter set to 0.4)
        # avhrr_loess = compute_loess(avhrr_df['Year'].values, avhrr_df['Albedo'].values, frac=0.4)
        # ax1.plot(avhrr_df['Year_dt'], avhrr_loess, 
        #          linewidth=2.5, color=color_avhrr, linestyle='--', alpha=0.9,
        #          label='AVHRR LOESS Smooth')
        
        # 3. Linear trend
        z = np.polyfit(avhrr_df['Year'].values, avhrr_df['Albedo'].values, 1)
        p = np.poly1d(z)
        ax1.plot(avhrr_df['Year_dt'], p(avhrr_df['Year']), 
                 linewidth=2.5, color=color_avhrr, linestyle=':', alpha=0.8)

    # --- CERES Data (Secondary Axis - Right) ---
    ax2 = None
    if ceres_df is not None:
        # Create secondary axis for CERES
        ax2 = ax1.twinx()
        color_ceres = '#FF7F0E' # Orange
        
        # 1. Raw Data
        ax2.plot(ceres_df['Year_dt'], ceres_df['Albedo'],
                 linewidth=3, color=color_ceres, alpha=0.8,
                 marker='s', markersize=4, linestyle='-',
                 label='CERES Albedo')
        
        # # 2. LOESS trend (LOESS 'frac' parameter set to 0.4)
        # ceres_loess = compute_loess(ceres_df['Year'].values, ceres_df['Albedo'].values, frac=0.4)
        # ax2.plot(ceres_df['Year_dt'], ceres_loess,
        #          linewidth=2.5, color=color_ceres, linestyle='--', alpha=0.9,
        #          label='CERES LOESS Smooth')
        
        # 3. Linear trend
        z = np.polyfit(ceres_df['Year'].values, ceres_df['Albedo'].values, 1)
        p = np.poly1d(z)
        ax2.plot(ceres_df['Year_dt'], p(ceres_df['Year']),
                 linewidth=2.5, color=color_ceres, linestyle=':', alpha=0.8)
        
        # Axis Formatting for CERES
        ax2.set_ylabel('CERES Albedo', color=color_ceres, fontsize=18, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_ceres, labelsize=15)
        ax2.grid(False) # Disable grid on secondary axis

    # --- Common & AVHRR Formatting (Primary Axis - Left) ---
    ax1.set_xlabel('Year', fontsize=18, fontweight='bold')
    if avhrr_df is not None:
        ax1.set_ylabel('AVHRR Albedo', color=color_avhrr, fontsize=18, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_avhrr, labelsize=15)
        # Apply custom x-axis formatting from the style example
        format_x_axis_ticks(ax1, avhrr_df['Year_dt'])
    else:
        # If no AVHRR, use CERES data for x-axis range and format
        ax1.set_ylabel('AVHRR Albedo (Missing)', fontsize=18, fontweight='bold')
        format_x_axis_ticks(ax1, ceres_df['Year_dt'])


    # # Set common x-limits for alignment
    # min_year = min([df['Year'].min() for df in [avhrr_df, ceres_df] if df is not None])
    # max_year = max([df['Year'].max() for df in [avhrr_df, ceres_df] if df is not None])
    # # Convert to datetime objects for x-axis limits
    # start_lim = pd.to_datetime(min_year-4, format='%Y')
    # end_lim = pd.to_datetime(max_year+2, format='%Y')
    # ax1.set_xlim(start_lim, end_lim)

    # Set common x-limits for alignment
    min_year = min([df['Year'].min() for df in [avhrr_df, ceres_df] if df is not None])
    max_year = max([df['Year'].max() for df in [avhrr_df, ceres_df] if df is not None])

    # Convert the integer year limits to Timestamp objects, anchored at Jan 1st.
    # This results in a full Timestamp (e.g., 1978-01-01 00:00:00) which is
    # necessary for ax1.set_xlim to work with datetime data.
    start_lim = year_to_start_of_day_timestamp(min_year - 4)
    end_lim = year_to_start_of_day_timestamp(max_year + 2)
    ax1.tick_params(axis='x', labelcolor='k', labelsize=15)

    ax1.set_xlim(start_lim, end_lim)
    
    ax1.grid(True, alpha=0.3, zorder=0)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1.extend(lines2)
        labels1.extend(labels2)
    
    ax1.legend(lines1, labels1, loc='lower left', framealpha=0.9, fontsize=12, ncol=2)
    
    # # Title
    # avhrr_range = f"AVHRR ({avhrr_df['Year'].min()}-{avhrr_df['Year'].max()})" if avhrr_df is not None else 'AVHRR (Missing)'
    # ceres_range = f"CERES ({ceres_df['Year'].min()}-{ceres_df['Year'].max()})" if ceres_df is not None else 'CERES (Missing)'

    # ax1.set_title(f'Panel 2: Arctic Surface Albedo Trends (Annual Mean)\n{avhrr_range} & {ceres_range}',
    #               fontsize=14, fontweight='bold', pad=20)
    
    # Final layout adjustments
    plt.tight_layout()
    
    if save_fig:
        filename = 'panel2_albedo_styled.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Panel 2 saved as: {filename}")
    
    plt.show()
    return fig

def main():
    """
    Main function to run the combined analysis
    """
    print("=== ARCTIC CLIMATE SYSTEM ANALYSIS ===")
    print("Panel 1: Sea ice extent & volume with regime shift")
    print("Panel 2: Surface albedo trends and process linkages")
    
    # File paths
    extent_file = "Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v4.0.xlsx"
    c3_volume_file = "/Volumes/Yotta_1/C3_ice_volume.txt"
    avhrr_volume_file = "/Volumes/Yotta_1/AVHRR_ice_volume.txt"
    avhrr_albedo_file = "AVHRR_albedo.txt"
    ceres_albedo_file = "CERES_albedo.txt"
    
    # Load all datasets
    print("\nLoading datasets...")
    extent_raw = load_sea_ice_data(extent_file)
    c3_volume_df = load_volume_data(c3_volume_file, "C3")
    avhrr_volume_df = load_volume_data(avhrr_volume_file, "AVHRR")
    avhrr_albedo_df = load_albedo_data(avhrr_albedo_file, "AVHRR")
    ceres_albedo_df = load_albedo_data(ceres_albedo_file, "CERES")
    
    # Prepare extent data
    extent_df = None
    if extent_raw is not None:
        extent_df = prepare_extent_data(extent_raw)
        if extent_df is not None:
            print(f"✓ Extent data prepared: {len(extent_df)} years")
    
    if extent_df is None:
        print("❌ Failed to prepare extent data")
        return
    
    # Create individual panels
    print("\nGenerating visualizations...")
    
    # Panel 1: Sea ice
    fig1 = create_panel1_sea_ice(extent_df, c3_volume_df, avhrr_volume_df)
    
    # Panel 2: Albedo
    fig2 = create_panel2_albedo(avhrr_albedo_df, ceres_albedo_df)
    
    # # Combined figure
    # fig3 = create_combined_figure(extent_df, c3_volume_df, avhrr_volume_df,
                                 # avhrr_albedo_df, ceres_albedo_df)
    
    # Analysis summary
    print("\n=== ANALYSIS SUMMARY ===")
    
    # Regime shift analysis
    pre_1995 = extent_df[extent_df['Year'] < 1995]['Anomaly'].mean()
    post_1995 = extent_df[extent_df['Year'] >= 1995]['Anomaly'].mean()
    print(f"\nRegime Shift Analysis:")
    print(f"  Pre-1995 mean anomaly: {pre_1995:.3f} million km²")
    print(f"  Post-1995 mean anomaly: {post_1995:.3f} million km²")
    print(f"  Change: {post_1995 - pre_1995:.3f} million km²")
    
    # Trend analysis
    if len(extent_df) > 2:
        slope, intercept = np.polyfit(extent_df['Year'], extent_df['Anomaly'], 1)
        print(f"\nExtent trend: {slope:.4f} million km²/year")
    
    if avhrr_albedo_df is not None and len(avhrr_albedo_df) > 2:
        slope, intercept = np.polyfit(avhrr_albedo_df['Year'], avhrr_albedo_df['Albedo'], 1)
        print(f"AVHRR albedo trend: {slope:.5f}/year")
    
    if ceres_albedo_df is not None and len(ceres_albedo_df) > 2:
        slope, intercept = np.polyfit(ceres_albedo_df['Year'], ceres_albedo_df['Albedo'], 1)
        print(f"CERES albedo trend: {slope:.5f}/year")
    
    print("\n✓ Analysis complete!")
    print("✓ Generated Panel 1 (sea ice), Panel 2 (albedo), and combined figure")

if __name__ == "__main__":
    main()
