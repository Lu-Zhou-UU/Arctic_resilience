#!/usr/bin/env python
"""
Plot Arctic Albedo Analysis from AVHRR and CERES datasets
Comparison of two different satellite-based albedo measurements
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_albedo_data(filename, dataset_name):
    """
    Load albedo data from txt file
    
    Parameters
    ----------
    filename : str
        Path to the txt file
    dataset_name : str
        Name of the dataset for display purposes
        
    Returns
    -------
    pd.DataFrame
        Albedo data with Year and Albedo columns
    """
    try:
        # Read the txt file
        df = pd.read_csv(filename, sep=' ', header=None, names=['Year', 'Albedo'])
        print(f"✓ Successfully loaded {dataset_name} albedo data from {filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"  Albedo range: {df['Albedo'].min():.3f} - {df['Albedo'].max():.3f}")
        
        # Display first few rows
        print(f"\nFirst few rows of {dataset_name} albedo data:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name} albedo data: {e}")
        print(f"Tip: Check if the file {filename} exists and has the correct format (year albedo)")
        return None

def create_individual_plots(avhrr_df, ceres_df, save_fig=True):
    """
    Create individual plots for each dataset due to scale differences
    """
    print(f"\n=== CREATING INDIVIDUAL ALBEDO PLOTS ===")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot AVHRR data (top subplot)
    if avhrr_df is not None:
        ax1.plot(avhrr_df['Year'], avhrr_df['Albedo'], 
                linewidth=2.5, color='steelblue', alpha=0.8, 
                marker='o', markersize=4, label='AVHRR Albedo')
        ax1.scatter(avhrr_df['Year'], avhrr_df['Albedo'], 
                   s=25, color='darkblue', alpha=0.7, zorder=5)
        
        ax1.set_ylabel('AVHRR Albedo', fontsize=12, fontweight='bold', color='steelblue')
        ax1.set_title('AVHRR Arctic Albedo (1982-2022)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Set appropriate y-limits for AVHRR
        y_margin = (avhrr_df['Albedo'].max() - avhrr_df['Albedo'].min()) * 0.1
        ax1.set_ylim(avhrr_df['Albedo'].min() - y_margin, avhrr_df['Albedo'].max() + y_margin)
    
    # Plot CERES data (bottom subplot)
    if ceres_df is not None:
        ax2.plot(ceres_df['Year'], ceres_df['Albedo'], 
                linewidth=2.5, color='darkorange', alpha=0.8, 
                marker='s', markersize=4, label='CERES Albedo')
        ax2.scatter(ceres_df['Year'], ceres_df['Albedo'], 
                   s=25, color='darkorange', alpha=0.7, zorder=5)
        
        ax2.set_ylabel('CERES Albedo', fontsize=12, fontweight='bold', color='darkorange')
        ax2.set_title('CERES Arctic Albedo (2000-2024)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        # Set appropriate y-limits for CERES
        y_margin = (ceres_df['Albedo'].max() - ceres_df['Albedo'].min()) * 0.1
        ax2.set_ylim(ceres_df['Albedo'].min() - y_margin, ceres_df['Albedo'].max() + y_margin)
    
    # Common x-axis label
    ax2.set_xlabel('Year', fontsize=18, fontweight='bold')
    
    # Main title
    fig.suptitle('Arctic Albedo: Separate Scale Analysis\nAVHRR vs CERES Satellite Measurements', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if save_fig:
        filename = 'arctic_albedo_individual_plots.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Individual albedo plots saved as: {filename}")
    
    # Show plot
    plt.show()
    
    return fig, (ax1, ax2)

def create_dual_axis_albedo_plot(avhrr_df, ceres_df, save_fig=True):
    """
    Create a dual-axis plot with separate scales for AVHRR and CERES albedo
    PRIMARY VISUALIZATION METHOD due to different scales
    """
    print(f"\n=== CREATING DUAL-AXIS ALBEDO PLOT (PRIMARY METHOD) ===")
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot AVHRR data on left axis
    if avhrr_df is not None:
        color1 = 'steelblue'
        ax1.set_xlabel('Year', fontsize=18, fontweight='bold')
        ax1.set_ylabel('AVHRR Albedo', color=color1, fontsize=18, fontweight='bold')
        
        line1 = ax1.plot(avhrr_df['Year'], avhrr_df['Albedo'], 
                        linewidth=3, color=color1, alpha=0.8, 
                        marker='o', markersize=5, label='AVHRR Albedo')
        ax1.scatter(avhrr_df['Year'], avhrr_df['Albedo'], 
                   s=40, color='darkblue', alpha=0.7, zorder=5)
        ax1.tick_params(axis='x', labelcolor='k', labelsize=15)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=15)
        ax1.grid(True, alpha=0.3, zorder=0)
        
        # Set optimal y-limits for AVHRR
        y_range = avhrr_df['Albedo'].max() - avhrr_df['Albedo'].min()
        y_margin = y_range * 0.1
        ax1.set_ylim(avhrr_df['Albedo'].min() - y_margin, 
                     avhrr_df['Albedo'].max() + y_margin)
    
    # Create secondary axis for CERES data
    ax2 = None
    if ceres_df is not None:
        ax2 = ax1.twinx()
        color2 = 'darkorange'
        ax2.set_ylabel('CERES Albedo', color=color2, fontsize=18, fontweight='bold')
        
        line2 = ax2.plot(ceres_df['Year'], ceres_df['Albedo'], 
                        linewidth=3, color=color2, alpha=0.8, 
                        marker='s', markersize=5, label='CERES Albedo')
        ax2.scatter(ceres_df['Year'], ceres_df['Albedo'], 
                   s=40, color='darkorange', alpha=0.7, zorder=5)
        
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=15)
        ax2.grid(False)  # Disable grid to prevent misalignment
        
        # Set optimal y-limits for CERES
        y_range = ceres_df['Albedo'].max() - ceres_df['Albedo'].min()
        y_margin = y_range * 0.1
        ax2.set_ylim(ceres_df['Albedo'].min() - y_margin, 
                     ceres_df['Albedo'].max() + y_margin)
    
    # # Add title
    # ax1.set_title('Arctic Albedo: Dual-Axis Comparison\nAVHRR (1982-2022) vs CERES (2000-2024) - Optimized Scales', 
    #              fontsize=14, fontweight='bold', pad=20)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels() if avhrr_df is not None else ([], [])
    lines2, labels2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', framealpha=0.9, fontsize=18)
    
    # Print scale information
    if avhrr_df is not None and ceres_df is not None:
        print(f"\nScale Information:")
        print(f"  AVHRR range: {avhrr_df['Albedo'].min():.3f} - {avhrr_df['Albedo'].max():.3f}")
        print(f"  CERES range: {ceres_df['Albedo'].min():.3f} - {ceres_df['Albedo'].max():.3f}")
        scale_ratio = ceres_df['Albedo'].mean() / avhrr_df['Albedo'].mean()
        print(f"  CERES/AVHRR mean ratio: {scale_ratio:.2f}")
        print(f"  CERES values are ~{scale_ratio:.1f}x higher than AVHRR")
    
    # # Add scale indicator text box
    # if avhrr_df is not None and ceres_df is not None:
    #     textstr = f'Scale Difference:\nCERES ≈ {scale_ratio:.1f}× AVHRR\n\nDifferent measurement\nmethodologies'
    #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #     ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
    #             verticalalignment='top', bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    if save_fig:
        filename = 'arctic_albedo_dual_axis_primary.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ PRIMARY dual-axis albedo plot saved as: {filename}")
    
    # Show plot
    plt.show()
    
    return fig, ax1, ax2

def create_overlapping_period_plot(avhrr_df, ceres_df, save_fig=True):
    """
    Create a plot focusing on the overlapping time period
    """
    print(f"\n=== CREATING OVERLAPPING PERIOD ANALYSIS ===")
    
    if avhrr_df is None or ceres_df is None:
        print("Cannot create overlapping analysis - one or both datasets missing")
        return None, None
    
    # Find overlapping years
    avhrr_years = set(avhrr_df['Year'])
    ceres_years = set(ceres_df['Year'])
    overlap_years = sorted(avhrr_years.intersection(ceres_years))
    
    if len(overlap_years) == 0:
        print("No overlapping years found between datasets")
        return None, None
    
    print(f"Overlapping period: {min(overlap_years)} - {max(overlap_years)} ({len(overlap_years)} years)")
    
    # Filter data for overlapping period
    avhrr_overlap = avhrr_df[avhrr_df['Year'].isin(overlap_years)].sort_values('Year')
    ceres_overlap = ceres_df[ceres_df['Year'].isin(overlap_years)].sort_values('Year')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top plot: Time series comparison
    ax1.plot(avhrr_overlap['Year'], avhrr_overlap['Albedo'], 
            linewidth=2.5, color='steelblue', alpha=0.8, 
            marker='o', markersize=5, label='AVHRR')
    ax1.plot(ceres_overlap['Year'], ceres_overlap['Albedo'], 
            linewidth=2.5, color='darkorange', alpha=0.8, 
            marker='s', markersize=5, label='CERES')
    
    ax1.set_xlabel('Year', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Albedo', fontsize=20, fontweight='bold')
    ax1.set_title(f'Overlapping Period Comparison ({min(overlap_years)}-{max(overlap_years)})', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(framealpha=0.9)
    
    # Bottom plot: Scatter plot correlation
    ax2.scatter(avhrr_overlap['Albedo'], ceres_overlap['Albedo'], 
               s=50, color='purple', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add correlation line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        avhrr_overlap['Albedo'], ceres_overlap['Albedo'])
    line_x = np.array([avhrr_overlap['Albedo'].min(), avhrr_overlap['Albedo'].max()])
    line_y = slope * line_x + intercept
    ax2.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.8, 
            label=f'R² = {r_value**2:.3f}, p = {p_value:.3f}')
    
    ax2.set_xlabel('AVHRR Albedo', fontsize=20, fontweight='bold')
    ax2.set_ylabel('CERES Albedo', fontsize=20, fontweight='bold')
    ax2.set_title('AVHRR vs CERES Correlation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(framealpha=0.9)
    
    # Add year labels to some points
    for i in range(0, len(overlap_years), max(1, len(overlap_years)//5)):
        idx = avhrr_overlap['Year'] == overlap_years[i]
        if idx.any():
            x_val = avhrr_overlap[idx]['Albedo'].iloc[0]
            y_val = ceres_overlap[ceres_overlap['Year'] == overlap_years[i]]['Albedo'].iloc[0]
            ax2.annotate(str(overlap_years[i]), (x_val, y_val), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=18, alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    if save_fig:
        filename = 'arctic_albedo_overlapping_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Overlapping period analysis saved as: {filename}")
    
    # Show plot
    plt.show()
    
    return fig, (ax1, ax2)

def analyze_albedo_trends(df, dataset_name):
    """
    Analyze trends in albedo data
    """
    print(f"\n=== {dataset_name.upper()} ALBEDO TREND ANALYSIS ===")
    
    if df is None:
        print(f"{dataset_name} data not available")
        return
    
    # Basic statistics
    print(f"{dataset_name} Albedo Data Summary:")
    print(f"  Time period: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  Number of years: {len(df)}")
    print(f"  Mean albedo: {df['Albedo'].mean():.4f}")
    print(f"  Standard deviation: {df['Albedo'].std():.4f}")
    print(f"  Minimum albedo: {df['Albedo'].min():.4f} ({df.loc[df['Albedo'].idxmin(), 'Year']})")
    print(f"  Maximum albedo: {df['Albedo'].max():.4f} ({df.loc[df['Albedo'].idxmax(), 'Year']})")
    
    # Trend calculation
    if len(df) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Year'], df['Albedo'])
        
        print(f"\n{dataset_name} Albedo Trend Analysis:")
        print(f"  Linear trend: {slope:.6f} per year")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Standard error: {std_err:.6f}")
        
        if p_value < 0.05:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
            
        if slope < 0:
            print(f"  ↓ Declining trend (albedo decreasing) - {significance}")
        else:
            print(f"  ↑ Increasing trend (albedo increasing) - {significance}")
        
        # Calculate total change over period
        total_change = slope * (df['Year'].max() - df['Year'].min())
        print(f"  Total change over period: {total_change:.4f}")
        print(f"  Percentage change: {(total_change / df['Albedo'].mean()) * 100:.2f}%")

def compare_albedo_datasets(avhrr_df, ceres_df):
    """
    Compare the two albedo datasets during overlapping period
    """
    print(f"\n=== ALBEDO DATASET COMPARISON ===")
    
    if avhrr_df is None or ceres_df is None:
        print("Cannot compare datasets - one or both are missing")
        return
    
    # Find overlapping years
    avhrr_years = set(avhrr_df['Year'])
    ceres_years = set(ceres_df['Year'])
    overlap_years = sorted(avhrr_years.intersection(ceres_years))
    
    print(f"Dataset Coverage:")
    print(f"  AVHRR years: {avhrr_df['Year'].min()} - {avhrr_df['Year'].max()} ({len(avhrr_df)} years)")
    print(f"  CERES years: {ceres_df['Year'].min()} - {ceres_df['Year'].max()} ({len(ceres_df)} years)")
    print(f"  Overlapping years: {len(overlap_years)} years ({min(overlap_years)}-{max(overlap_years)})")
    
    if len(overlap_years) > 0:
        # Analyze overlapping period
        avhrr_overlap = avhrr_df[avhrr_df['Year'].isin(overlap_years)].sort_values('Year')
        ceres_overlap = ceres_df[ceres_df['Year'].isin(overlap_years)].sort_values('Year')
        
        print(f"\nOverlapping Period Analysis ({min(overlap_years)} - {max(overlap_years)}):")
        print(f"  AVHRR mean albedo: {avhrr_overlap['Albedo'].mean():.4f}")
        print(f"  CERES mean albedo: {ceres_overlap['Albedo'].mean():.4f}")
        print(f"  Mean difference (CERES - AVHRR): {ceres_overlap['Albedo'].mean() - avhrr_overlap['Albedo'].mean():.4f}")
        print(f"  Relative difference: {((ceres_overlap['Albedo'].mean() - avhrr_overlap['Albedo'].mean()) / avhrr_overlap['Albedo'].mean()) * 100:.1f}%")
        
        # Calculate correlation
        if len(avhrr_overlap) == len(ceres_overlap):
            correlation = np.corrcoef(avhrr_overlap['Albedo'], ceres_overlap['Albedo'])[0, 1]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                avhrr_overlap['Albedo'], ceres_overlap['Albedo'])
            
            print(f"\nCorrelation Analysis:")
            print(f"  Correlation coefficient: {correlation:.4f}")
            print(f"  R-squared: {r_value**2:.4f}")
            print(f"  Linear relationship: CERES = {slope:.3f} × AVHRR + {intercept:.3f}")
            print(f"  P-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print(f"  Correlation is statistically significant")
            else:
                print(f"  Correlation is not statistically significant")

def main():
    """
    Main function to run the albedo analysis
    """
    print("=== Arctic Albedo Analysis: AVHRR vs CERES (Dual-Axis Focus) ===")
    print("Note: Using separate y-axes due to significant scale differences")
    print("AVHRR: ~0.11-0.23, CERES: ~0.34-0.42")
    
    # File names
    avhrr_filename = "AVHRR_albedo.txt"
    ceres_filename = "CERES_albedo.txt"
    
    # Load albedo data
    avhrr_df = load_albedo_data(avhrr_filename, "AVHRR")
    ceres_df = load_albedo_data(ceres_filename, "CERES")
    
    if avhrr_df is None and ceres_df is None:
        print("❌ Failed to load both albedo datasets. Exiting.")
        return
    
    # Create visualizations - emphasizing dual-axis approach
    if avhrr_df is not None and ceres_df is not None:
        # PRIMARY: Dual-axis plot (optimal for different scales)
        fig1, ax1_1, ax1_2 = create_dual_axis_albedo_plot(avhrr_df, ceres_df)
        
        # # SECONDARY: Individual plots for detailed view
        # fig2, (ax2_1, ax2_2) = create_individual_plots(avhrr_df, ceres_df)
        
        # # TERTIARY: Overlapping period correlation analysis
        # fig3, (ax3_1, ax3_2) = create_overlapping_period_plot(avhrr_df, ceres_df)
        
        # # Compare datasets
        # compare_albedo_datasets(avhrr_df, ceres_df)
        
    elif avhrr_df is not None:
        # Only AVHRR data available
        print("Only AVHRR data available - creating single dataset plot")
        fig, (ax1, ax2) = create_individual_plots(avhrr_df, None)
        
    elif ceres_df is not None:
        # Only CERES data available  
        print("Only CERES data available - creating single dataset plot")
        fig, (ax1, ax2) = create_individual_plots(None, ceres_df)
    
    # Analyze trends for individual datasets
    analyze_albedo_trends(avhrr_df, "AVHRR")
    analyze_albedo_trends(ceres_df, "CERES")
    
    print(f"\n✓ Albedo analysis complete!")
    print(f"✓ Primary visualization: Dual-axis plot optimized for scale differences")
    print(f"✓ Secondary visualizations: Individual plots and correlation analysis")

if __name__ == "__main__":
    main()