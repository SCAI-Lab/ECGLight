import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

def replot_idealized_shap_all_leads(shap_values_csv, top_regions_csv, output_dir, n_timesteps=40, fs=500):
    """
    Replot idealized heartbeat with SHAP values for all leads below it in order of importance
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SHAP values and top regions
    print("Loading SHAP values and top regions...")
    shap_df = pd.read_csv(shap_values_csv)
    top_regions_df = pd.read_csv(top_regions_csv)
    
    print(f"SHAP DataFrame shape: {shap_df.shape}")
    print(f"Number of timesteps inferred from columns: {len([col for col in shap_df.columns if '_t' in col]) // 12}")  # 12 leads
    
    # Calculate mean absolute SHAP values for each feature
    mean_shap_abs = np.abs(shap_df).mean(axis=0)
    
    # Extract unique leads and their overall importance
    leads = []
    lead_importance = {}
    
    for feature_name in mean_shap_abs.index:
        if '_t' in feature_name:
            lead = feature_name.split('_t')[0]
            if lead not in lead_importance:
                lead_importance[lead] = 0
            lead_importance[lead] += mean_shap_abs[feature_name]
    
    # Sort leads by overall importance
    sorted_leads = sorted(lead_importance.items(), key=lambda x: x[1], reverse=True)
    lead_names = [lead for lead, importance in sorted_leads]
    lead_importances = [importance for lead, importance in sorted_leads]
    
    print("Lead importance ranking:")
    for i, (lead, importance) in enumerate(sorted_leads):
        print(f"{i+1}. {lead}: {importance:.4f}")
    
    # Dynamically determine n_timesteps from the data
    actual_timesteps = infer_timesteps_from_data(shap_df, lead_names[0])
    print(f"Using {actual_timesteps} timesteps based on data")
    
    # Create time points in percentage (0-100%)
    time_points_pct = np.linspace(0, 100, actual_timesteps)
    
    # Create main plot with idealized ECG at top and all leads below
    create_main_comprehensive_plot(lead_names, shap_df, time_points_pct, output_dir, lead_importances, actual_timesteps)
    
    return lead_names, lead_importances

def infer_timesteps_from_data(shap_df, sample_lead):
    """Infer number of timesteps from the data"""
    lead_columns = [col for col in shap_df.columns if col.startswith(sample_lead + '_t')]
    return len(lead_columns)

def create_main_comprehensive_plot(lead_names, shap_df, time_points_pct, output_dir, lead_importances, n_timesteps):
    """
    Create main comprehensive plot with idealized ECG at top and all leads' SHAP values below
    """
    n_leads = len(lead_names)
    
    # Create larger figure with better spacing
    fig = plt.figure(figsize=(18, 4 + n_leads * 1.5))  # Increased size
    
    # Create grid with more space for idealized ECG and better spacing for leads
    gs = GridSpec(n_leads + 1, 1, figure=fig, height_ratios=[3] + [1.2] * n_leads, hspace=0.4)
    
    # Plot 1: Idealized ECG at the top
    ax_ideal = fig.add_subplot(gs[0])
    plot_idealized_heartbeat(ax_ideal, time_points_pct)
    
    # Plot 2: SHAP values for each lead below (in order of importance)
    for i, lead in enumerate(lead_names):
        ax_shap = fig.add_subplot(gs[i + 1])
        
        # Get SHAP values for this lead
        lead_shap_values = extract_lead_shap_values(shap_df, lead, n_timesteps)
        
        # Plot SHAP bars for this lead with cardiac cycle regions
        plot_lead_shap_bars(ax_shap, time_points_pct, lead_shap_values, lead, lead_importances[i], i + 1, n_leads)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/idealized_with_all_leads_shap.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/idealized_with_all_leads_shap.pdf', 
                bbox_inches='tight')
    plt.close()
    
    print(f"Main comprehensive plot saved: {output_dir}/idealized_with_all_leads_shap.png")

def extract_lead_shap_values(shap_df, lead, n_timesteps):
    """
    Extract SHAP values for a specific lead across all timesteps
    """
    lead_shap_values = np.zeros(n_timesteps)
    found_features = 0
    
    for t in range(n_timesteps):
        feature_name = f"{lead}_t{t}"
        if feature_name in shap_df.columns:
            # Take mean absolute SHAP value for this time point
            lead_shap_values[t] = np.abs(shap_df[feature_name]).mean()
            found_features += 1
        else:
            print(f"Warning: Feature {feature_name} not found in SHAP DataFrame")
            lead_shap_values[t] = 0
    
    print(f"Lead {lead} - Found {found_features}/{n_timesteps} features, SHAP range: [{lead_shap_values.min():.6f}, {lead_shap_values.max():.6f}]")
    return lead_shap_values

def plot_idealized_heartbeat(ax, time_points_pct):
    """
    Plot accurate idealized ECG heartbeat on given axis with proper waveform characteristics
    """
    # Create time array
    t = time_points_pct
    
    # Initialize ECG signal
    ideal_ecg = np.zeros_like(t, dtype=float)
    
    # Define standard ECG regions with PR segment removed and P-wave extended
    ecg_regions = [
        {'name': 'P Wave', 'description': 'Atrial Depolarization', 'start': 0, 'end': 25, 'center': 12.5, 'color': 'lightblue', 'alpha': 0.5},
        {'name': 'QRS Complex', 'description': 'Ventricular Depolarization', 'start': 25, 'end': 42, 'center': 33.5, 'color': 'lightgreen', 'alpha': 0.5},
        {'name': 'ST Segment', 'description': 'Ventricular Plateau', 'start': 42, 'end': 58, 'center': 50, 'color': 'orange', 'alpha': 0.5},
        {'name': 'T Wave', 'description': 'Ventricular Repolarization', 'start': 58, 'end': 83, 'center': 70.5, 'color': 'violet', 'alpha': 0.5},
        {'name': 'TP Segment', 'description': 'Ventricular Diastole', 'start': 83, 'end': 100, 'center': 91.5, 'color': 'saddlebrown', 'alpha': 0.5}
    ]
    
    # Define precise landmark positions and values FIRST to ensure consistency
    # We align the signal generation to these landmarks
    landmark_data = [
        # P wave peak - Center of P region
        {'time': 12.5, 'label': 'P', 'value': 0.25, 'description': 'Atrial Depolarization'},
        # Q wave nadir - Early QRS
        {'time': 27.0, 'label': 'Q', 'value': -0.15, 'description': 'Start of Ventricular Depolarization'},
        # R wave peak - Center of QRS
        {'time': 33.5, 'label': 'R', 'value': 1.0, 'description': 'Ventricular Depolarization Peak'},
        # S wave nadir - Late QRS
        {'time': 39.0, 'label': 'S', 'value': -0.25, 'description': 'End of Ventricular Depolarization'},
        # T wave peak - Center of T region
        {'time': 70.5, 'label': 'T', 'value': 0.35, 'description': 'Ventricular Repolarization'}
    ]
    
    # Helper to create Gaussian waves
    def gaussian(x, mu, sig, amp):
        return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # Create accurate ECG waveform components using Gaussians for natural shape
    for i, time_val in enumerate(t):
        val = 0.0
        
        # P Wave: centered at 12.5, wide
        val += gaussian(time_val, 12.5, 4.0, 0.25)
        
        # Q Wave: centered at 27.0, narrow
        val += gaussian(time_val, 27.0, 1.2, -0.15)
        
        # R Wave: centered at 33.5, narrow and tall
        val += gaussian(time_val, 33.5, 1.5, 1.0)
        
        # S Wave: centered at 39.0, narrow
        val += gaussian(time_val, 39.0, 1.2, -0.25)
        
        # T Wave: centered at 70.5, wide
        val += gaussian(time_val, 70.5, 6.0, 0.35)
        
        # Assign to array
        ideal_ecg[i] = val
    
    # Plot cardiac cycle regions with increased alpha
    for region in ecg_regions:
        ax.axvspan(region['start'], region['end'], alpha=region['alpha'], color=region['color'], zorder=0)
        
        # Add region labels - name above and description below
        ax.text(region['center'], -0.45, region['name'], 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=region['color']),
                zorder=10)
        ax.text(region['center'], -0.65, region['description'], 
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=region['color']),
                zorder=10)
    
    # Plot idealized ECG
    ax.plot(t, ideal_ecg, 'k-', linewidth=2, label='Idealized ECG', zorder=5)
    ax.set_title('Idealized ECG with Standard Regions', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3, zorder=1)
    
    # Plot ECG landmarks with annotations
    for landmark in landmark_data:
        # Calculate actual value on the curve at this time
        # This ensures the point is exactly ON the line, even if Gaussian summation shifts it slightly
        curve_val = 0.0
        t_lm = landmark['time']
        curve_val += gaussian(t_lm, 12.5, 4.0, 0.25)
        curve_val += gaussian(t_lm, 27.0, 1.2, -0.15)
        curve_val += gaussian(t_lm, 33.5, 1.5, 1.0)
        curve_val += gaussian(t_lm, 39.0, 1.2, -0.25)
        curve_val += gaussian(t_lm, 70.5, 6.0, 0.35)

        ax.plot(t_lm, curve_val, 'ro', markersize=8, zorder=10)
        
        # Position text based on whether it's a peak or trough
        if landmark['value'] > 0:
            text_y_offset = 0.15
            va = 'bottom'
        else:
            text_y_offset = -0.15
            va = 'top'
        
        ax.annotate(landmark['label'], 
                   (t_lm, curve_val), 
                   xytext=(0, text_y_offset * 30),
                   textcoords='offset points',
                   ha='center', va=va,
                   fontweight='bold', 
                   fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='red'),
                   zorder=15)
        
    ax.set_ylim(-0.8, 1.5)  # Increased range to accommodate annotations
    ax.set_xlim(0, 100)
    
    # Remove x-axis labels and ticks for the top plot
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_xlabel('')
    
    ax.tick_params(axis='y', which='major', labelsize=10)
    
    # Add a legend for the waveform
    ax.legend(loc='upper right', framealpha=0.9)

def plot_lead_shap_bars(ax, time_points_pct, shap_importance, lead, lead_importance, rank, total_leads):
    """
    Plot SHAP importance bars for a single lead with cardiac cycle regions
    """
    if len(shap_importance) == len(time_points_pct):
        # Define cardiac cycle regions with PR segment removed and P-wave extended
        ecg_regions = [
            {'name': 'P Wave', 'start': 0, 'end': 25, 'color': 'lightblue', 'alpha': 0.5},
            {'name': 'QRS Complex', 'start': 25, 'end': 42, 'color': 'lightgreen', 'alpha': 0.5},
            {'name': 'ST Segment', 'start': 42, 'end': 58, 'color': 'orange', 'alpha': 0.5},
            {'name': 'T Wave', 'start': 58, 'end': 83, 'color': 'violet', 'alpha': 0.5},
            {'name': 'TP Segment', 'start': 83, 'end': 100, 'color': 'saddlebrown', 'alpha': 0.5}
        ]
        
        # Plot cardiac cycle regions as background
        for region in ecg_regions:
            ax.axvspan(region['start'], region['end'], alpha=region['alpha'], color=region['color'], zorder=0)
        
        bar_width = 100 / len(time_points_pct) * 0.8
        
        # Debug info
        max_shap = np.max(shap_importance) if len(shap_importance) > 0 else 0
        print(f"  {rank}. {lead} - Max SHAP: {max_shap:.6f}")
        
        # Create bars with dark grey color
        if max_shap > 0:
            bars = ax.bar(time_points_pct, shap_importance, alpha=0.8, color='darkslategray', 
                         width=bar_width, edgecolor='black', linewidth=0.5, zorder=5)
            
            # Highlight top region
            top_idx = np.argmax(shap_importance)
            top_value = shap_importance[top_idx]
            top_time = time_points_pct[top_idx]
            
            # Highlight the top bar with red
            bars[top_idx].set_color('red')
            bars[top_idx].set_alpha(1.0)
            bars[top_idx].set_edgecolor('darkred')
            bars[top_idx].set_linewidth(1)
            
            # Add annotation for top region
            if top_value > max_shap * 0.1:  # Only annotate if significant
                ax.text(top_time, top_value * 0.9, 
                       f'{top_time:.1f}%', 
                       ha='center', va='top', fontsize=8, fontweight='bold', zorder=10,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
        else:
            # If no SHAP values, plot zeros with different color
            bars = ax.bar(time_points_pct, shap_importance, alpha=0.3, color='gray', 
                         width=bar_width, edgecolor='darkgray', linewidth=0.5, zorder=5)
        
        # Set labels with better formatting
        ax.set_ylabel(f'{lead}\n({lead_importance:.3f})\nSHAP Value', fontsize=10, rotation=0, 
                     ha='right', va='center', labelpad=10)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_xlim(0, 100)
        
        # Remove x-axis labels and ticks for all but bottom plot
        if rank < total_leads:
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('Cardiac Cycle (%)', fontsize=11)
            ax.tick_params(axis='x', labelsize=9)
        
        # Set y-axis limit based on max SHAP value with some margin
        max_shap_val = max_shap if max_shap > 0 else 0.001
        ax.set_ylim(0, max_shap_val * 1.3)
        ax.tick_params(axis='y', labelsize=8)
        
        # Remove y-axis ticks and labels (the numbers 1-12)
        ax.set_yticklabels([])
        ax.set_yticks([])

def create_lead_importance_table(lead_names, lead_importances, output_dir):
    """
    Create a simple table showing lead importance ranking
    """
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Rank': range(1, len(lead_names) + 1),
        'Lead': lead_names,
        'Total_SHAP_Importance': lead_importances
    })
    
    # Save to CSV
    importance_df.to_csv(f'{output_dir}/lead_importance_ranking.csv', index=False)
    
    # Create simple text summary
    with open(f'{output_dir}/lead_importance_summary.txt', 'w') as f:
        f.write("Lead Importance Ranking (by Total SHAP Value)\n")
        f.write("=" * 50 + "\n")
        for i, (lead, importance) in enumerate(zip(lead_names, lead_importances)):
            f.write(f"{i+1:2d}. {lead:4s}: {importance:.4f}\n")
    
    print(f"Lead importance table saved: {output_dir}/lead_importance_ranking.csv")
    return importance_df

# Function to fix other plots (avg_with_shap plots)
def fix_other_plots(analysis_output_dir, n_timesteps=40):
    """
    Fix the other plots that are not showing SHAP values
    """
    print("\nFixing other plots...")
    
    # Load the necessary data
    shap_values_csv = f'{analysis_output_dir}/shap/shap_values_detailed.csv'
    top_regions_csv = f'{analysis_output_dir}/shap/top_shap_regions.csv'
    
    if not os.path.exists(shap_values_csv) or not os.path.exists(top_regions_csv):
        print("Required files not found. Skipping other plots.")
        return
    
    shap_df = pd.read_csv(shap_values_csv)
    top_regions_df = pd.read_csv(top_regions_csv)
    
    # Get lead names from top regions
    lead_names = top_regions_df['feature'].unique()
    
    # Create time points
    time_points_pct = np.linspace(0, 100, n_timesteps)
    
    # Recreate avg_with_shap plots for each lead
    for lead in lead_names:
        create_fixed_avg_shap_plot(lead, shap_df, time_points_pct, analysis_output_dir, n_timesteps)

def create_fixed_avg_shap_plot(lead, shap_df, time_points_pct, output_dir, n_timesteps):
    """
    Create fixed avg_with_shap plot for a single lead
    """
    # Extract SHAP values
    lead_shap_values = extract_lead_shap_values(shap_df, lead, n_timesteps)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot 1: Placeholder for average heartbeat (you would need the original data for this)
    ax1.text(0.5, 0.5, f'Average Heartbeats - {lead}\n(Original data required)', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=14)
    ax1.set_title(f'Average Heartbeats - {lead}', fontsize=16)
    ax1.set_ylabel('Normalized Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SHAP importance with updated regions
    # Define cardiac cycle regions with PR segment removed and P-wave extended
    ecg_regions = [
        {'name': 'P Wave', 'start': 0, 'end': 25, 'color': 'lightblue', 'alpha': 0.3},
        {'name': 'QRS Complex', 'start': 25, 'end': 42, 'color': 'lightgreen', 'alpha': 0.3},
        {'name': 'ST Segment', 'start': 42, 'end': 58, 'color': 'orange', 'alpha': 0.3},
        {'name': 'T Wave', 'start': 58, 'end': 83, 'color': 'violet', 'alpha': 0.3},
        {'name': 'TP Segment', 'start': 83, 'end': 100, 'color': 'saddlebrown', 'alpha': 0.3}
    ]
    
    # Plot cardiac cycle regions as background
    for region in ecg_regions:
        ax2.axvspan(region['start'], region['end'], alpha=region['alpha'], color=region['color'], zorder=0)
    
    bar_width = 100 / len(time_points_pct) * 0.8
    ax2.bar(time_points_pct, lead_shap_values, alpha=0.7, color='darkslategray', 
            width=bar_width, label='SHAP Importance', zorder=5)
    ax2.set_title(f'SHAP Importance - {lead}', fontsize=16)
    ax2.set_xlabel('Cardiac Cycle (%)')
    ax2.set_ylabel('SHAP Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heartbeat_plots/avg_with_shap_{lead}_FIXED.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fixed plot saved: {output_dir}/heartbeat_plots/avg_with_shap_{lead}_FIXED.png")

# Additional function to create standalone idealized ECG plot
def create_standalone_idealized_ecg_plot(output_dir, n_timesteps=40):
    """
    Create a standalone idealized ECG plot with the updated regions
    """
    time_points_pct = np.linspace(0, 100, n_timesteps)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_idealized_heartbeat(ax, time_points_pct)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/standalone_idealized_ecg.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/standalone_idealized_ecg.pdf', 
                bbox_inches='tight')
    plt.close()
    
    print(f"Standalone idealized ECG plot saved: {output_dir}/standalone_idealized_ecg.png")

# Main execution function
def main():
    """
    Main function to run the SHAP replotting
    """
    # Define paths
    analysis_output_dir = 'results/heartbeat_analysis_500Hz_good'
    shap_values_csv = f'{analysis_output_dir}/shap/shap_values_detailed.csv'
    top_regions_csv = f'{analysis_output_dir}/shap/top_shap_regions.csv'
    output_dir = f'{analysis_output_dir}/shap_replots'
    
    # Check if files exist
    if not os.path.exists(shap_values_csv):
        print(f"Error: SHAP values file not found at {shap_values_csv}")
        return
    if not os.path.exists(top_regions_csv):
        print(f"Error: Top regions file not found at {top_regions_csv}")
        return
    
    print("Starting SHAP replotting...")
    print(f"SHAP values file: {shap_values_csv}")
    print(f"Top regions file: {top_regions_csv}")
    print(f"Output directory: {output_dir}")
    
    # Replot all leads
    lead_names, lead_importances = replot_idealized_shap_all_leads(
        shap_values_csv, top_regions_csv, output_dir
    )
    
    # Create lead importance table
    importance_df = create_lead_importance_table(lead_names, lead_importances, output_dir)
    
    # Create standalone idealized ECG plot
    create_standalone_idealized_ecg_plot(output_dir)
    
    # Fix other plots
    fix_other_plots(analysis_output_dir)
    
    print("\n" + "="*60)
    print("SHAP REPLOTTING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Lead ranking (by importance):")
    for i, (lead, importance) in enumerate(zip(lead_names, lead_importances)):
        print(f"  {i+1:2d}. {lead}: {importance:.4f}")

# Alternative: Function to use in your existing code
def replot_from_existing_analysis(analysis_output_dir):
    """
    Replot from existing analysis output directory
    """
    shap_values_csv = f'{analysis_output_dir}/shap/shap_values_detailed.csv'
    top_regions_csv = f'{analysis_output_dir}/shap/top_shap_regions.csv'
    output_dir = f'{analysis_output_dir}/shap_replots'
    
    return replot_idealized_shap_all_leads(shap_values_csv, top_regions_csv, output_dir)

# Example usage:
if __name__ == "__main__":
    main()