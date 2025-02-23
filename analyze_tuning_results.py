"""
Analysis script for magnetic density tuning results.

This script:
1. Loads results from tuning runs
2. Generates visualizations and statistics
3. Identifies best parameter combinations
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

def load_results(results_dir: Path) -> pd.DataFrame:
    """Load and clean results from CSV file."""
    results_file = results_dir / "results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"No results file found at {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Remove failed trials if error column exists
    if 'error' in df.columns:
        df = df[~df['error'].notna()]
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Verify required columns exist
    required_columns = [
        'avg_density', 'max_density', 'pct_above_0.1_target', 'pct_above_0.5_target',
        'guidance_scale', 'density_weight', 'moment_weight', 'volume_weight', 'target_volume'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in results file: {missing_columns}")
    
    return df

def safe_correlation(x, y):
    """Compute correlation coefficient safely, handling constant inputs."""
    if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
        return 0.0  # No correlation for constant inputs
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return stats.pearsonr(x, y)[0]
    except:
        return 0.0

def plot_parameter_importance(df: pd.DataFrame, output_dir: Path):
    """Create plots showing parameter importance for key metrics."""
    metrics = ['avg_density', 'max_density', 'pct_above_0.1_target', 'pct_above_0.5_target']
    params = ['guidance_scale', 'density_weight', 'moment_weight', 'volume_weight', 'target_volume']
    
    for metric in metrics:
        plt.figure(figsize=(15, 10))
        for i, param in enumerate(params, 1):
            plt.subplot(2, 3, i)
            
            # Only plot if parameter has variation
            if len(df[param].unique()) > 1:
                sns.scatterplot(data=df, x=param, y=metric)
                
                # Add trend line
                try:
                    z = np.polyfit(df[param], df[metric], 1)
                    p = np.poly1d(z)
                    plt.plot(df[param], p(df[param]), "r--", alpha=0.8)
                except:
                    pass  # Skip trend line if fit fails
                
                # Add correlation coefficient
                corr = safe_correlation(df[param], df[metric])
                plt.text(0.05, 0.95, f'r = {corr:.2f}', 
                        transform=plt.gca().transAxes,
                        verticalalignment='top')
            else:
                plt.text(0.5, 0.5, f'Constant value: {df[param].iloc[0]}',
                        ha='center', va='center')
            
            plt.title(f'{metric} vs {param}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'parameter_importance_{metric}.png')
        plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Create correlation matrix heatmap."""
    metrics = ['avg_density', 'max_density', 'pct_above_0.1_target', 'pct_above_0.5_target']
    params = ['guidance_scale', 'density_weight', 'moment_weight', 'volume_weight', 'target_volume']
    
    # Compute correlation matrix manually to handle constant inputs
    cols = params + metrics
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for i in cols:
        for j in cols:
            try:
                corr = safe_correlation(df[i], df[j])
                corr_matrix.loc[i, j] = corr if not np.isnan(corr) else 0.0
            except:
                corr_matrix.loc[i, j] = 0.0
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, fmt='.2f')
    plt.title('Parameter and Metric Correlations')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png')
    plt.close()

def plot_density_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot density distributions for different parameter ranges."""
    params = ['guidance_scale', 'density_weight', 'moment_weight']
    
    for param in params:
        # Only create plot if parameter has enough unique values
        if len(df[param].unique()) >= 2:
            plt.figure(figsize=(10, 6))
            
            try:
                # Try to create bins, handling duplicates
                df['param_bin'] = pd.qcut(df[param], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                
                # Plot density distributions
                sns.boxplot(data=df, x='param_bin', y='avg_density')
                plt.title(f'Density Distribution by {param}')
                plt.xlabel(f'{param} Range')
                plt.ylabel('Average Magnetic Density (μB/Å³)')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'density_dist_{param}.png')
                plt.close()
                
                # Remove temporary bin column
                df.drop('param_bin', axis=1, inplace=True)
            except:
                plt.close()
                print(f"Warning: Could not create distribution plot for {param}")

def find_best_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """Identify best parameter configurations based on different criteria."""
    criteria = {
        'highest_avg_density': df['avg_density'].idxmax(),
        'highest_max_density': df['max_density'].idxmax(),
        'highest_pct_above_0.1': df['pct_above_0.1_target'].idxmax(),
        'highest_pct_above_0.5': df['pct_above_0.5_target'].idxmax()
    }
    
    best_configs = pd.DataFrame()
    for criterion, idx in criteria.items():
        config = df.loc[idx]
        config['criterion'] = criterion
        best_configs = pd.concat([best_configs, config.to_frame().T])
    
    return best_configs

def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate a text report summarizing the results."""
    report_lines = [
        "Magnetic Density Tuning Results",
        "===========================",
        "",
        f"Number of trials: {len(df)}"
    ]
    
    # Add date range if timestamp exists
    if 'timestamp' in df.columns:
        report_lines.extend([
            f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}",
            ""
        ])
    
    report_lines.extend([
        "Overall Statistics",
        "-----------------",
        f"Average magnetic density: {df['avg_density'].mean():.6f} ± {df['avg_density'].std():.6f} μB/Å³",
        f"Maximum magnetic density achieved: {df['max_density'].max():.6f} μB/Å³",
        f"Average % above 0.1 target: {df['pct_above_0.1_target'].mean()*100:.1f}%",
        f"Average % above 0.5 target: {df['pct_above_0.5_target'].mean()*100:.1f}%",
        "",
        "Parameter Ranges",
        "----------------"
    ])
    
    # Add parameter range information
    params = ['guidance_scale', 'density_weight', 'moment_weight', 'volume_weight', 'target_volume']
    for param in params:
        unique_values = sorted(df[param].unique())
        report_lines.append(f"{param}: {unique_values}")
    
    report_lines.extend([
        "",
        "Best Configurations",
        "------------------"
    ])
    
    best_configs = find_best_configurations(df)
    for _, config in best_configs.iterrows():
        report_lines.extend([
            f"\nBest for: {config['criterion']}",
            f"- Guidance scale: {config['guidance_scale']:.1f}",
            f"- Density weight: {config['density_weight']:.1f}",
            f"- Moment weight: {config['moment_weight']:.1f}",
            f"- Volume weight: {config['volume_weight']:.1f}",
            f"- Target volume: {config['target_volume']:.1f}",
            f"Results:",
            f"- Average density: {config['avg_density']:.6f} μB/Å³",
            f"- Maximum density: {config['max_density']:.6f} μB/Å³",
            f"- % above 0.1 target: {config['pct_above_0.1_target']*100:.1f}%",
            f"- % above 0.5 target: {config['pct_above_0.5_target']*100:.1f}%"
        ])
    
    # Write report
    with open(output_dir / "tuning_report.txt", 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description="Analyze magnetic density tuning results")
    parser.add_argument("--results-dir", type=str, default="tuning_results",
                      help="Directory containing results")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory for analysis output (defaults to results-dir/analysis)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    df = load_results(results_dir)
    
    print(f"Loaded {len(df)} trials")
    print("\nSummary of current results:")
    print(f"Average density: {df['avg_density'].mean():.6f} ± {df['avg_density'].std():.6f} μB/Å³")
    print(f"Maximum density: {df['max_density'].max():.6f} μB/Å³")
    print(f"% above 0.1 target: {df['pct_above_0.1_target'].mean()*100:.1f}%")
    print(f"% above 0.5 target: {df['pct_above_0.5_target'].mean()*100:.1f}%")
    
    print("\nParameter ranges:")
    params = ['guidance_scale', 'density_weight', 'moment_weight', 'volume_weight', 'target_volume']
    for param in params:
        unique_values = sorted(df[param].unique())
        print(f"{param}: {unique_values}")
    
    print("\nGenerating visualizations...")
    plot_parameter_importance(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_density_distributions(df, output_dir)
    
    print("Generating report...")
    generate_report(df, output_dir)
    
    print(f"\nAnalysis completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 