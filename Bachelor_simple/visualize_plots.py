import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os


def load_data(filepath):
    # Load CSV data into a DataFrame without hardcoding headers initially
    df = pd.read_csv(filepath)

    # Optional: Print the shape of the data for debugging purposes
    print(f"CSV Loaded. DataFrame shape: {df.shape}")

    return df


def create_individual_coverage_plots(df):
    """Create separate plots for each coverage rate"""

    # Calculate low-risk deaths (total deaths - high-risk deaths)
    df = df.copy()
    df['low_risk_deaths'] = df['final_deaths'] - df['high_risk_deaths']

    # Group by strategy, timing, and coverage_rate
    summary = df.groupby(['strategy', 'timing', 'coverage_rate']).agg({
        'high_risk_deaths': ['mean', 'std'],
        'low_risk_deaths': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['strategy', 'timing', 'coverage_rate',
                       'high_risk_mean', 'high_risk_std',
                       'low_risk_mean', 'low_risk_std']

    # Get unique combinations of strategy and timing
    strategy_timing = summary[['strategy', 'timing']].drop_duplicates()

    # Colors for risk levels
    colors = {'high_risk': '#d62728', 'low_risk': '#2ca02c'}

    # Get coverage rates
    coverage_rates = sorted(summary['coverage_rate'].unique())

    figures = []

    for coverage in coverage_rates:
        print(f"\n=== Coverage Rate: {coverage * 100:.0f}% ===")

        # Create individual figure for this coverage rate
        fig, ax = plt.subplots(figsize=(8, 6))

        coverage_data = summary[summary['coverage_rate'] == coverage]

        # Align with strategy_timing order
        aligned_data = []
        for _, st_row in strategy_timing.iterrows():
            match = coverage_data[
                (coverage_data['strategy'] == st_row['strategy']) &
                (coverage_data['timing'] == st_row['timing'])
                ]
            if len(match) > 0:
                aligned_data.append(match.iloc[0])
            else:
                aligned_data.append({
                    'high_risk_mean': 0, 'low_risk_mean': 0,
                    'high_risk_std': 0, 'low_risk_std': 0
                })

        aligned_df = pd.DataFrame(aligned_data)

        # Print the values for this coverage rate
        for i, (_, st_row) in enumerate(strategy_timing.iterrows()):
            low_risk_deaths = aligned_df['low_risk_mean'].iloc[i]
            high_risk_deaths = aligned_df['high_risk_mean'].iloc[i]
            total_deaths = low_risk_deaths + high_risk_deaths
            low_risk_std = aligned_df['low_risk_std'].iloc[i]
            high_risk_std = aligned_df['high_risk_std'].iloc[i]

            print(f"{st_row['strategy']} {st_row['timing']}:")
            print(f"  Low-risk deaths: {low_risk_deaths:.1f} (±{low_risk_std:.1f})")
            print(f"  High-risk deaths: {high_risk_deaths:.1f} (±{high_risk_std:.1f})")
            print(f"  Total deaths: {total_deaths:.1f}")

        x_positions = np.arange(len(strategy_timing))

        # Create stacked bars
        bars1 = ax.bar(x_positions, aligned_df['low_risk_mean'],
                       label='Low Risk',
                       color=colors['low_risk'],
                       yerr=aligned_df['low_risk_std'], capsize=3)

        bars2 = ax.bar(x_positions, aligned_df['high_risk_mean'],
                       bottom=aligned_df['low_risk_mean'],
                       label='High Risk',
                       color=colors['high_risk'],
                       yerr=aligned_df['high_risk_std'], capsize=3)

        # Add values inside bars
        for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Low risk values (bottom bars)
            low_risk_height = aligned_df['low_risk_mean'].iloc[j]
            if low_risk_height > 0:
                ax.text(bar1.get_x() + bar1.get_width() / 2., low_risk_height / 2,
                        f'{int(low_risk_height):,}',
                        ha='center', va='center', fontweight='bold', color='white')

            # High risk values (top bars)
            high_risk_height = aligned_df['high_risk_mean'].iloc[j]
            if high_risk_height > 0:
                ax.text(bar2.get_x() + bar2.get_width() / 2.,
                        low_risk_height + high_risk_height / 2,
                        f'{int(high_risk_height):,}',
                        ha='center', va='center', fontweight='bold', color='white')

        # Customize plot
        ax.set_title(f'Deaths by Vaccination Strategy and Timing\n({coverage * 100:.0f}% Coverage)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy & Timing', fontsize=12)
        ax.set_ylabel('Average Final Deaths', fontsize=12)

        # Create x-axis labels
        labels = [f"{row['strategy']}\n{row['timing']}" for _, row in strategy_timing.iterrows()]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=0, ha='center')

        # Add grid and legend
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right')

        plt.tight_layout()
        figures.append(fig)

    return figures


def create_coverage_effect_plot(df):
    """Create line plot showing effect of coverage rate on deaths"""
    # Filter for meaningful comparisons (exclude no vaccination)
    df_vacc = df[df['coverage_rate'] > 0].copy()

    plt.figure(figsize=(12, 8))

    # Create subplots for each strategy-timing combination
    combinations = df_vacc.groupby(['strategy', 'timing']).size().index

    for i, (strategy, timing) in enumerate(combinations):
        subset = df_vacc[(df_vacc['strategy'] == strategy) &
                         (df_vacc['timing'] == timing)]

        coverage_summary = subset.groupby('coverage_rate')['final_deaths'].agg(['mean', 'std'])

        plt.errorbar(coverage_summary.index * 100, coverage_summary['mean'],
                     yerr=coverage_summary['std'],
                     label=f'{strategy} {timing}', marker='o', capsize=3)

    plt.xlabel('Vaccination Coverage Rate (%)')
    plt.ylabel('Average Final Deaths')
    plt.title('Effect of Vaccination Coverage Rate on Deaths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def create_high_risk_protection_plot(df):
    """Create plot showing high-risk death rates by strategy"""
    plt.figure(figsize=(10, 6))

    # Calculate high-risk death rates
    df_analysis = df.copy()
    df_analysis['high_risk_death_rate'] = df_analysis['high_risk_deaths'] / (
            df_analysis['high_risk_deaths'] + df_analysis[
        'high_risk_vaccinated'] + 1)  # +1 to avoid division by zero

    # Group by strategy and timing
    summary = df_analysis.groupby(['strategy', 'timing'])['high_risk_death_rate'].agg(['mean', 'std']).reset_index()

    x_positions = np.arange(len(summary))
    bars = plt.bar(x_positions, summary['mean'],
                   yerr=summary['std'], capsize=5, alpha=0.8, color='red')

    plt.xlabel('Vaccination Strategy and Timing')
    plt.ylabel('High-Risk Death Rate')
    plt.title('High-Risk Death Rates by Vaccination Approach')

    labels = [f"{row['strategy']}\n{row['timing']}" for _, row in summary.iterrows()]
    plt.xticks(x_positions, labels)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, summary['mean'])):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    return plt.gcf()


def main():
    """Main function to create all visualizations"""
    # Load data
    df = load_data('data/epidemic_simulation_results.csv')

    if len(df) == 0:
        print("No valid data to visualize")
        return

    print(f"Creating visualizations from {len(df)} valid simulations...")

    # Create individual coverage plots
    coverage_figures = create_individual_coverage_plots(df)

    # Save each coverage plot separately
    coverage_rates = sorted(df['coverage_rate'].unique())
    for i, (fig, coverage) in enumerate(zip(coverage_figures, coverage_rates)):
        filename = f'plots/death_comparison_{int(coverage * 100)}pct.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")

    # Create other plots
    fig2 = create_coverage_effect_plot(df)
    fig2.savefig('plots/coverage_effect.png', dpi=300, bbox_inches='tight')
    print("Saved coverage_effect.png")

    fig3 = create_high_risk_protection_plot(df)
    fig3.savefig('plots/high_risk_protection.png', dpi=300, bbox_inches='tight')
    print("Saved high_risk_protection.png")

    # Show basic statistics
    print("\nBasic Statistics:")
    print(f"Average deaths across all simulations: {df['final_deaths'].mean():.1f}")
    print(f"Average total infected: {df['total_infected'].mean():.1f}")
    print(f"Average epidemic duration: {df['epidemic_duration'].mean():.1f} days")

    plt.show()


if __name__ == "__main__":
    main()
