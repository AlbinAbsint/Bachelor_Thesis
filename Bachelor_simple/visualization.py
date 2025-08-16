import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_and_prepare_data(filepath='data/epidemic_simulation_results.csv'):
    """Load and prepare the simulation data"""
    df = pd.read_csv(filepath)
    return df


def create_network_comparison_table(df):
    """Create a clean comparison table for network types"""
    # Group by rewiring probability and calculate statistics
    network_stats = df.groupby('rewiring_prob').agg({
        'final_deaths': ['mean', 'std'],
        'vaccines_effective': ['mean', 'std']
    }).round(2)

    # Flatten column names
    network_stats.columns = ['deaths_mean', 'deaths_std', 'vaccines_mean', 'vaccines_std']

    return network_stats


def create_interaction_effects_table(df):
    """Create interaction effects table"""
    interaction_stats = df.groupby(['rewiring_prob', 'strategy', 'timing']).agg({
        'final_deaths': 'mean',
        'high_risk_deaths': 'mean'
    }).round(1)

    return interaction_stats


def plot_network_comparison(df):
    """Create a clean bar plot comparing network types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Deaths comparison
    network_stats = df.groupby('rewiring_prob')['final_deaths'].agg(['mean', 'std'])

    bars1 = ax1.bar(['Small-world\n(p=0.3)', 'Random\n(p=0.9)'],
                    network_stats['mean'],
                    yerr=network_stats['std'],
                    capsize=5, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax1.set_ylabel('Final Deaths')
    ax1.set_title('Network Type Comparison: Total Mortality')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars1, network_stats['mean'], network_stats['std']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 1,
                 f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')

    # Final epidemic size comparison (total affected: recovered + vaccinated + deaths)
    df['final_epidemic_size'] = df['total_infected']
    epidemic_stats = df.groupby('rewiring_prob')['final_epidemic_size'].agg(['mean', 'std'])

    bars2 = ax2.bar(['Small-world\n(p=0.3)', 'Random\n(p=0.9)'],
                    epidemic_stats['mean'],
                    yerr=epidemic_stats['std'],
                    capsize=5, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2.set_ylabel('Final Epidemic Size')
    ax2.set_title('Network Type Comparison: Total Affected')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars2, epidemic_stats['mean'], epidemic_stats['std']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + std + 10,
                 f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/network_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_strategy_comparison_bars(df):
    """Create bar plots comparing vaccination strategies across network types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    interaction_data = df.groupby(['rewiring_prob', 'strategy', 'timing']).agg({
        'final_deaths': 'mean',
        'high_risk_deaths': 'mean'
    }).reset_index()

    # Create labels for combinations
    interaction_data['combo'] = (interaction_data['strategy'] + '\n' +
                                 interaction_data['timing'])

    # Filter for specific network types
    small_world = interaction_data[interaction_data['rewiring_prob'] == 0.3]
    random_net = interaction_data[interaction_data['rewiring_prob'] == 0.9]

    x_pos = np.arange(len(small_world))
    width = 0.35

    # Plot 1: Final deaths by network type and strategy/timing
    bars1 = ax1.bar(x_pos - width / 2, small_world['final_deaths'], width,
                    label='Small-world (p=0.3)', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x_pos + width / 2, random_net['final_deaths'], width,
                    label='Random (p=0.9)', color='#A23B72', alpha=0.8)

    ax1.set_xlabel('Strategy & Timing')
    ax1.set_ylabel('Final Deaths')
    ax1.set_title('Final Deaths by Network Type and Vaccination Strategy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(small_world['combo'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: High-risk deaths
    bars3 = ax2.bar(x_pos - width / 2, small_world['high_risk_deaths'], width,
                    label='Small-world (p=0.3)', color='#2E86AB', alpha=0.8)
    bars4 = ax2.bar(x_pos + width / 2, random_net['high_risk_deaths'], width,
                    label='Random (p=0.9)', color='#A23B72', alpha=0.8)

    ax2.set_xlabel('Strategy & Timing')
    ax2.set_ylabel('High-Risk Deaths')
    ax2.set_title('High-Risk Deaths by Network Type and Vaccination Strategy')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(small_world['combo'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/strategy_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_strategy_comparison_heatmaps(df):
    """Create heatmaps showing vaccination strategy effectiveness"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Prepare data
    interaction_data = df.groupby(['rewiring_prob', 'strategy', 'timing']).agg({
        'final_deaths': 'mean',
        'high_risk_deaths': 'mean'
    }).reset_index()

    # Plot 1: Strategy comparison heatmap for final deaths
    pivot_deaths = interaction_data.pivot_table(
        values='final_deaths',
        index=['strategy', 'timing'],
        columns='rewiring_prob'
    )

    sns.heatmap(pivot_deaths, annot=True, fmt='.1f', cmap='RdYlBu_r',
                ax=ax1, cbar_kws={'label': 'Final Deaths'})
    ax1.set_title('Final Deaths Heatmap')
    ax1.set_xlabel('Rewiring Probability')

    # Plot 2: High-risk deaths heatmap
    pivot_high_risk = interaction_data.pivot_table(
        values='high_risk_deaths',
        index=['strategy', 'timing'],
        columns='rewiring_prob'
    )

    sns.heatmap(pivot_high_risk, annot=True, fmt='.1f', cmap='RdYlBu_r',
                ax=ax2, cbar_kws={'label': 'High-Risk Deaths'})
    ax2.set_title('High-Risk Deaths Heatmap')
    ax2.set_xlabel('Rewiring Probability')

    plt.tight_layout()
    plt.savefig('figures/strategy_comparison_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table_figure(df):
    """Create a summary table figure"""
    # Calculate summary statistics
    summary = df.groupby(['strategy', 'timing']).agg({
        'final_deaths': 'mean',
        'high_risk_deaths': 'mean',
        'vaccines_effective': 'mean',
        'vaccines_wasted': 'mean',
        'epidemic_duration': 'mean'
    }).round(1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = []
    headers = ['Strategy-Timing', 'Total Deaths', 'High-Risk Deaths', 'Effective Vaccines', 'Wasted Vaccines',
               'Duration']

    for (strategy, timing), row in summary.iterrows():
        table_data.append([
            f'{strategy}-{timing}',
            f"{row['final_deaths']:.1f}",
            f"{row['high_risk_deaths']:.1f}",
            f"{row['vaccines_effective']:.1f}",
            f"{row['vaccines_wasted']:.1f}",
            f"{row['epidemic_duration']:.1f}"
        ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style header row (row 0)
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    for i in range(1, len(table_data) + 1):  # Start from 1, go to len(table_data) + 1
        color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    plt.title('Summary Statistics by Vaccination Strategy and Timing',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to generate all focused visualizations"""
    # Load data
    df = load_and_prepare_data()

    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)

    print("Creating focused visualizations...")

    # Generate visualizations
    plot_network_comparison(df)
    plot_strategy_comparison_bars(df)
    plot_strategy_comparison_heatmaps(df)
    create_summary_table_figure(df)

    # Print the key statistics
    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)

    network_stats = create_network_comparison_table(df)
    print("\nNETWORK TYPE COMPARISON:")
    print(
        f"Small-world (p=0.3): {network_stats.loc[0.3, 'deaths_mean']:.1f}±{network_stats.loc[0.3, 'deaths_std']:.1f} deaths")
    print(
        f"Random (p=0.9):      {network_stats.loc[0.9, 'deaths_mean']:.1f}±{network_stats.loc[0.9, 'deaths_std']:.1f} deaths")

    interaction_stats = create_interaction_effects_table(df)
    print("\nVACCINATION STRATEGY EFFECTS:")
    for (rewiring_prob, strategy, timing), row in interaction_stats.iterrows():
        network_type = 'Small-world' if rewiring_prob == 0.3 else 'Random'
        print(
            f"{network_type} + {strategy} {timing}: {row['final_deaths']:.1f} total deaths, {row['high_risk_deaths']:.1f} high-risk deaths")

    print(f"\nVisualization files saved to 'figures/' directory")


if __name__ == "__main__":
    main()