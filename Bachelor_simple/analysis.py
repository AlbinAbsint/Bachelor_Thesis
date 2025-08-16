import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


class EpidemicAnalysis:
    """Statistical analysis for epidemic simulation results"""

    def __init__(self, data_path: str):
        """Initialize with data from CSV file"""
        self.data = pd.read_csv(data_path)
        self.outcomes = ['final_deaths', 'epidemic_duration', 'high_risk_deaths']

    def compare_strategies(self, outcome='final_deaths'):
        """Compare vaccination strategies using Mann-Whitney U test"""
        random_data = self.data[self.data['strategy'] == 'random'][outcome]
        targeted_data = self.data[self.data['strategy'] == 'targeted'][outcome]

        statistic, p_value = stats.mannwhitneyu(random_data, targeted_data, alternative='two-sided')

        print(f"\nStrategy Comparison ({outcome}):")
        print(f"Mann-Whitney U statistic: {statistic}")
        print(f"p-value: {p_value:.6f}")
        print(f"Random median: {random_data.median():.2f}")
        print(f"Targeted median: {targeted_data.median():.2f}")

        if p_value < 0.05:
            print("Significant difference between strategies")
        else:
            print("No significant difference between strategies")

        return statistic, p_value

    def compare_rewiring_probs(self, outcome='final_deaths'):
        """Compare rewiring probabilities using Kruskal-Wallis test"""
        groups = []
        labels = []

        for prob in sorted(self.data['rewiring_prob'].unique()):
            group_data = self.data[self.data['rewiring_prob'] == prob][outcome]
            groups.append(group_data)
            labels.append(f"p={prob}")

        statistic, p_value = stats.kruskal(*groups)

        print(f"\nRewiring Probability Comparison ({outcome}):")
        print(f"Kruskal-Wallis statistic: {statistic}")
        print(f"p-value: {p_value:.6f}")

        # Print medians for each group
        for i, label in enumerate(labels):
            print(f"{label} median: {groups[i].median():.2f}")

        if p_value < 0.05:
            print("Significant difference between rewiring probabilities")
            self._post_hoc_analysis(groups, labels)
        else:
            print("No significant difference between rewiring probabilities")

        return statistic, p_value

    @staticmethod
    def _post_hoc_analysis(groups, labels):
        """Perform post-hoc pairwise comparisons with Bonferroni correction"""
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        alpha_corrected = 0.05 / n_comparisons

        print(f"\nPost-hoc pairwise comparisons (Bonferroni corrected α = {alpha_corrected:.4f}):")

        for i, j in combinations(range(len(groups)), 2):
            stat, p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            significance = '*' if p < alpha_corrected else ''
            print(f"{labels[i]} vs {labels[j]}: p = {p:.6f} {significance}")

    @staticmethod
    def calculate_effect_size(group1, group2):
        """Calculate rank-biserial correlation as effect size"""
        n1, n2 = len(group1), len(group2)
        u_statistic, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        # Rank-biserial correlation
        r = 1 - (2 * u_statistic) / (n1 * n2)

        # Interpretation
        if abs(r) < 0.1:
            magnitude = "negligible"
        elif abs(r) < 0.3:
            magnitude = "small"
        elif abs(r) < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"

        return r, magnitude

    def comprehensive_analysis(self):
        """Run complete non-parametric analysis for all outcomes"""
        for outcome in self.outcomes:
            print(f"\n{'=' * 60}")
            print(f"ANALYSIS FOR: {outcome.upper()}")
            print(f"{'=' * 60}")

            # Strategy comparison
            self.compare_strategies(outcome)

            # Calculate effect size for strategy comparison
            random_data = self.data[self.data['strategy'] == 'random'][outcome]
            targeted_data = self.data[self.data['strategy'] == 'targeted'][outcome]
            effect_size, magnitude = self.calculate_effect_size(random_data, targeted_data)
            print(f"Effect size (rank-biserial correlation): {effect_size:.3f} ({magnitude})")

            # Rewiring probability comparison
            self.compare_rewiring_probs(outcome)

    def create_visualizations(self, save_plots=False):
        """Create box plots for non-parametric data visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for i, outcome in enumerate(self.outcomes):
            # Box plots by strategy
            sns.boxplot(data=self.data, x='strategy', y=outcome, ax=axes[0, i])
            axes[0, i].set_title(f'{outcome.replace("_", " ").title()} by Strategy')

            # Box plots by rewiring probability
            sns.boxplot(data=self.data, x='rewiring_prob', y=outcome, ax=axes[1, i])
            axes[1, i].set_title(f'{outcome.replace("_", " ").title()} by Rewiring Probability')
            axes[1, i].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_plots:
            plt.savefig('plots/epidemic_analysis_plots.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'epidemic_analysis_plots.png'")

        plt.show()

    def summary_statistics(self):
        """Generate summary statistics by groups"""
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        # Group by key variables including coverage_rate
        summary = self.data.groupby(['rewiring_prob', 'strategy', 'coverage_rate'])[self.outcomes].agg(
            ['median', 'mean', 'std']).round(2)
        print(summary)


def main():
    """Main execution function for epidemic analysis"""
    try:
        # Initialize analysis
        analysis = EpidemicAnalysis('data/epidemic_simulation_results.csv')

        print("Starting comprehensive epidemic simulation analysis...")

        # Generate summary statistics
        analysis.summary_statistics()

        # Run comprehensive statistical analysis
        analysis.comprehensive_analysis()

        # Create visualizations
        analysis.create_visualizations(save_plots=True)

        print("\nAnalysis complete!")

    except FileNotFoundError:
        print("Error: Could not find 'data/epidemic_simulation_results.csv'")
        print("Make sure to run the simulation first using main.py")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    main()
