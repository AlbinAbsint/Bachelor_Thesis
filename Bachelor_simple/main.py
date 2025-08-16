from simulation import EpidemicSimulation, SimulationConfig


def main():
    """Main execution function"""
    # Load configuration
    config = SimulationConfig()

    # Create and run simulation
    simulation = EpidemicSimulation(config)
    results_df = simulation.run_all_simulations()

    # Save results
    results_df.to_csv('data/epidemic_simulation_results.csv', index=False)
    print(f"\nSimulation complete! Results saved to 'data/epidemic_simulation_results.csv'")
    print(f"Total simulations run: {len(results_df)}")

    # Complete summary (not truncated)
    print("\nComplete summary:")
    summary = results_df.groupby(['rewiring_prob', 'strategy', 'timing', 'coverage_rate'])['final_deaths'].mean()
    for index, value in summary.items():
        print(f"{index}: {value:.2f}")


if __name__ == "__main__":
    main()
