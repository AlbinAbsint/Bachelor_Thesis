# simulation.py
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple
from network import WattsStrogatzNetwork
from epidemic_model import SIRVDModel
from config import SimulationConfig


class EpidemicSimulation:
    """Main simulation runner for SIRVD epidemic model"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = []

    def _create_network(self, rewiring_prob: float, seed: int) -> WattsStrogatzNetwork:
        """Create and configure network with risk groups"""
        network = WattsStrogatzNetwork(
            n_nodes=self.config.network.n_nodes,
            k_neighbors=self.config.network.k_neighbors,
            rewiring_prob=rewiring_prob,
            seed=seed
        )

        network.assign_risk_groups(
            high_risk_fraction=self.config.population.high_risk_fraction,
            seed=seed + 1
        )
        return network

    def _create_model(self, network: WattsStrogatzNetwork, seed: int) -> SIRVDModel:
        """Create epidemic model with configuration parameters"""
        return SIRVDModel(
            network=network,
            transmission_rate=self.config.epidemic.transmission_rate,
            recovery_rate=self.config.epidemic.recovery_rate,
            high_risk_mortality=self.config.population.high_risk_mortality,
            low_risk_mortality=self.config.population.low_risk_mortality,
            seed=seed
        )

    @staticmethod
    def _apply_vaccination_if_needed(model: SIRVDModel, vacc_params: Dict,
                                     timing: str, seed: int) -> Tuple[bool, int, int]:
        """Apply vaccination based on timing and return whether it was applied plus dose counts"""
        if vacc_params['timing'] == timing:
            effective, wasted = model.apply_vaccination(
                strategy=vacc_params['strategy'],
                coverage_rate=vacc_params['coverage_rate'],
                seed=seed
            )
            return True, effective, wasted
        return False, 0, 0

    def _check_early_vaccination_threshold(self, model: SIRVDModel) -> bool:
        """Check if early vaccination threshold is reached"""
        threshold = int(self.config.vaccination.early_threshold * self.config.network.n_nodes)
        return len(model.states['I']) >= threshold

    def _run_epidemic_loop(self, model: SIRVDModel, vacc_params: Dict, seed: int) -> Tuple[int, int]:
        """Run the main epidemic simulation loop and return vaccination metrics"""
        early_vaccination_applied = False
        total_effective = 0
        total_wasted = 0

        while not model.is_epidemic_over():
            model.step()

            # Apply early vaccination if conditions are met
            if (vacc_params['timing'] == 'early' and
                    not early_vaccination_applied and
                    self._check_early_vaccination_threshold(model)):
                applied, effective, wasted = self._apply_vaccination_if_needed(model, vacc_params, 'early', seed)
                if applied:
                    total_effective += effective
                    total_wasted += wasted
                    early_vaccination_applied = True

        return total_effective, total_wasted

    @staticmethod
    def _calculate_high_risk_outcomes(model: SIRVDModel,
                                      network: WattsStrogatzNetwork) -> Tuple[int, int]:
        """Calculate high-risk specific deaths and vaccinations"""
        high_risk_nodes = network.get_high_risk_nodes()

        high_risk_deaths = len([node for node in model.states['D']
                                if node in high_risk_nodes])
        high_risk_vaccinated = len([node for node in model.states['V']
                                    if node in high_risk_nodes])

        return high_risk_deaths, high_risk_vaccinated

    @staticmethod
    def _build_result_dict(run_id: int, network_params: Dict, vacc_params: Dict,
                           final_state: Dict, high_risk_deaths: int,
                           high_risk_vaccinated: int, duration: int,
                           vaccines_effective: int, vaccines_wasted: int,
                           total_infected: int, peak_infected: int) -> Dict:
        """Build result dictionary from simulation outcomes"""
        return {
            'run_id': run_id,
            'rewiring_prob': network_params['rewiring_prob'],
            'strategy': vacc_params['strategy'],
            'timing': vacc_params['timing'],
            'coverage_rate': vacc_params['coverage_rate'],
            'final_deaths': final_state['D'],
            'final_recovered': final_state['R'],
            'final_vaccinated': final_state['V'],
            'peak_infected': peak_infected,
            'total_infected': total_infected,
            'high_risk_deaths': high_risk_deaths,
            'high_risk_vaccinated': high_risk_vaccinated,
            'epidemic_duration': duration,
            'vaccines_effective': vaccines_effective,
            'vaccines_wasted': vaccines_wasted
        }

    def run_single_simulation(self, network_params: Dict, vacc_params: Dict,
                              run_id: int, seed: int) -> Dict:
        """Run a single epidemic simulation"""
        # Create network and model
        network = self._create_network(network_params['rewiring_prob'], seed)
        model = self._create_model(network, seed + 2)

        # Track vaccination metrics
        total_effective = 0
        total_wasted = 0

        # Apply start vaccination if needed
        applied, effective, wasted = self._apply_vaccination_if_needed(model, vacc_params, 'start', seed + 3)
        if applied:
            total_effective += effective
            total_wasted += wasted

        # Seed infection
        model.seed_infection(n_infected=1, seed=seed + 4)

        # Run epidemic simulation
        early_effective, early_wasted = self._run_epidemic_loop(model, vacc_params, seed + 5)
        total_effective += early_effective
        total_wasted += early_wasted

        # Run final validation
        model.run_final_validation()

        # Collect and return results
        final_state = model.get_current_state()
        high_risk_deaths, high_risk_vaccinated = self._calculate_high_risk_outcomes(model, network)
        total_infected = model.get_cumulative_infections()
        peak_infected = model.get_peak_infected()  # Extract peak infected

        return self._build_result_dict(
            run_id, network_params, vacc_params, final_state,
            high_risk_deaths, high_risk_vaccinated, model.timestep,
            total_effective, total_wasted, total_infected, peak_infected
        )

    def _generate_parameter_combinations(self) -> List[Tuple]:
        """Generate all parameter combinations for the experiment"""
        return list(product(
            self.config.network.rewiring_probs,
            self.config.vaccination.strategies,
            self.config.vaccination.timings,
            self.config.vaccination.coverage_rates
        ))

    def _print_progress(self, combinations: List, rewiring_prob: float,
                        strategy: str, timing: str, coverage_rate: float) -> None:
        """Print simulation progress information"""
        if not hasattr(self, '_total_printed'):
            total_sims = len(combinations) * self.config.n_replicates
            print(
                f"Running {len(combinations)} combinations × {self.config.n_replicates} replicates = {total_sims} simulations")
            self._total_printed = True

        print(f"Running: p={rewiring_prob}, {strategy} {timing} vaccination, {coverage_rate * 100:.0f}% coverage")

    def run_all_simulations(self) -> pd.DataFrame:
        """Run all experimental combinations"""
        combinations = self._generate_parameter_combinations()
        run_id = 0

        for rewiring_prob, strategy, timing, coverage_rate in combinations:
            network_params = {'rewiring_prob': rewiring_prob}
            vacc_params = {
                'strategy': strategy,
                'timing': timing,
                'coverage_rate': coverage_rate
            }

            self._print_progress(combinations, rewiring_prob, strategy, timing, coverage_rate)

            # Run replicates for this combination
            for replicate in range(self.config.n_replicates):
                seed = run_id * 1000 + replicate

                result = self.run_single_simulation(network_params, vacc_params, run_id, seed)
                result['replicate'] = replicate
                self.results.append(result)

                run_id += 1

        return pd.DataFrame(self.results)
