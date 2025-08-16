# epidemic_model.py
import numpy as np
from typing import Dict, Set, List, Tuple
from network import WattsStrogatzNetwork


class SIRVDModel:
    """SIRVD epidemic model where vaccination creates immunity"""

    def __init__(self, network: WattsStrogatzNetwork, transmission_rate: float,
                 recovery_rate: float, high_risk_mortality: float,
                 low_risk_mortality: float, seed: int):
        self.network = network
        self.transmission_rate = transmission_rate
        self.recovery_rate = recovery_rate
        self.high_risk_mortality = high_risk_mortality
        self.low_risk_mortality = low_risk_mortality
        self.timestep = 0
        self.peak_infected = 0

        # Set random seed
        np.random.seed(seed)

        # Initialize all nodes as susceptible
        self.states = {
            'S': set(range(network.n_nodes)),  # Susceptible
            'I': set(),  # Infected
            'R': set(),  # Recovered
            'V': set(),  # Vaccinated (immune)
            'D': set()  # Dead
        }

        # Track cumulative infections
        self.total_ever_infected = set()

        # Track state history for analysis
        self.history = []
        self._record_state()

    def seed_infection(self, n_infected: int = 1, seed: int = None):
        """Randomly infect initial nodes"""
        if seed is not None:
            np.random.seed(seed)

        n_infected = min(n_infected, len(self.states['S']))

        if n_infected > 0:
            initial_infected = np.random.choice(
                list(self.states['S']),
                size=n_infected,
                replace=False
            )

            for node in initial_infected:
                self._change_state(node, 'S', 'I')
                self.total_ever_infected.add(node)  # Track initial infections

    def step(self):
        """Execute one time step of the SIRVD model"""
        self.timestep += 1

        new_infections = self._process_transmission()
        recoveries, deaths = self._process_recovery_and_death()

        self._apply_state_changes(new_infections, recoveries, deaths)
        self._record_state()

        return len(new_infections), len(recoveries), len(deaths)

    def _process_transmission(self) -> Set[int]:
        """Process transmission events S -> I"""
        new_infections = set()

        for infected_node in self.states['I']:
            neighbors = self.network.get_neighbors(infected_node)

            for neighbor in neighbors:
                if neighbor in self.states['S']:
                    if np.random.random() < self.transmission_rate:
                        new_infections.add(neighbor)

        return new_infections

    def _process_recovery_and_death(self) -> Tuple[Set[int], Set[int]]:
        """Process recovery and death events I -> R or I -> D"""
        recoveries = set()
        deaths = set()

        for infected_node in self.states['I']:
            mortality_rate = self._get_mortality_rate(infected_node)

            if np.random.random() < mortality_rate:
                deaths.add(infected_node)
            elif np.random.random() < self.recovery_rate:
                recoveries.add(infected_node)

        return recoveries, deaths

    def _get_mortality_rate(self, node: int) -> float:
        """Get mortality rate for a node based on risk level"""
        risk_level = self.network.get_risk_level(node)
        return (self.high_risk_mortality if risk_level == 'high'
                else self.low_risk_mortality)

    def _apply_state_changes(self, new_infections: Set[int], recoveries: Set[int], deaths: Set[int]):
        """Apply all state changes for this timestep"""
        for node in new_infections:
            self._change_state(node, 'S', 'I')
            self.total_ever_infected.add(node)  # Track cumulative infections

        for node in recoveries:
            self._change_state(node, 'I', 'R')

        for node in deaths:
            self._change_state(node, 'I', 'D')

    def apply_vaccination(self, strategy: str, coverage_rate: float, seed: int = None):
        """Apply vaccination to population - S/I/R nodes can receive it, only susceptible nodes benefit"""
        if seed is not None:
            np.random.seed(seed)

        # Get all alive nodes (S, I, R - excluding D and V)
        alive_nodes = list(self.states['S'] | self.states['I'] | self.states['R'])

        if not alive_nodes:
            return 0, 0  # Return (effective, wasted) doses

        # Calculate total vaccines to administer based on total alive population
        n_to_vaccinate = int(len(alive_nodes) * coverage_rate)
        n_to_vaccinate = min(n_to_vaccinate, len(alive_nodes))

        if n_to_vaccinate == 0:
            return 0, 0

        # Select nodes based on strategy
        if strategy == 'random':
            nodes_to_vaccinate = np.random.choice(
                alive_nodes,
                size=n_to_vaccinate,
                replace=False
            )
        elif strategy == 'targeted':
            # Target high-risk individuals from ALL alive nodes (not just susceptible)
            high_risk_nodes = self.network.get_high_risk_nodes()
            high_risk_alive = [node for node in alive_nodes if node in high_risk_nodes]

            if len(high_risk_alive) >= n_to_vaccinate:
                # Enough high-risk nodes available
                nodes_to_vaccinate = np.random.choice(
                    high_risk_alive,
                    size=n_to_vaccinate,
                    replace=False
                )
            else:
                # Not enough high-risk nodes, vaccinate all high-risk + some low-risk
                remaining_vaccines = n_to_vaccinate - len(high_risk_alive)
                low_risk_alive = [node for node in alive_nodes if node not in high_risk_nodes]

                if remaining_vaccines > 0 and low_risk_alive:
                    additional_nodes = np.random.choice(
                        low_risk_alive,
                        size=min(remaining_vaccines, len(low_risk_alive)),
                        replace=False
                    )
                    nodes_to_vaccinate = np.concatenate([high_risk_alive, additional_nodes])
                else:
                    nodes_to_vaccinate = high_risk_alive
        else:
            raise ValueError(f"Unknown vaccination strategy: {strategy}")

        # Apply vaccination - only S nodes benefit, track waste
        vaccines_effective = 0
        vaccines_wasted = 0

        for node in nodes_to_vaccinate:
            if node in self.states['S']:
                self._change_state(node, 'S', 'V')
                vaccines_effective += 1
            else:
                vaccines_wasted += 1  # Node was I or R - vaccine wasted

        return vaccines_effective, vaccines_wasted

    def _change_state(self, node: int, from_state: str, to_state: str):
        """Internal method to change node state"""
        self.states[from_state].discard(node)
        self.states[to_state].add(node)

    def validate_states(self) -> bool:
        """Validate that states are consistent and don't exceed population"""
        total_nodes = self.network.n_nodes
        is_valid = True

        # Check 1: No node appears in multiple states
        all_state_nodes = set()
        for state_name, nodes in self.states.items():
            for node in nodes:
                if node in all_state_nodes:
                    print(f"ERROR: Node {node} appears in multiple states")
                    is_valid = False
                all_state_nodes.add(node)

        # Check 2: Total nodes in all states doesn't exceed population
        total_in_states = sum(len(nodes) for nodes in self.states.values())
        if total_in_states > total_nodes:
            print(f"ERROR: Total nodes in states ({total_in_states}) exceeds population ({total_nodes})")
            is_valid = False

        # Check 3: All nodes are accounted for
        if total_in_states != total_nodes:
            missing = total_nodes - total_in_states
            print(f"ERROR: {missing} nodes are not in any state")
            is_valid = False

        # Check 4: Total ever infected doesn't exceed population
        if len(self.total_ever_infected) > total_nodes:
            print(f"ERROR: Total ever infected ({len(self.total_ever_infected)}) exceeds population ({total_nodes})")
            is_valid = False

        # Check 5: All infected nodes are valid node IDs
        for node in self.total_ever_infected:
            if node < 0 or node >= total_nodes:
                print(f"ERROR: Invalid node ID {node} in infection tracker")
                is_valid = False

        # Check 6: All nodes in infection tracker should be in I, R, or D states
        expected_infected_states = self.states['I'] | self.states['R'] | self.states['D']
        if not self.total_ever_infected.issubset(expected_infected_states):
            orphaned = self.total_ever_infected - expected_infected_states
            print(f"ERROR: {len(orphaned)} nodes in infection tracker are not in I/R/D states")
            is_valid = False

        return is_valid

    def _record_state(self):
        """Record current state for history tracking"""
        current_infected = len(self.states['I'])
        self.peak_infected = max(self.peak_infected, current_infected)

        self.history.append({
            'timestep': self.timestep,
            'S': len(self.states['S']),
            'I': current_infected,
            'R': len(self.states['R']),
            'V': len(self.states['V']),
            'D': len(self.states['D'])
        })

    def get_peak_infected(self) -> int:
        """Get the maximum number of simultaneously infected individuals"""
        return self.peak_infected

    def get_current_state(self) -> Dict[str, int]:
        """Get current state counts"""
        return {
            'S': len(self.states['S']),
            'I': len(self.states['I']),
            'R': len(self.states['R']),
            'V': len(self.states['V']),
            'D': len(self.states['D']),
            'timestep': self.timestep
        }

    def is_epidemic_over(self) -> bool:
        """Check if epidemic has ended (no more infected)"""
        return len(self.states['I']) == 0

    def get_final_outbreak_size(self) -> float:
        """Get final attack rate (fraction who were infected)"""
        final_susceptible = len(self.states['S'])
        return 1.0 - (final_susceptible / self.network.n_nodes)

    def get_susceptible_nodes(self) -> Set[int]:
        """Get current susceptible nodes (for vaccination targeting)"""
        return self.states['S'].copy()

    def get_cumulative_infections(self) -> int:
        """Get total number of nodes that have ever been infected"""
        return len(self.total_ever_infected)

    def get_infection_rate(self) -> float:
        """Get the proportion of population that has ever been infected"""
        return len(self.total_ever_infected) / self.network.n_nodes

    def run_final_validation(self) -> bool:
        """Run comprehensive validation at the end of simulation"""
        is_valid = self.validate_states()
        return is_valid
