# config.py
from dataclasses import dataclass
from typing import List


@dataclass
class NetworkConfig:
    """Network topology parameters"""
    n_nodes: int = 1000
    k_neighbors: int = 6  # Initial degree (keep constant for fair comparison)
    rewiring_probs: List[float] = None

    def __post_init__(self):
        if self.rewiring_probs is None:
            self.rewiring_probs = [0.3]  # Rewiring probabilities for network topology


@dataclass
class PopulationConfig:
    """Population and risk group parameters"""
    high_risk_fraction: float = 0.35
    high_risk_mortality: float = 0.05
    low_risk_mortality: float = 0.01


@dataclass
class VaccinationConfig:
    """Vaccination parameters"""
    coverage_rates: List[float] = None
    strategies: List[str] = None
    timings: List[str] = None
    early_threshold: float = 0.1  # Threshold for early vaccination as a fraction of population

    def __post_init__(self):
        if self.coverage_rates is None:
            self.coverage_rates = [0.15, 0.20, 0.25]  # Vaccination coverage rates
        if self.strategies is None:
            self.strategies = ['random', 'targeted']  # Vaccination strategies
        if self.timings is None:
            self.timings = ['start', 'early']  # Vaccination timing options


@dataclass
class EpidemicConfig:
    """SIRVD model parameters"""
    transmission_rate: float = 0.3  # Probability of transmission per contact
    recovery_rate: float = 0.15  # Probability of recovery per infected individual


@dataclass
class SimulationConfig:
    """Overall simulation parameters"""
    n_replicates: int = 500  # Number of simulation runs
    network: NetworkConfig = None
    population: PopulationConfig = None
    vaccination: VaccinationConfig = None
    epidemic: EpidemicConfig = None

    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.population is None:
            self.population = PopulationConfig()
        if self.vaccination is None:
            self.vaccination = VaccinationConfig()
        if self.epidemic is None:
            self.epidemic = EpidemicConfig()
