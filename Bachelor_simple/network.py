# network.py
import numpy as np
import networkx as nx
from typing import Dict, Set, List


class WattsStrogatzNetwork:
    """Manages Watts-Strogatz network topology and node attributes"""

    def __init__(self, n_nodes: int, k_neighbors: int, rewiring_prob: float, seed: int):
        if k_neighbors % 2 != 0:
            raise ValueError("k_neighbors must be even")
        if not 0 <= rewiring_prob <= 1:
            raise ValueError("rewiring_prob must be between 0 and 1")
        self.n_nodes = n_nodes
        self.k_neighbors = k_neighbors
        self.rewiring_prob = rewiring_prob
        self.seed = seed

        # Generate network
        np.random.seed(seed)
        self.graph = nx.watts_strogatz_graph(n_nodes, k_neighbors, rewiring_prob, seed)

        # Initialize node attributes
        self.risk_groups = {}  # node_id -> 'high' or 'low'

    def assign_risk_groups(self, high_risk_fraction: float, seed: int):
        """Randomly assign nodes to high/low risk groups"""
        np.random.seed(seed)
        n_high_risk = int(self.n_nodes * high_risk_fraction)

        high_risk_nodes = np.random.choice(
            self.n_nodes,
            size=n_high_risk,
            replace=False
        )

        for node in range(self.n_nodes):
            self.risk_groups[node] = 'high' if node in high_risk_nodes else 'low'

    def get_high_risk_nodes(self) -> Set[int]:
        """Return set of high-risk node IDs"""
        return {node for node, risk in self.risk_groups.items() if risk == 'high'}

    def get_low_risk_nodes(self) -> Set[int]:
        """Return set of low-risk node IDs"""
        return {node for node, risk in self.risk_groups.items() if risk == 'low'}

    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node"""
        return list(self.graph.neighbors(node))

    def get_risk_level(self, node: int) -> str:
        """Get risk level of node"""
        return self.risk_groups.get(node, 'low')

    def get_network_stats(self) -> Dict:
        """Calculate network statistics"""
        return {
            'clustering_coefficient': nx.average_clustering(self.graph),
            'average_path_length': nx.average_shortest_path_length(self.graph),
            'degree_distribution': dict(self.graph.degree())
        }
