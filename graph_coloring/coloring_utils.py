import networkx as nx
import random
from typing import Dict


def count_distinct_colors(coloring: Dict[int, int]) -> int:
    """Calculate the number of distinct colors used in the coloring"""
    return len(set(coloring.values()))


def calculate_loss(graph: nx.Graph, coloring: Dict[int, int]) -> int:
    """Loss function: counts number of adjacent vertices with same color"""
    conflicts = 0
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            conflicts += 1
    return conflicts


def is_valid_coloring(graph: nx.Graph, coloring: Dict[int, int]) -> bool:
    """Check if coloring is valid (no adjacent vertices have same color)"""
    return calculate_loss(graph, coloring) == 0


def generate_random_coloring(graph: nx.Graph, max_colors: int) -> Dict[int, int]:
    """Generate a random coloring"""
    nodes = list(graph.nodes())
    coloring = {node: random.randint(0, max_colors - 1) for node in nodes}
    return coloring
