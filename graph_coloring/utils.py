import networkx as nx
from typing import Dict
import random


def calculate_loss(graph: nx.Graph, coloring: Dict[int, int]) -> int:
    """Calculate loss as number of conflicts (adjacent vertices with same color)"""
    conflicts = 0
    # Two vertices
    for u, v in graph.edges():
        # If two connected vertices have the same color, conflics++
        if coloring[u] == coloring[v]:
            conflicts += 1
    return conflicts


def generate_random_coloring(graph: nx.Graph, num_colors: int) -> Dict[int, int]:
    """Generate random coloring using num_colors colors"""
    # Using random.randint(0, num_colors) would give possible num_colors + 1
    return {node: random.randint(0, num_colors - 1) for node in graph.nodes()}


def get_neighbor(
    graph: nx.Graph, coloring: Dict[int, int], num_colors: int
) -> Dict[int, int]:
    """Generate neighbor solution by changing color of random vertex"""
    new_coloring = coloring.copy()
    node = random.choice(list(graph.nodes()))
    available_colors = list(range(num_colors))
    # Remove the node's color from the color pool to not choose randomly some new from available colors
    available_colors.remove(coloring[node])
    new_coloring[node] = random.choice(available_colors)
    return new_coloring
