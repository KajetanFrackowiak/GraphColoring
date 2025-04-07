import networkx as nx
from typing import Dict
import random


def calculate_loss(graph: nx.Graph, coloring: Dict[int, int]) -> int:
    """Calculate loss as number of conflicts (adjacent vertices with same color)"""
    conflicts = 0
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            conflicts += 1
    return conflicts


def generate_random_coloring(graph: nx.Graph, num_colors: int) -> Dict[int, int]:
    """Generate random coloring using num_colors colors"""
    return {node: random.randint(0, num_colors - 1) for node in graph.nodes()}


def get_neighbor(
    graph: nx.Graph, coloring: Dict[int, int], num_colors: int
) -> Dict[int, int]:
    """Generate neighbor solution by changing color of random vertex"""
    new_coloring = coloring.copy()
    node = random.choice(list(graph.nodes()))
    available_colors = list(range(num_colors))
    # Remove the node's color from the color pool to not choose again the same
    available_colors.remove(coloring[node])
    new_coloring[node] = random.choice(available_colors)
    return new_coloring
