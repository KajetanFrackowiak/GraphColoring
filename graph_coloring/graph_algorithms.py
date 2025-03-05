from .coloring_utils import generate_random_coloring, calculate_loss, is_valid_coloring
import networkx as nx
from typing import Tuple, Dict


def sampling_coloring(
    graph: nx.Graph, n_samples: int, max_colors: int
) -> Tuple[Dict[int, int], int]:
    """Sampling-based coloring algorithm"""
    best_coloring: Dict[int, int] = None
    min_loss: int = float("inf")
    attempts = 0

    for _ in range(n_samples):
        attempts += 1
        coloring = generate_random_coloring(graph, max_colors)
        loss = calculate_loss(graph, coloring)

        if loss < min_loss:
            min_loss = loss
            best_coloring = coloring

        if loss == 0 and is_valid_coloring(graph, coloring):
            return best_coloring, min_loss, attempts

    print("Not found!")
    return best_coloring, min_loss, attempts
