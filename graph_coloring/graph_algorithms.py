from .coloring_utils import generate_random_coloring, is_valid_coloring, calculate_loss, count_distinct_colors, optimize_coloring
import random
import networkx as nx
from typing import Tuple, Dict
from itertools import product

from typing import Dict, Tuple
import networkx as nx
from .coloring_utils import calculate_loss, generate_random_coloring, is_valid_coloring, optimize_coloring

def sampling_coloring(graph: nx.Graph, n_samples: int) -> Tuple[Dict[int, int], float, int]:
    """Sampling-based coloring algorithm returning the best found coloring"""
    n_nodes = len(graph.nodes())
    initial_colors = max(2, n_nodes // 2)  # Heurystyczne założenie, min. 2 kolory
    best_coloring = None
    min_loss = float("inf")
    attempts = 0

    for _ in range(n_samples):
        attempts += 1
        coloring = generate_random_coloring(graph, initial_colors)
        loss = calculate_loss(graph, coloring)

        if loss < min_loss:
            min_loss = loss
            best_coloring = coloring

        if is_valid_coloring(graph, coloring):
            optimized_coloring = optimize_coloring(graph, coloring)
            optimized_loss = calculate_loss(graph, optimized_coloring)
            if optimized_loss < min_loss:
                min_loss = optimized_loss
                best_coloring = optimized_coloring

    if min_loss > len(best_coloring):
        print("No valid coloring found, returning best solution!")
    else:
        print("Valid coloring found!")
    return best_coloring, min_loss, attempts

# Algorytm pełnego przeglądu, Determnistyczny
def brute_force_coloring(graph: nx.Graph, max_colors: int) -> Tuple[Dict[int, int], float, int]:
    """Brute force search to find the best valid coloring of a graph"""
    best_coloring = None
    best_loss = float("inf")  
    attempts = 0

    for num_colors in range(1, max_colors + 1):
        all_colorings = generate_all_colorings(graph, num_colors)
        for coloring in all_colorings:
            if not is_valid_coloring(graph, coloring):
                continue
            
            loss = calculate_loss(graph, coloring, conflict_weight=1, color_weight=0)
            attempts += 1
            
            if loss < best_loss:
                best_loss = loss
                best_coloring = coloring

            if loss == 0:  # Perfect coloring (no conflicts)
                print("Valid coloring found!")
                return best_coloring, best_loss, attempts

        if best_coloring is not None:
            break
    
    print("No valid coloring found, returning best solution!")
    return best_coloring if best_coloring is not None else None, best_loss, attempts


def generate_all_colorings(graph: nx.Graph, num_colors: int):
    """Generate all possible colorings for the graph with given number of colors"""
    nodes = list(graph.nodes())
    color_options = range(num_colors)

    all_colorings = []
    for coloring_tuple in product(color_options, repeat=len(nodes)):
        coloring = {node: color for node, color in zip(nodes, coloring_tuple)}
        all_colorings.append(coloring)
    
    return all_colorings