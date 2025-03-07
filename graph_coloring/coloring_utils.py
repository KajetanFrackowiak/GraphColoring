import networkx as nx
import random
from typing import Dict

def count_distinct_colors(coloring: Dict[int, int]) -> int:
    """Calculate the number of distinct colors used in the coloring"""
    return len(set(coloring.values()))

def calculate_loss(graph: nx.Graph, coloring: Dict[int, int], conflict_weight: float = 10.0, color_weight: float = 1.0) -> float:
    """Loss function: penalizes conflicts and number of colors"""
    conflicts = 0
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            conflicts += 1
    num_colors = count_distinct_colors(coloring)
    return conflict_weight * conflicts + color_weight * num_colors

def is_valid_coloring(graph: nx.Graph, coloring: Dict[int, int]) -> bool:
    """Check if coloring is valid (no adjacent vertices have same color)"""
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False
    return True

def generate_random_coloring(graph: nx.Graph, num_colors: int) -> Dict[int, int]:
    """Generate a random coloring with exactly num_colors used"""
    nodes = list(graph.nodes())
    if num_colors > len(nodes):
        num_colors = len(nodes)  # Nie więcej kolorów niż wierzchołków
    
    # Przydziel losowe kolory, upewniając się, że wszystkie num_colors są użyte
    coloring = {}
    available_colors = list(range(num_colors))
    # Najpierw przypisz każdy kolor przynajmniej raz
    random.shuffle(nodes)
    for i in range(min(num_colors, len(nodes))):
        coloring[nodes[i]] = available_colors[i]
    # Pozostałe wierzchołki losuj z dostępnych kolorów
    for node in nodes[num_colors:]:
        coloring[node] = random.choice(available_colors)
    return coloring

def optimize_coloring(graph: nx.Graph, coloring: Dict[int, int], max_attempts: int = 100) -> Dict[int, int]:
    """Optimize coloring by reducing the number of colors"""
    current_coloring = coloring.copy()
    current_num_colors = count_distinct_colors(current_coloring)
    
    for _ in range(max_attempts):
        if not is_valid_coloring(graph, current_coloring):
            break
        for node in graph.nodes():
            current_color = current_coloring[node]
            for new_color in range(current_num_colors):
                if new_color != current_color:
                    current_coloring[node] = new_color
                    if is_valid_coloring(graph, current_coloring):
                        if count_distinct_colors(current_coloring) < current_num_colors:
                            current_num_colors = count_distinct_colors(current_coloring)
                        break
                    else:
                        current_coloring[node] = current_color
    return current_coloring

def get_neighbor(graph: nx.Graph, coloring: Dict[int, int], num_colors: int) -> Dict[int, int]:
    """Generate a neighboring solution with the same number of colors"""
    new_coloring = coloring.copy()
    node_to_change = random.choice(list(graph.nodes()))
    current_color = new_coloring[node_to_change]
    new_color = random.choice([c for c in range(num_colors) if c != current_color])
    new_coloring[node_to_change] = new_color
    return new_coloring