from graph_coloring.coloring_utils import generate_random_coloring, count_distinct_colors
from graph_coloring.graph_algorithms import sampling_coloring, brute_force_coloring
from graph_coloring.visualization import generate_chart
import networkx as nx
import yaml
import argparse

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose algorithm and parameters")
    parser.add_argument("--algorithm", type=int, help="1: Sampling, 2:Brute Force", required=True)
    parser.add_argument("--nodes", type=int, help="Number of nodes", default=10)
    parser.add_argument("--edge_prob", type=float, help="Edge probability", default=0.8)
    parser.add_argument("--samples", type=int, help="Number of samples", default=100)
    parser.add_argument("--max_colors", type=int, help="Max number of colors", default=5)
    args = parser.parse_args()


    G = nx.erdos_renyi_graph(args.nodes, args.edge_prob)
    
    if args.algorithm == 1:
        coloring_result, min_loss, attempts = sampling_coloring(G, args.n_samples) 
    elif args.algorithm == 2:
        coloring_result, min_loss, attempts = brute_force_coloring(G, max_colors=args.max_colors)
    else:
        raise ValueError("Only algorithm 1 (sampling) is implemented for this heuristic")

    print(f"Attempts {attempts}")
    if coloring_result is not None:
        
        print(f"Number of colors after algorithm: {count_distinct_colors(coloring_result)}")
    else:
        print("No valid coloring found, so no colors to count")
    print(f"Minimum loss (conflicts + color penalty): {min_loss}")

    if coloring_result is not None:
        generate_chart(G, coloring_result)
    else:
        print("No valid coloring found, so cannot generate chart")