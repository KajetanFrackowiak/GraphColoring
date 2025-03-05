from graph_coloring.coloring_utils import (
    generate_random_coloring,
    count_distinct_colors,
)
from graph_coloring.graph_algorithms import sampling_coloring
from graph_coloring.visualization import generate_chart

import networkx as nx
import yaml
import argparse

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write number of your algorithm and if you want change parameters to it"
    )
    parser.add_argument(
        "--algorithm", type=int, help="Then number of the algorithm", required=True
    )
    parser.add_argument("--nodes", type=int, help="Number of initial nodes")
    parser.add_argument("--edge_prob", type=float, help="Probability of each edge")
    parser.add_argument("--samples", type=int, help="Number of samples")
    parser.add_argument("--max_colors", type=int, help="Maximum amount of colors")
    args = parser.parse_args()

    n_nodes = args.nodes if args.nodes else config["build_graph"]["n_nodes"]
    edge_probability = (
        args.edge_prob if args.edge_prob else config["build_graph"]["edge_probability"]
    )
    max_colors = config["build_graph"]["n_nodes"]
    n_samples = args.samples if args.samples else config["algorithm"]["n_samples"]

    G = nx.erdos_renyi_graph(n_nodes, edge_probability)
    # Initial random coloring
    initial_coloring = generate_random_coloring(G, max_colors)
    print(f"Initial number of colors: {count_distinct_colors(initial_coloring)}")

    # Run the sampling coloring algorithm
    if args.algorithm == 1:
        coloring_result, min_loss, attempts = sampling_coloring(
            G, n_samples, max_colors
        )

    print(f"Number of colors after algorithm: {count_distinct_colors(coloring_result)}")
    print(f"Minimum loss (conflicts): {min_loss}")
    print(f"Attempts: {attempts}")

    generate_chart(G, coloring_result)
