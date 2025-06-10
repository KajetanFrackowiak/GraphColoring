import time
import networkx as nx
import argparse
import sys

from graph_coloring.graph_algorithms import (
    brute_force_coloring,
    deterministic_hill_climbing,
    stochastic_hill_climbing,
    tabu_search,
    CoolingSchedule,
    simulated_annealing,
    CrossoverType,
    MutationType,
    TerminationType,
    genetic_algorithm,
    parallel_genetic_algorithm,
    island_genetic_algorithm, 
)
from graph_coloring.visualization import generate_chart

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph coloring algorithms")
    parser.add_argument(
        "--algorithm",
        type=int,
        help="Algorithm to use:\n"
        "1: Brute Force\n"
        "2: Deterministic hill climbing\n"
        "3: Stochastic hill climbing\n"
        "4: Tabu\n"
        "5: Cooling\n"
        "6: Genetic Algorithm\n"
        "7: Parallel Genetic Algorithm\n"
        "8: Island Genetic Algorithm\n",
        required=True,
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edge_prob", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_colors", type=int, default=5)

    parser.add_argument("--tabu_size", type=int, default=7)

    parser.add_argument(
        "--initial_temp", type=float, default=100.0, help="Initial temperature for SA"
    )
    parser.add_argument(
        "--min_temp", type=float, default=0.1, help="Minimum temperature for SA"
    )
    parser.add_argument(
        "--cooling",
        type=str,
        choices=["linear", "exponential", "logarithmic"],
        default="exponential",
        help="Cooling schedule for SA",
    )

    parser.add_argument("--population_size", type=int, default=60)
    parser.add_argument("--elite_size", type=int, default=5)
    parser.add_argument(
        "--crossover", type=str, choices=["uniform", "single_point"], default="uniform"
    )
    parser.add_argument(
        "--mutation", type=str, choices=["random", "swap"], default="random"
    )
    parser.add_argument(
        "--termination",
        type=str,
        choices=["generations", "fitness"],
        default="generations",
    )
    parser.add_argument("--tournament_size", type=int, default=5)
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes for parallel GA",
    )

    parser.add_argument("--num_islands", type=int, default=4)
    parser.add_argument("--migration_rate", type=float, default=0.1)
    parser.add_argument("--migration_interval", type=int, default=10)

    parser.add_argument(
        "--es_mu", type=int, default=15, help="Parent population size for ES"
    )
    parser.add_argument(
        "--es_lambda", type=int, default=30, help="Offspring population size for ES"
    )

    args = parser.parse_args()

    G = nx.erdos_renyi_graph(args.nodes, args.edge_prob)

    if args.algorithm == 1:
        alg_start = time.perf_counter()
        coloring, loss, attempts = brute_force_coloring(G, args.max_colors)
        alg_end = time.perf_counter()
    elif args.algorithm == 2:
        alg_start = time.perf_counter()
        coloring, loss, attempts = deterministic_hill_climbing(G)
        alg_end = time.perf_counter()
    elif args.algorithm == 3:
        alg_start = time.perf_counter()
        coloring, loss, attempts = stochastic_hill_climbing(G, args.samples)
        alg_end = time.perf_counter()
    elif args.algorithm == 4:
        alg_start = time.perf_counter()
        coloring, loss, attempts = tabu_search(G, args.tabu_size)
        alg_end = time.perf_counter()
    elif args.algorithm == 5:
        alg_start = time.perf_counter()
        schedule = CoolingSchedule(args.cooling)
        coloring, loss, attempts = simulated_annealing(
            G, args.initial_temp, args.min_temp, args.samples, schedule
        )
        alg_end = time.perf_counter()
    elif args.algorithm == 6:
        alg_start = time.perf_counter()
        crossover = CrossoverType(args.crossover)
        mutation = MutationType(args.mutation)
        termination = TerminationType(args.termination)
        coloring, loss, attempts = genetic_algorithm(
            G,
            population_size=args.population_size,
            elite_size=args.elite_size,
            max_generations=args.samples,
            crossover_type=crossover,
            mutation_type=mutation,
            termination_type=termination,
            tournament_size=args.tournament_size,
        )
        alg_end = time.perf_counter()
    elif args.algorithm == 7:  # Parallel GA
        alg_start = time.perf_counter()
        crossover = CrossoverType(args.crossover)
        mutation = MutationType(args.mutation)
        termination = TerminationType(args.termination)
        coloring, loss, attempts = parallel_genetic_algorithm(
            G,
            population_size=args.population_size,
            elite_size=args.elite_size,
            max_generations=args.samples,
            crossover_type=crossover,
            mutation_type=mutation,
            termination_type=termination,
            num_processes=args.num_processes,
            tournament_size=args.tournament_size,
        )
        alg_end = time.perf_counter()
    elif args.algorithm == 8:
        alg_start = time.perf_counter()
        crossover = CrossoverType(args.crossover)
        mutation = MutationType(args.mutation)
        termination = TerminationType(args.termination)
        coloring, loss, attempts = island_genetic_algorithm(
            G,
            num_islands=args.num_islands,
            migration_rate=args.migration_rate,
            migration_interval=args.migration_interval,
            population_size=args.population_size,
            elite_size=args.elite_size,
            max_generations=args.samples,
            crossover_type=crossover,
            mutation_type=mutation,
            termination_type=termination,
            num_processes=args.num_processes,
            tournament_size=args.tournament_size,
        )
        alg_end = time.perf_counter()
   

    print(f"Attempts: {attempts}")
    print(f"Number of conflicts: {loss}")
    print(f"Number of colors used: {len(set(coloring.values()))}")
    print(
        f"Algorithm {args.algorithm} execution time: {alg_end - alg_start:.6f} seconds"
    )
    generate_chart(G, coloring)