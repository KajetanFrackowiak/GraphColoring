from enum import Enum

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from graph_coloring.graph_algorithms import (
    sampling_coloring,
    brute_force_coloring,
    deterministic_hill_climbing,
    stochastic_hill_climbing,
    tabu_search,
    simulated_annealing,
    genetic_algorithm,
    island_genetic_algorithm,
    evolution_strategy,
    CrossoverType,
    MutationType,
    TerminationType,
    CoolingSchedule,
)


@dataclass
class AlgorithmResult:
    name: str
    params: Dict
    coloring: Dict[int, int]
    conflicts: int
    execution_time: float
    memory_usage: float
    convergence: List[float]
    attempts: int


class GraphColoringComparator:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        print(f"Creating directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.convergence_data = {}

    def measure_memory(self) -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024

    def run_experiment(self, G: nx.Graph, algorithms_config: List[Dict]) -> None:
        for config in algorithms_config:
            print(f"Running {config['name']}...")
            start_memory = self.measure_memory()
            start_time = time.perf_counter()
            convergence = []
            max_time = 60  # 60 seconds timeout

            try:
                if config["name"] == "Brute Force":
                    coloring, conflicts, attempts = brute_force_coloring(
                        G, config["params"]["max_colors"]
                    )
                elif config["name"] == "Hill Climbing":
                    coloring, conflicts, attempts = deterministic_hill_climbing(G)
                elif config["name"] == "Stochastic HC":
                    coloring, conflicts, attempts = stochastic_hill_climbing(
                        G, config["params"]["max_iterations"]
                    )
                elif config["name"] == "Tabu Search":
                    coloring, conflicts, attempts = tabu_search(
                        G,
                        config["params"]["tabu_size"],
                        config["params"]["max_iterations"],
                    )
                elif config["name"] == "Simulated Annealing":
                    coloring, conflicts, attempts = simulated_annealing(
                        G,
                        config["params"]["initial_temp"],
                        config["params"]["min_temp"],
                        config["params"]["max_iterations"],
                        config["params"]["schedule"],
                    )
                elif config["name"] == "Genetic Algorithm":
                    coloring, conflicts, attempts = genetic_algorithm(
                        G,
                        population_size=config["params"]["population_size"],
                        elite_size=config["params"]["elite_size"],
                        max_generations=config["params"]["max_generations"],
                        crossover_type=config["params"]["crossover_type"],
                        mutation_type=config["params"]["mutation_type"],
                        termination_type=config["params"]["termination_type"],
                    )
                elif config["name"] == "Island GA":
                    coloring, conflicts, attempts = island_genetic_algorithm(
                        G,
                        num_islands=config["params"]["num_islands"],
                        migration_rate=config["params"]["migration_rate"],
                        migration_interval=config["params"]["migration_interval"],
                        population_size=config["params"]["population_size"],
                        elite_size=config["params"]["elite_size"],
                        max_generations=config["params"]["max_generations"],
                    )
                elif config["name"] == "Evolution Strategy":
                    coloring, conflicts, attempts = evolution_strategy(
                        G,
                        num_colors=config["params"]["num_colors"],
                        loss_function=config["params"]["loss_function"],
                        mu=config["params"]["mu"],
                        lambda_=config["params"]["lambda_"],
                        generations=config["params"]["generations"],
                    )

                if time.perf_counter() - start_time > max_time:
                    print(f"Timeout for {config['name']}")
                    continue

                execution_time = time.perf_counter() - start_time
                memory_usage = self.measure_memory() - start_memory

                result = AlgorithmResult(
                    name=config["name"],
                    params=config["params"],
                    coloring=coloring,
                    conflicts=conflicts,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    convergence=convergence,
                    attempts=attempts,
                )
                self.results.append(result)
                print(f"Finished {config['name']} in {execution_time:.2f}s")

            except Exception as e:
                print(f"Error in {config['name']}: {str(e)}")
                continue

    def plot_comparisons(self) -> None:
        """Generate comparison plots"""
        if not self.results:
            print("No results to plot")
            return

        print("Starting plot generation...")

        # Set matplotlib backend to Agg
        plt.switch_backend("Agg")

        try:
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax1, ax2, ax3 = axes
            print("Figure created")
        except Exception as e:
            print(f"Error creating figure: {e}")
            return

        # Prepare data
        names = [r.name for r in self.results]
        times = [r.execution_time for r in self.results]
        memory = [r.memory_usage for r in self.results]
        conflicts = [r.conflicts for r in self.results]

        print("Data prepared for plotting")

        # Plot execution times
        try:
            sns.barplot(x=names, y=times, ax=ax1)
            ax1.set_title("Execution Time")
            ax1.set_xlabel("Algorithm")
            ax1.set_ylabel("Time (s)")
            ax1.tick_params(axis="x", rotation=45)
            print("Execution time plot created")
        except Exception as e:
            print(f"Error creating execution time plot: {e}")

        # Plot memory usage
        try:
            sns.barplot(x=names, y=memory, ax=ax2)
            ax2.set_title("Memory Usage")
            ax2.set_xlabel("Algorithm")
            ax2.set_ylabel("Memory (MB)")
            ax2.tick_params(axis="x", rotation=45)
            print("Memory usage plot created")
        except Exception as e:
            print(f"Error creating memory usage plot: {e}")

        # Plot solution quality
        try:
            sns.barplot(x=names, y=conflicts, ax=ax3)
            ax3.set_title("Solution Quality")
            ax3.set_xlabel("Algorithm")
            ax3.set_ylabel("Conflicts")
            ax3.tick_params(axis="x", rotation=45)
            print("Solution quality plot created")
        except Exception as e:
            print(f"Error creating solution quality plot: {e}")

        plt.tight_layout()
        print("Layout adjusted")

        # Save plot
        try:
            plot_path = self.output_dir / "comparison.png"
            print(f"Saving plot to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Plot saved successfully")
        except Exception as e:
            print(f"Error saving plot: {e}")

    def save_results(self) -> None:
        """Save results to JSON and generate plots"""
        if not self.results:
            print("No results to save")
            return

        # Save results to JSON with enum handling
        results_dict = {
            "graph_info": {
                "nodes": len(self.results[0].coloring),
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            },
            "algorithms": [],
        }

        # Convert enum values to strings in parameters
        for result in self.results:
            params = {}
            for key, value in result.params.items():
                if isinstance(value, Enum):
                    params[key] = value.value
                else:
                    params[key] = value

            algorithm_data = {
                "name": result.name,
                "parameters": params,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage,
                "conflicts": result.conflicts,
                "attempts": result.attempts,
            }
            results_dict["algorithms"].append(algorithm_data)

            # Save JSON results
        json_path = self.output_dir / "results.json"
        print(f"Saving results to: {json_path}")
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print("Results saved to JSON")

        # Generate and save plots
        self.plot_comparisons()
        plt.close("all")  # Close all figures to free memory
        print("Plots generated and saved")


def main():
    # Base configuration for all algorithms
    base_params = {"max_iterations": 100, "max_colors": 5}

    algorithms_config = [
        {"name": "Brute Force", "params": {"max_colors": base_params["max_colors"]}},
        {"name": "Hill Climbing", "params": {}},
        {
            "name": "Stochastic HC",
            "params": {"max_iterations": base_params["max_iterations"]},
        },
        {
            "name": "Tabu Search",
            "params": {"tabu_size": 7, "max_iterations": base_params["max_iterations"]},
        },
        {
            "name": "Simulated Annealing",
            "params": {
                "initial_temp": 100.0,
                "min_temp": 0.1,
                "max_iterations": base_params["max_iterations"],
                "schedule": CoolingSchedule.EXPONENTIAL,
            },
        },
        {
            "name": "Genetic Algorithm",
            "params": {
                "population_size": 50,
                "elite_size": 5,
                "max_generations": base_params["max_iterations"],
                "crossover_type": CrossoverType.UNIFORM,
                "mutation_type": MutationType.RANDOM,
                "termination_type": TerminationType.GENERATIONS,
            },
        },
        {
            "name": "Island GA",
            "params": {
                "num_islands": 4,
                "migration_rate": 0.1,
                "migration_interval": 10,
                "population_size": 50,
                "elite_size": 5,
                "max_generations": base_params["max_iterations"],
            },
        },
        {
            "name": "Evolution Strategy",
            "params": {
                "num_colors": base_params["max_colors"],
                "loss_function": "ackley",
                "mu": 15,
                "lambda_": 30,
                "generations": base_params["max_iterations"],
            },
        },
    ]

    # Test different graph sizes
    graph_sizes = [8, 10]
    edge_probs = [0.3, 0.5]

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    for n in graph_sizes:
        for p in edge_probs:
            print(f"\nTesting graph with {n} nodes and {p} edge probability")
            G = nx.erdos_renyi_graph(n, p)

            output_dir = Path(f"results/n{n}_p{p}")
            output_dir.mkdir(parents=True, exist_ok=True)

            comparator = GraphColoringComparator(output_dir=str(output_dir))
            comparator.run_experiment(G, algorithms_config)
            comparator.save_results()
            # comparator.plot_comparisons()
            print(f"Results saved in: {output_dir}")
            print(f"Plot saved as: {output_dir}/comparison.png")


if __name__ == "__main__":
    main()
