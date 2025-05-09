from enum import Enum

import networkx as nx
import time
import psutil
import json
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from graph_coloring.graph_algorithms import (
    brute_force_coloring,
    deterministic_hill_climbing,
    stochastic_hill_climbing,
    tabu_search,
    simulated_annealing,
    genetic_algorithm,
    island_genetic_algorithm,
    CrossoverType,
    MutationType,
    TerminationType,
    CoolingSchedule, sampling_coloring,
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
    def __init__(self, output_dir: str = "results", max_time: int = 60):
        self.output_dir = Path(output_dir)
        print(f"Creating directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.convergence_data = {}
        self.max_time = max_time

    def measure_memory(self) -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024

    def run_experiment(self, G: nx.Graph, algorithms_config: List[Dict]) -> None:
        for config in algorithms_config:
            print(f"Running {config['name']}...")
            print(f"Maximum allowed time: {self.max_time} seconds")
            initial_memory = self.measure_memory()
            peak_memory = initial_memory
            start_time = time.perf_counter()
            convergence = []

            try:
                # Use multiprocessing to enforce timeout
                from multiprocessing import Process, Manager

                # Create a manager to share data between processes
                with Manager() as manager:
                    result_dict = manager.dict()
                    result_dict["coloring"] = None
                    result_dict["conflicts"] = None
                    result_dict["attempts"] = None
                    result_dict["completed"] = False

                    # Define the function to run the algorithm
                    def run_algorithm(G, config, result_dict):
                        try:
                            if config["name"] == "Sampling Coloring":
                                coloring, conflicts, attempts = sampling_coloring(G, config["params"]["n_samples"])
                            elif config["name"] == "Brute Force":
                                coloring, conflicts, attempts = brute_force_coloring(
                                    G, config["params"]["max_colors"]
                                )
                            elif config["name"] == "Deterministic Hill Climbing":
                                coloring, conflicts, attempts = deterministic_hill_climbing(G)
                            elif config["name"] == "Stochastic Hill Climbing":
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
                            elif config["name"] == "Parallel Genetic Algorithm":
                                coloring, conflicts, attempts = genetic_algorithm(
                                    G,
                                    population_size=config["params"]["population_size"],
                                    elite_size=config["params"]["elite_size"],
                                    max_generations=config["params"]["max_generations"],
                                    crossover_type=config["params"]["crossover_type"],
                                    mutation_type=config["params"]["mutation_type"],
                                    termination_type=config["params"]["termination_type"],
                                    num_processes=config["params"]["num_processes"]
                                )
                            elif config["name"] == "Island Genetic Algorithm":
                                coloring, conflicts, attempts = island_genetic_algorithm(
                                    G,
                                    num_islands=config["params"]["num_islands"],
                                    migration_rate=config["params"]["migration_rate"],
                                    migration_interval=config["params"]["migration_interval"],
                                    population_size=config["params"]["population_size"],
                                    elite_size=config["params"]["elite_size"],
                                    max_generations=config["params"]["max_generations"],
                                )

                            # Store results in shared dictionary
                            result_dict["coloring"] = coloring
                            result_dict["conflicts"] = conflicts
                            result_dict["attempts"] = attempts
                            result_dict["completed"] = True
                        except Exception as e:
                            print(f"Algorithm process error: {e}")

                    # Create and start the process
                    p = Process(target=run_algorithm, args=(G, config, result_dict))
                    p.start()

                    # Wait for the process with timeout
                    p.join(self.max_time)

                    # If process is still alive after timeout, terminate it
                    if p.is_alive():
                        p.terminate()
                        p.join()
                        print(f"Timeout for {config['name']} (exceeded {self.max_time}s)")
                        continue

                    # Get the results from the process
                    if result_dict["completed"]:
                        coloring = result_dict["coloring"]
                        conflicts = result_dict["conflicts"]
                        attempts = result_dict["attempts"]
                    else:
                        print(f"Algorithm {config['name']} did not complete successfully")
                        continue

                # Measure final memory and calculate peak
                final_memory = self.measure_memory()
                peak_memory = max(peak_memory, final_memory)
                memory_usage = peak_memory - initial_memory

                execution_time = time.perf_counter() - start_time

                # Create result
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
                print(f"Finished {config['name']} in {execution_time:.2f}s, conflicts: {conflicts}")
                print(f"Memory: initial={initial_memory:.2f}MB, peak={peak_memory:.2f}MB, usage={memory_usage:.2f}MB")

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
    n = int(input("Enter number of vertices (e.g., 120): ") or 120)
    p = float(input("Enter edge probability (e.g., 0.4): ") or 0.4)
    k = int(input("Enter number of colors (e.g., 3): ") or 3)
    iters = int(input("Enter number of iterations (e.g., 100): ") or 100)
    max_time = int(input("Enter maximum execution time in seconds (e.g., 300): ") or 300)

    # Base configuration for all algorithms
    base_params = {"max_iterations": iters, "max_colors": k}

    algorithms_config = [
        {"name": "Sampling Coloring", "params": {"n_samples": iters}},
        {"name": "Brute Force", "params": {"max_colors": base_params["max_colors"]}},
        {"name": "Deterministic Hill Climbing", "params": {}},
        {
            "name": "Stochastic Hill Climbing",
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
            "name": "Parallel Genetic Algorithm",
            "params": {
                "population_size": 50,
                "elite_size": 5,
                "max_generations": base_params["max_iterations"],
                "crossover_type": CrossoverType.UNIFORM,
                "mutation_type": MutationType.RANDOM,
                "termination_type": TerminationType.GENERATIONS,
                "num_processes": 8
            },
        },
        {
            "name": "Island Genetic Algorithm",
            "params": {
                "num_islands": 4,
                "migration_rate": 0.1,
                "migration_interval": 10,
                "population_size": 50,
                "elite_size": 5,
                "max_generations": base_params["max_iterations"],
            },
        },
    ]

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    print(f"\nTesting graph with {n} nodes and {p} edge probability")
    G = nx.erdos_renyi_graph(n, p)

    output_dir = Path(f"results/n{n}_p{p}")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparator = GraphColoringComparator(output_dir=str(output_dir), max_time=max_time)
    comparator.run_experiment(G, algorithms_config)
    comparator.save_results()
    print(f"Results saved in: {output_dir}")
    print(f"Plot saved as: {output_dir}/comparison.png")


if __name__ == "__main__":
    main()