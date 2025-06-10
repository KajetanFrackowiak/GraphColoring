from typing import Dict, Tuple, List
import networkx as nx
from itertools import product
from .utils import calculate_loss, generate_random_coloring, get_neighbor
from math import exp
import random
import multiprocessing as mp
from multiprocessing.pool import Pool
from functools import partial

from enum import Enum
import numpy as np

# Algorytm pełnego przeglądu (Deterministic)
def brute_force_coloring(
    graph: nx.Graph, max_colors: int
) -> Tuple[Dict[int, int], int, int]:
    """Deterministic search starting from minimum possible colors"""
    nodes = list(graph.nodes())
    best_coloring = None
    best_loss = float("inf")
    attempts = 1
    min_colors = 1  # Start from minimum possible colors

    # Try coloring the graph with min_colors to max_colors
    for num_colors in range(min_colors, max_colors + 1):
        # Try all combinations
        for colors in product(range(num_colors), repeat=len(nodes)):
            attempts += 1
            coloring = dict(zip(nodes, colors))
            loss = calculate_loss(graph, coloring)

            if loss < best_loss:
                best_loss = loss
                best_coloring = coloring
                if calculate_loss(graph, coloring) == 0:  # No conflicts
                    return best_coloring, best_loss, attempts

    return best_coloring, best_loss, attempts


# Algorytm wpinaczkowy (Deterministic)
def deterministic_hill_climbing(graph: nx.Graph) -> Tuple[Dict[int, int], int, int]:
    """Hill climbing with deterministic selection of best neighbor"""
    num_colors = len(graph.nodes()) // 2
    current = generate_random_coloring(graph, num_colors)
    current_loss = calculate_loss(graph, current)
    attempts = 1

    while True:
        best_neighbor = current
        best_neighbor_loss = current_loss

        # Check all possible neighbors
        for node in graph.nodes():
            # for each node try each node and find where is the best loss
            for color in range(num_colors):
                if color != current[node]:
                    neighbor = current.copy()
                    neighbor[node] = color
                    neighbor_loss = calculate_loss(graph, neighbor)
                    attempts += 1

                    if neighbor_loss < best_neighbor_loss:
                        best_neighbor = neighbor
                        best_neighbor_loss = neighbor_loss

        # If no better neighbor found, return current solution
        if best_neighbor_loss >= current_loss:
            return current, current_loss, attempts

        current = best_neighbor
        current_loss = best_neighbor_loss


# Algorytm wpinaczkowy (Heuristic)
def stochastic_hill_climbing(
    graph: nx.Graph, max_attempts: int = 1000
) -> Tuple[Dict[int, int], int, int]:
    """Hill climbing with random neighbor selection"""
    num_colors = len(graph.nodes()) // 2
    current = generate_random_coloring(graph, num_colors)
    current_loss = calculate_loss(graph, current)
    attempts = 1

    while attempts < max_attempts:
        # Get random neighbor (get the same coloring except one that was changed)
        neighbor = get_neighbor(graph, current, num_colors)
        neighbor_loss = calculate_loss(graph, neighbor)
        attempts += 1

        # Accept if the new coloring is better
        if neighbor_loss < current_loss:
            current = neighbor
            current_loss = neighbor_loss
            if current_loss == 0:
                break

    return current, current_loss, attempts


# Algorytm tabu (Heuristic)
def tabu_search(
    graph: nx.Graph, tabu_size: int, max_iterations: int = 1000
) -> Tuple[Dict[int, int], int, int]:
    """Tabu Search algorithm for graph coloring"""
    num_colors = len(graph.nodes()) // 2
    current = generate_random_coloring(graph, num_colors)
    current_loss = calculate_loss(graph, current)
    best_solution = current.copy()
    best_loss = current_loss
    tabu_list = []
    attempts = 1

    while attempts < max_iterations and current_loss > 0:
        best_neighbor = None
        best_neighbor_loss = float("inf")
        best_move = None

        # Examine all possible moves
        for node in graph.nodes():
            for color in range(num_colors):
                if color != current[node]:
                    move = (node, current[node], color)
                    if move not in tabu_list:
                        neighbor = current.copy()
                        neighbor[node] = color
                        neighbor_loss = calculate_loss(graph, neighbor)
                        attempts += 1

                        if neighbor_loss < best_neighbor_loss:
                            best_neighbor = neighbor
                            best_neighbor_loss = neighbor_loss
                            best_move = move

        # If no non-tabu move found (cannot be better, found only tabu moves)
        if best_neighbor is None:
            break

        # Update current solution
        current = best_neighbor
        current_loss = best_neighbor_loss

        # Update best solution if improved
        if current_loss < best_loss:
            best_solution = current.copy()
            best_loss = current_loss

        # Update tabu list
        tabu_list.append(best_move)
        # If tabu list is too big
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_solution, best_loss, attempts


class CoolingSchedule(Enum):
    # The rate at which the temperature is reduced during the search
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"


def get_temperature(
    initial_temp: float, current_iter: int, max_iter: int, schedule: CoolingSchedule
) -> float:
    """Calculate temperature based on selected cooling"""
    if schedule == CoolingSchedule.LINEAR:
        return initial_temp * (1 - current_iter / max_iter)
    elif schedule == CoolingSchedule.EXPONENTIAL:
        return initial_temp * (0.95**current_iter)
    elif schedule == CoolingSchedule.LOGARITHMIC:
        return initial_temp / (1 + np.log(1 + current_iter))
    return 0


def get_gaussian_neighbor(
    graph: nx.Graph, current: Dict[int, int], num_colors: int, std_dev: float = 1.0
) -> Dict[int, int]:
    """Generate neighbor using Gaussian distribution for color selection"""
    neighbor = current.copy()
    node = np.random.choice(list(graph.nodes()))
    current_color = current[node]

    # Use Gaussian distribution centered around current color
    new_color = int(round(random.gauss(current_color, std_dev))) % num_colors
    # Change existing node to some new one
    neighbor[node] = new_color
    return neighbor


# Algorytm symulowanego wyżarzania (Heuristic)
def simulated_annealing(
    graph: nx.Graph,
    initial_temp: float = 100.0,
    min_temp: float = 0.1,
    max_iterations: int = 1000,
    schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL,
) -> Tuple[Dict[int, int], int, int]:
    """Simulated Annealing for graph coloring"""
    num_colors = len(graph.nodes()) // 2
    current = generate_random_coloring(graph, num_colors)
    current_loss = calculate_loss(graph, current)
    best_solution = current.copy()
    best_loss = current_loss
    temperature = initial_temp
    attempts = 1

    while temperature > min_temp and attempts < max_iterations and current_loss > 0:
        # Generate neighbor using Gaussian distribution
        neighbor = get_gaussian_neighbor(graph, current, num_colors)
        neighbor_loss = calculate_loss(graph, neighbor)
        attempts += 1

        # Calculate acceptance probability
        delta = neighbor_loss - current_loss
        if delta < 0 or random.random() < exp(-delta / temperature):
            current = neighbor
            current_loss = neighbor_loss

            if current_loss < best_loss:
                best_solution = current.copy()
                best_loss = current_loss

        # Update temperature
        temperature = get_temperature(initial_temp, attempts, max_iterations, schedule)

    return best_solution, best_loss, attempts


# Algorytm genetyczny (Heuristic)
class CrossoverType(Enum):
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"


class MutationType(Enum):
    RANDOM = "random"
    SWAP = "swap"


class TerminationType(Enum):
    GENERATIONS = "generations"
    FITNESS = "fitness"


class Individual:
    def __init__(self, coloring: Dict[int, int], fitness: int):
        self.coloring = coloring
        self.fitness = fitness

    def copy(self):
        return Individual(self.coloring.copy(), self.fitness)


def uniform_crossover(
    parent1: Individual, parent2: Individual, crossover_rate: float = 0.5
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Uniform crossover between two parents"""
    child1 = parent1.coloring.copy()
    child2 = parent2.coloring.copy()
    
    # Cross each node instead of from some point only (to cross, random value must be greather then crossover_rate)
    for node in child1.keys():
        if random.random() < crossover_rate:
            child1[node], child2[node] = child2[node], child1[node]

    return child1, child2


def single_point_crossover(
    parent1: Individual, parent2: Individual
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Single point crossover between two parents"""
    nodes = list(parent1.coloring.keys())
    point = random.randint(1, len(nodes) - 1)

    child1 = parent1.coloring.copy()
    child2 = parent2.coloring.copy()

    # Cross from some point all children, without random possibility on crossing
    for node in nodes[point:]:
        child1[node], child2[node] = child2[node], child1[node]

    return child1, child2


def random_mutation(
    coloring: Dict[int, int], num_colors: int, mutation_rate: float = 0.1
) -> Dict[int, int]:
    """Random color mutation"""
    mutated = coloring.copy()
    for node in mutated:
        if random.random() < mutation_rate:
            # Change node on some other random node from pull of possible colors
            mutated[node] = random.randint(0, num_colors - 1)
    return mutated


def swap_mutation(
    coloring: Dict[int, int], mutation_rate: float = 0.1
) -> Dict[int, int]:
    """Swap colors between two random nodes"""
    mutated = coloring.copy()
    nodes = list(mutated.keys())

    if random.random() < mutation_rate and len(nodes) >= 2:
        # Take some random formation of nodes
        i, j = random.sample(population=nodes, k=2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    """Selects a parent using tournament selection."""
    if not population:
        raise ValueError("Population cannot be empty")
    if tournament_size <= 0:
        raise ValueError("Tournament size cannot be negative")
    tournament_contenders = random.sample(population, min(tournament_size, len(population)))
    return max(tournament_contenders, key=lambda individual: individual.fitness)

# FITNESS == -LOSS
def genetic_algorithm(
    graph: nx.Graph,
    population_size: int = 50,
    elite_size: int = 5,
    max_generations: int = 100,
    crossover_type: CrossoverType = CrossoverType.UNIFORM,
    mutation_type: MutationType = MutationType.RANDOM,
    termination_type: TerminationType = TerminationType.GENERATIONS,
    tournament_size: int = 5,
) -> Tuple[Dict[int, int], int, int]:
    """Genetic Algorithm for graph coloring"""
    num_colors = len(graph.nodes()) // 2
    attempts = 0

    # Initialize population
    population = []
    for _ in range(population_size):
        coloring = generate_random_coloring(graph, num_colors)
        fitness = -calculate_loss(
            graph, coloring
        )  # Negative, because lower loss means better, and bigger fitness means better, we want to minimalize loss
            # and to maximize fitness so we invert the loss
        attempts += 1  # Count initial population evaluations
        population.append(Individual(coloring, fitness))

    best_solution = max(population, key=lambda x: x.fitness)

    generation = 0
    while True:
        # Sort population by fitness, bigger fitness means better, so we want to sort in desc
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Early stopping conditions
        if best_solution.fitness == 0:  # Perfect solution found
            break
        if (
            termination_type == TerminationType.GENERATIONS
            and generation >= max_generations
        ):
            break

        # Create new population
        new_population = []
        # Stay only elite from population (natural selection) 
        new_population.extend(population[:elite_size])

        # Create offspring until population is filled
        while len(new_population) < population_size:
            # Select parents from top half of population (sorted by fitness)
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            # Ensure parents are different
            if len(population) > 1:
                while parent2 is parent1:
                    parent2 = tournament_selection(population, tournament_size)

            # Apply crossover
            if crossover_type == CrossoverType.UNIFORM:
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = single_point_crossover(parent1, parent2)

            # Apply mutation
            if mutation_type == MutationType.RANDOM:
                child1 = random_mutation(child1, num_colors)
                child2 = random_mutation(child2, num_colors)
            else:
                child1 = swap_mutation(child1)
                child2 = swap_mutation(child2)

            # Evaluate new solutions
            fitness1 = -calculate_loss(graph, child1)
            fitness2 = -calculate_loss(graph, child2)
            attempts += 2  # Count offspring evaluations

            new_population.extend(
                [Individual(child1, fitness1), Individual(child2, fitness2)]
            )

            # Early stopping if perfect solution found
            if fitness1 == 0 or fitness2 == 0:
                best_solution = Individual(child1 if fitness1 == 0 else child2, 0)
                return best_solution.coloring, 0, attempts

        # Update population
        population = new_population[:population_size]

        # Update best solution
        current_best = max(population, key=lambda x: x.fitness)
        if current_best.fitness > best_solution.fitness:
            best_solution = current_best.copy()

        generation += 1

    return best_solution.coloring, -best_solution.fitness, attempts


# Algorytm genetyczny - wersja równoległa (Heuristic)

def evaluate_population_parallel(
    graph: nx.Graph, coloring: List[Dict[int, int]], pool: Pool
) -> List[int]:
    """Evaluate population fitness in parallel"""
    evaluate_func = partial(calculate_loss, graph)
    return pool.map(evaluate_func, coloring)


def parallel_genetic_algorithm(
    graph: nx.Graph,
    population_size: int = 50,
    elite_size: int = 5,
    max_generations: int = 100,
    crossover_type: CrossoverType = CrossoverType.UNIFORM,
    mutation_type: MutationType = MutationType.RANDOM,
    termination_type: TerminationType = TerminationType.GENERATIONS,
    num_processes: int = None,
    tournament_size: int = 5,
) -> Tuple[Dict[int, int], int, int]:
    """Parallel Genetic Algorithm for graph coloring"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    num_colors = len(graph.nodes()) // 2
    attempts = 0

    # Intialize multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Initialize population
        coloring = [
            generate_random_coloring(graph, num_colors) for _ in range(population_size)
        ]
        fitnesses = evaluate_population_parallel(graph, coloring, pool)
        attempts += population_size

        population = [Individual(c, -f) for c, f in zip(coloring, fitnesses)]
        best_solution = max(population, key=lambda x: x.fitness)

        generation = 0
        while True:
            # Sort population by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Early stopping conditions (loss == 0, because fitenss is the inverse of loss)
            if best_solution.fitness == 0:
                break
            if (
                termination_type == TerminationType.GENERATIONS
                and generation >= max_generations
            ):
                break

            # Create new population
            new_population = []
            new_population.extend(population[:elite_size])

            # Create offspring list for parallel evaluation
            offspring_colorings = []
            while len(new_population) + len(offspring_colorings) // 2 < population_size:
                parent1 = tournament_selection(population, tournament_size)
                parent2 = tournament_selection(population, tournament_size)
                if len(population) > 1:
                    while parent2 is parent1:
                        parent2 = tournament_selection(population, tournament_size)

                # Apply crossover
                if crossover_type == CrossoverType.UNIFORM:
                    child1, child2 = uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = single_point_crossover(parent1, parent2)

                # Apply mutation
                if mutation_type == MutationType.RANDOM:
                    child1 = random_mutation(child1, num_colors)
                    child2 = random_mutation(child2, num_colors)
                else:
                    child1 = swap_mutation(child1)
                    child2 = swap_mutation(child2)

                offspring_colorings.extend([child1, child2])

            # offspring_fitnesses and offspring_colorings are separeted, because offspring_fitnesses is in parallel and the second not
            offspring_fitnesses = evaluate_population_parallel(
                graph, offspring_colorings, pool
            )
            attempts += len(offspring_fitnesses)

            # Create new individuals
            for i in range(0, len(offspring_colorings), 2):
                # In not parallel children and fitness are together in new_population

                child1_coloring = offspring_colorings[i]
                child2_coloring = offspring_colorings[i + 1]
                # In parallel, because loss calculating is time-consuming
                child1_fitness = offspring_fitnesses[i]
                child2_fitness = offspring_fitnesses[i + 1]

                # Early stopping if perfect solution found
                if child1_fitness == 0 or child2_fitness == 0:
                    best_solution = Individual(
                        child1_coloring if child1_fitness == 0 else child2_coloring, 0
                    )
                    return best_solution.coloring, 0, attempts

                new_population.extend(
                    [
                        Individual(child1_coloring, -child1_fitness),
                        Individual(child2_coloring, -child2_fitness),
                    ]
                )

            # Update population
            population = new_population[:population_size]

            # Update best solution
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_solution.fitness:
                best_solution = current_best.copy()

            generation += 1

        return best_solution.coloring, -best_solution.fitness, attempts

# Algorytm genetyczny - wersja wyspowa (Heuristic)
def island_genetic_algorithm(
    graph: nx.Graph,
    num_islands: int = 4,
    migration_rate: float = 0.1,
    migration_interval: int = 10,
    population_size: int = 50,
    elite_size: int = 5,
    max_generations: int = 100,
    crossover_type: CrossoverType = CrossoverType.UNIFORM,
    mutation_type: MutationType = MutationType.RANDOM,
    termination_type: TerminationType = TerminationType.GENERATIONS,
    num_processes: int = None,
    tournament_size: int = 5,
) -> Tuple[Dict[int, int], int, int]:
    """Island Model Genetic Algorithm for graph coloring"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    num_colors = len(graph.nodes()) // 2
    attempts = 0
    island_size = population_size // num_islands

    with mp.Pool(processes=num_processes) as pool:
        # Initialize islands
        islands = []
        for _ in range(num_islands):
            colorings = [
                generate_random_coloring(graph, num_colors) for _ in range(island_size)
            ]
            fitnesses = evaluate_population_parallel(graph, colorings, pool)
            attempts += island_size
            population = [Individual(c, -f) for c, f in zip(colorings, fitnesses)]
            # Each island contain many individuals (population)
            # with Individual.coloring Dict[int, int], Individual.fitness
            islands.append(population)

        # Inner max takes best Individual from each island
        # Outer max takes best Individual from all best Individuals
        best_solution = max(
            (max(island, key=lambda x: x.fitness) for island in islands),
            key=lambda x: x.fitness,
        )

        generation = 0
        while True:
            # Process each island
            for i in range(num_islands):
                # Descending, because bigger x.fitness is better
                islands[i].sort(key=lambda x: x.fitness, reverse=True)

                current_best = max(islands[i], key=lambda x: x.fitness)
                if current_best.fitness > best_solution.fitness:
                    best_solution = current_best.copy()

                if best_solution.fitness == 0:
                    return best_solution.coloring, 0, attempts

                if (
                    termination_type == TerminationType.GENERATIONS
                    and generation >= max_generations
                ):
                    # Because the algorithm is to maximize fitness, and we wanted to minimize graph,
                    # so we find max fintess and return with minus sign as best
                    return best_solution.coloring, -best_solution.fitness, attempts

                # Create new population for island
                new_population = []
                # Stays only elite
                new_population.extend(islands[i][:elite_size])

                # Create offspring list for parallel evaluation
                offspring_colorings = []
                
                # the condition ensures that when generatng offspring, the total population size
                # (existing individuals + new offspring (since each pair produces two offspring) does not exceed this island capacity
                while len(new_population) + len(offspring_colorings) // 2 < island_size:
                    parent1 = tournament_selection(islands[i], tournament_size)
                    parent2 = tournament_selection(islands[i], tournament_size)
                    if len(islands[i]) > 1:
                        while parent2 is parent1:
                            parent2 = tournament_selection(islands[i], tournament_size)

                    if crossover_type == CrossoverType.UNIFORM:
                        child1, child2 = uniform_crossover(parent1, parent2)
                    else:
                        child1, child2 = single_point_crossover(parent1, parent2)

                    if mutation_type == MutationType.RANDOM:
                        child1 = random_mutation(child1, num_colors)
                        child2 = random_mutation(child2, num_colors)
                    else:
                        child1 = swap_mutation(child1)
                        child2 = swap_mutation(child2)

                    offspring_colorings.extend([child1, child2])

                # Parallel fitness evaluation
                offspring_fitnesses = evaluate_population_parallel(
                    graph, offspring_colorings, pool
                )
                attempts += len(offspring_colorings)

                # Create new individuals
                for j in range(0, len(offspring_colorings), 2):
                    child1_coloring = offspring_colorings[j]
                    child2_coloring = offspring_colorings[j + 1]
                    child1_fitness = offspring_fitnesses[j]
                    child2_fitness = offspring_fitnesses[j + 1]

                    if child1_fitness == 0 or child2_fitness == 0:
                        best_solution = Individual(
                            child1_coloring if child1_fitness == 0 else child2_coloring,
                            0,
                        )
                        return best_solution.coloring, 0, attempts

                    new_population.extend(
                        [
                            Individual(child1_coloring, -child1_fitness),
                            Individual(child2_coloring, -child2_fitness),
                        ]
                    )

                islands[i] = new_population[:island_size]

            # Migration phase
            if generation > 0 and generation % migration_interval == 0:
                migrants_per_island = int(island_size * migration_rate)
                for i in range(num_islands):
                    # Select best individuals as migrants
                    migrants = islands[i][:migrants_per_island]
                    # Send to next island (ring topology) to limit to max num_island
                    next_island = (i + 1) % num_islands
                    # Replace random individuals in target island
                    replace_indices = random.sample(
                        range(island_size), migrants_per_island
                    )
                    for idx, migrant in zip(replace_indices, migrants):
                        islands[next_island][idx] = migrant.copy()

            generation += 1

    return best_solution.coloring, -best_solution.fitness, attempts