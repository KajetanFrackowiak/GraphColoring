import queue
from multiprocessing import managers
from random import random

import networkx as nx

from graph_coloring.utils import generate_random_coloring, calculate_loss
from graph_coloring.graph_algorithms import Individual


class IslandManager(managers.BaseManager):
    pass


def create_island_server(port: int, island_queue: queue.Queue):
    """Create a server for distributed island communication"""
    IslandManager.register('get_island_queue', callable=lambda: island_queue)
    manager = IslandManager(address=('', port), authkey=b'island_ga')
    server = manager.get_server()
    return server


def connect_to_island(host: str, port: int) -> IslandManager:
    """Connect to a remote island"""
    IslandManager.register('get_island_queue')
    manager = IslandManager(address=(host, port), authkey=b'island_ga')
    manager.connect()
    return manager


def distributed_island_worker(host: str, port: int, graph: nx.Graph,
                              island_size: int, num_colors: int,
                              migration_rate: float, migration_interval: int):
    """Worker process for distributed island"""
    manager = connect_to_island(host, port)
    island_queue = manager.get_island_queue()

    # Initialize population
    population = [
        Individual(generate_random_coloring(graph, num_colors),
                   -calculate_loss(graph, coloring))
        for coloring in [generate_random_coloring(graph, num_colors)
                         for _ in range(island_size)]
    ]

    generation = 0
    while True:
        # Regular island evolution
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Migration phase
        if generation > 0 and generation % migration_interval == 0:
            # Send migrants
            migrants = random.sample(population[:int(len(population) * migration_rate)],
                                     k=int(island_size * migration_rate))
            try:
                island_queue.put(migrants, block=False)
            except queue.Full:
                pass

            # Receive migrants
            try:
                received_migrants = island_queue.get(block=False)
                replace_indices = random.sample(range(island_size), len(received_migrants))
                for idx, migrant in zip(replace_indices, received_migrants):
                    population[idx] = migrant.copy()
            except queue.Empty:
                pass

        generation += 1