import numpy as np
import matplotlib.pyplot as plt


# Contains local minimum
def rosenbrock(x):
    return sum(100 * (x[1:] - np.array(x[:-1]) ** 2) ** 2 + (1 - np.array(x[:-1])) ** 2)


# Contains local minimum
def sphere(x):
    return sum(np.array(x) ** 2)


# Does not contain local minimum
def rastrigin(x):
    return 10 * len(x) + sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) for x_i in x)


def plot_function(func, ax, title, limits=(-5, 5)):
    x = np.linspace(start=limits[0], stop=limits[1], num=400)
    y = np.linspace(start=limits[0], stop=limits[1], num=400)
    # Take two 1D arrays and create two 2D arrays representing all combinations of the x and y
    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [[func([xi, yi]) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)]
    )

    c = ax.contour(X, Y, Z, 50, cmap="inferno")
    ax.clabel(c, inline=True, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")


def create_population(pop_size, dimensions):
    return np.random.uniform(low=-5, high=5, size=(pop_size, dimensions))


# Mutation by adding noise to individuals
def mutate(x, mutation_rate=0.1):
    return x + mutation_rate * np.random.randn(*x.shape)


def select_best(population, fitness_values, num_best=10):
    # We want to select best individuals, not take values of best individuals
    sorted_indices = np.argsort(fitness_values)
    # Stay only best (natural selection)
    return population[sorted_indices[:num_best]]


# Evolution
def evolutionary_strategy(pop_size, dimensions, generations, func):
    population = create_population(pop_size, dimensions)
    best_fitness_values = []  # List of best fitness values in each generation
    history = []  # Record of the evolutionary path
    for generation in range(generations):
        fitness_values = np.array([func(ind) for ind in population])
        best_individuals = select_best(population, fitness_values)

        # Save the best solution from this generation
        best_solution = population[np.argmin(fitness_values)]
        history.append(best_solution)

        # Reproduction (mutation)
        new_population = []
        for individual in best_individuals:  # Loop over each of the best individuals
            # pop_size is the total number of  individuals I want in the new population
            for _ in range(pop_size // len(best_individuals)):  # Repeat for each "best individual"
                new_population.append(mutate(individual))
        population = np.array(new_population)

        # Rosenbrock and other functions are for benchmarks for minimization, smaller means better
        best_fitness = np.min(fitness_values)
        best_fitness_values.append(
            best_fitness
        )  # Add the best fitness from this generation
        print(f"Generation {generation}: Best fitness = {best_fitness}")

    return population[np.argmin(fitness_values)], best_fitness_values, np.array(history)

# Number of individuals (solutions) in the population at each generation
pop_size = 50
dimensions = 2
# Number of iterations of the evolutionary algorithm
generations = 100

best_solution_rosenbrock, best_fitness_rosenbrock, history_rosenbrock = (
    evolutionary_strategy(pop_size, dimensions, generations, rosenbrock)
)
best_solution_sphere, best_fitness_sphere, history_sphere = evolutionary_strategy(
    pop_size, dimensions, generations, sphere
)
best_solution_rastrigin, best_fitness_rastrigin, history_rastrigin = (
    evolutionary_strategy(pop_size, dimensions, generations, rastrigin)
)


print("Best solution for Rosenbrock:", best_solution_rosenbrock)
print("Best solution for Sphere:", best_solution_sphere)
print("Best solution for Rastrigin:", best_solution_rastrigin)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

plot_function(rosenbrock, axs[0], "Rosenbrock Function")
# 'ro-': plot red 'r' circles 'o' connected by lines '-'
axs[0].plot(
    # history_rosebrock dim: (generations, 2)
    # All across first dim
    history_rosenbrock[:, 0],
    # All across second dim
    history_rosenbrock[:, 1],
    "ro-",
    markersize=3,
    label="Trajectory",
)
axs[0].legend()

plot_function(sphere, axs[1], "Sphere Function")
axs[1].plot(
    history_sphere[:, 0], 
    history_sphere[:, 1], 
    "ro-", 
    markersize=3, 
    label="Trajectory"
)
axs[1].legend()

plot_function(rastrigin, axs[2], "Rastrigin Function")
axs[2].plot(
    history_rastrigin[:, 0],
    history_rastrigin[:, 1],
    "ro-",
    markersize=3,
    label="Trajectory",
)
axs[2].legend()

plt.tight_layout()
plt.show()
