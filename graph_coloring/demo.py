import random
import math
import operator
import matplotlib.pyplot as plt


# Funkcje do tworzenia drzewa
def generate_operator():
    return random.choice([operator.add, operator.sub, operator.mul, operator.truediv])


def generate_terminal():
    return random.choice([random.uniform(-1, 1), "x"])


def generate_tree(depth=6):
    if depth == 0:
        return generate_terminal()
    else:
        op = generate_operator()
        left = generate_tree(depth - 1)
        right = generate_tree(depth - 1)
        return (op, left, right)


# Funkcja oceny
def evaluate_tree(tree, x):
    if isinstance(tree, tuple):
        op, left, right = tree
        left_val = evaluate_tree(left, x)
        right_val = evaluate_tree(right, x)

        # Zapobiegamy dzieleniu przez zero
        if op == operator.truediv and right_val == 0:
            # print(f"Division by zero encountered! Returning inf.")
            return 100_000.0  # Zwróć bardzo dużą wartość zamiast dzielić przez zero
        return op(left_val, right_val)
    elif tree == "x":
        return x
    else:
        return tree


# Funkcja fitness (błąd średniokwadratowy)
def fitness(tree, data_points):
    total_error = 0
    for x, true_y in data_points:
        predicted_y = evaluate_tree(tree, x)
        total_error += (true_y - predicted_y) ** 2
    return total_error


# Algorytm genetyczny
def evolve_population(
    population, generations, data_points, mutation_rate=0.1, crossover_rate=0.7
):
    for generation in range(generations):
        # Selekcja (najlepsi indywiduali)
        sorted_population = sorted(population, key=lambda x: fitness(x, data_points))
        population = sorted_population[: len(population) // 2]

        # Krzyżowanie
        new_population = population.copy()
        while len(new_population) < len(population) * 2:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
                new_population.append(child)

        # Mutacja
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                mutate(new_population[i])

        population = new_population
        best_tree = sorted_population[0]
        print(
            f"Generation {generation}: Best fitness = {fitness(best_tree, data_points)}"
        )
    return sorted_population[0]


# Funkcje krzyżowania i mutacji
def crossover(tree1, tree2):
    if isinstance(tree1, tuple) and isinstance(tree2, tuple):
        if random.random() < 0.5:
            return (tree1[0], crossover(tree1[1], tree2[1]), tree1[2])
        else:
            return (tree1[0], tree1[1], crossover(tree1[2], tree2[2]))
    else:
        return random.choice([tree1, tree2])


def mutate(tree):
    if random.random() < 0.5:
        tree = generate_tree()
    return tree


# Tworzymy dane do nauki (sin(x))
data_points = [(x / 100.0, math.sin(x / 100.0)) for x in range(1000)]

# Inicjalizacja populacji
population = [generate_tree() for _ in range(10)]

# Ewolucja
best_tree = evolve_population(population, 100, data_points)
print(best_tree)

# Wizualizacja danych i najlepszego drzewa
x_values = [x / 100.0 for x in range(1000)]
true_y_values = [math.sin(x / 100.0) for x in range(1000)]
predicted_y_values = [evaluate_tree(best_tree, x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, true_y_values, label="Rzeczywista funkcja (sin(x))")
plt.plot(x_values, predicted_y_values, label="Aproksymacja drzewa")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Aproksymacja funkcji sin(x)")
plt.legend()
plt.grid(True)
plt.show()

# Wizualizacja pierwszego drzewa z populacji
first_tree = population[0]
first_predicted_y_values = [evaluate_tree(first_tree, x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, true_y_values, label="Rzeczywista funkcja (sin(x))")
plt.plot(x_values, first_predicted_y_values, label="Pierwsze drzewo z populacji")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Pierwsze drzewo z populacji vs sin(x)")
plt.legend()
plt.grid(True)
plt.show()
