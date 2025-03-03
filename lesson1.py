# funkcja celu/goal - albo maksymalizujemy albo minimalizujemy
# funkcja kosztu/straty/loss - funkcja celu ktora minimalizujemy
# funkcja oceny/przystosowania/fitness/zysku - funkcja celu ktora maksymalizujemy
from random import shuffle, randint


def random_solution(sample):
    current = sample.copy()
    shuffle(current)
    return current


def loss0(order):
    errors = 0
    for i in range(len(order) - 1):
        errors += 1 if order[i] > order[i + 1] else 0
    return errors


def loss1(order):
    errors = 0
    for i in range(int(len(order) / 2)):
        errors += 1 if order[i] > order[len(order) - i - 1] else 0
    return errors


def superloss(order):
    return loss0(order) + loss1(order)


def generate_neighbours(current):
    result = []
    for i in range(len(current)):
        neighbour = current.copy()
        neighbour[(i + 1) % len(current)], neighbour[i] = (
            current[i],
            current[(i + 1) % len(current)],
        )
        result.append(neighbour)
    return result


def hill_climbing(list_to_sort, loss, max_iterations):
    current = random_solution(list_to_sort)
    for i in range(max_iterations):
        neighbours = generate_neighbours(current)
        # print(loss(current), [ (loss(n),n) for n in neighbours ] )
        best_neighbour = neighbours[0]
        for j in neighbours[1:]:
            if loss(j) < loss(best_neighbour):
                best_neighbour = j
        if loss(best_neighbour) < loss(current):
            current = best_neighbour
        else:
            return current, i
    return current, max_iterations


def hill_climbing_stochastic(list_to_sort, loss, max_iterations):
    current = random_solution(list_to_sort)
    for i in range(max_iterations):
        neighbours = generate_neighbours(current)
        neighbour = neighbours[randint(0, len(neighbours) - 1)]
        if loss(neighbour) <= loss(current):
            current = neighbour
        if loss(current) == 0:
            return current, i
    return current, max_iterations


def random_probe(list_to_sort, loss, max_iterations):
    current = random_solution(list_to_sort)
    for i in range(max_iterations):
        r = random_solution(current)
        if loss(r) < loss(current):
            current = r
        if loss(current) == 0:
            return current, i
        # print(current, loss(current))
    return current, max_iterations


def tabu_search(list_to_sort, loss, max_iterations, tabu_size=100):
    current = random_solution(list_to_sort)
    global_best = [current.copy()]
    tabu_list = [current.copy()]

    for i in range(max_iterations):
        neighbours = [n for n in generate_neighbours(current) if n not in tabu_list]
        if len(neighbours) == 0:
            break
        best_neighbour = neighbours[0]
        for j in neighbours[1:]:
            if loss(j) < loss(best_neighbour):
                best_neighbour = j
        current = best_neighbour
        tabu_list.append(best_neighbour)
        if len(tabu_list) > tabu_size:
            tabu_list = tabu_list[-tabu_size:]
        if loss(current) < loss(global_best[-1]):
            global_best.append(current)
        if loss(current) == 0:
            return current, i
    return global_best[-1], max_iterations


def main(list_to_sort):
    l = superloss
    best, i = random_probe(list_to_sort, l, 1000000)  # hill_climbing(list_to_sort, 5)
    print("random_probe", best, i, l(best))
    for r in range(4):
        best, i = hill_climbing(
            list_to_sort, l, 1000000
        )  # hill_climbing(list_to_sort, 5)
        print("hill_climbing", best, i, l(best))

    best, i = hill_climbing_stochastic(list_to_sort, l, 1000)
    print("hill_climbing_stochastic", best, i, l(best))

    best, i = tabu_search(list_to_sort, l, 1000, 200)
    print("tabu_search", best, i, l(best), 1000)

    return


if __name__ == "__main__":
    main([1, 2, 5, 1, 3, 2, 8, 2, 4, 9])
