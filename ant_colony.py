import random
class Ant:
    def __init__(self):
        self.visited_vertices = []

def find_hamiltonian_cycle(graph, num_ants, num_iterations, alpha, beta, evaporation_rate):
    def initialize_ants(num_ants, num_vertices):
        ants = []
        for _ in range(num_ants):
            ants.append(Ant())
        return ants
    num_vertices = len(graph)
    pheromone = [[1.0] * num_vertices for _ in range(num_vertices)]  # инициализация феромонов
    best_path = None
    best_cost = float('inf')
    for _ in range(num_iterations):                     # прохождение по итерациям
        ants = initialize_ants(num_ants, num_vertices)  # инициализация муравьев
        for ant in ants:                                # для каждого муравья:
            visited = [False] * num_vertices            # создание массива посещенных
            current_vertex = random.randint(0, num_vertices - 1)
            visited[current_vertex] = True
            ant.visited_vertices.append(current_vertex)
            for _ in range(num_vertices - 1):
                next_vertex = select_next_vertex(graph, pheromone, current_vertex, visited, alpha, beta)
                ant.visited_vertices.append(next_vertex)
                visited[next_vertex] = True
                current_vertex = next_vertex
            ant.visited_vertices.append(ant.visited_vertices[0])  # возврат в исходную вершину
            ant_cost = calculate_path_cost(graph, ant.visited_vertices)  # вычисление стоимости пути
            if ant_cost < best_cost: # Проверка на лучший путь
                best_path = ant.visited_vertices
                best_cost = ant_cost

            update_pheromone(pheromone, ant.visited_vertices, ant_cost, evaporation_rate)  # обновление феромонов

    return best_path



def select_next_vertex(graph, pheromone, current_vertex, visited, alpha, beta):
    # Создание непосещеных вершин с помощью visited == False
    unvisited_vertices = [index for index, val in enumerate(visited) if not val]
    # Список пар (вершина, вероятность)
    probabilities = []
    for vertex in unvisited_vertices:
        pheromone_level = pheromone[current_vertex][vertex]
        visibility = 1 / graph[current_vertex][vertex]
        probability = pheromone_level ** alpha * visibility ** beta
        probabilities.append((vertex, probability))
    total_probability = sum(probability for _, probability in probabilities)
    #             T(t)^(alfa) * (1/D)^(beta)
    #   P(t) = --------------------------------
    #           sum(T(t)^(alfa) * (1/D)^(beta))
    probabilities = [(vertex, probability / total_probability) for vertex, probability in probabilities]
    # Сортировка в порядке убывания для выбора следующей вершины с наиб вер-тью
    probabilities.sort(key=lambda x: x[1], reverse=True)
    rand = random.uniform(0, 1)
    cumulative_probability = 0
    for vertex, probability in probabilities:
        cumulative_probability += probability
        if rand <= cumulative_probability:
            return vertex
def calculate_path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    return cost
def update_pheromone(pheromone, path, cost, evaporation_rate):
    for i in range(len(path) - 1):
        pheromone[path[i]][path[i + 1]] = (1 - evaporation_rate) * pheromone[path[i]][path[i + 1]] + 1 / cost




# Пример графа в виде матрицы смежности
graph = []
def load_map(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(height) for height in line.strip().split()]
            graph.append(row)
load_map('TestsFiles/task9/matrix_t9_010.txt')

num_ants = len(graph)   # количество муравьев
num_iterations = 51     # количество итераций
alpha = 1               # коэффициент влияния феромонов
beta = 2                # коэффициент влияния видимости
evaporation_rate = 0.5  # коэффициент испарения феромонов

# Поиск гамильтонова цикла методом муравьиной колонии
result = find_hamiltonian_cycle(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
print("Best path:", result)
length = 0
for i in range(len(result)-1):
    print(f'{result[i]+1}-{result[i+1]+1} {graph[result[i]][result[i+1]]}')
    length += graph[result[i]][result[i+1]]
print('Length', length + graph[result[-1]][result[0]])
