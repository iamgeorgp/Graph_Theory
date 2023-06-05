from collections import defaultdict
import pandas as pd
import time
import math
import random




class Graph:
    # Конструктор графа
    def __init__(self, file_path, file_type):
       self.vertices = {}  # Словарь вершин
       #chosing the file type
       match file_type:
            case 'e':  # Данные - список ребер
                self.load_from_edges_list_file_path(file_path)
            case 'm':  # Данные - матрица смежности
                self.load_form_adjacency_matrix_file_path(file_path)
            case 'l':  # Данные - список смежности
                self.load_from_adjacency_list_file_path(file_path)
            case _:  # Невалидный ввод
                raise ValueError("Invalid file type.")

    # Добавление вершины
    def add_vertex(self, vertex):
        vertex = int(vertex)
        if vertex not in self.vertices:
            self.vertices[vertex] = {}

    # Добавление ребра
    def add_edge(self, start, end, weight):
        self.add_vertex(start); self.add_vertex(end)
        self.vertices[start][end] = weight

    # Загрузка из матрицы
    def load_form_adjacency_matrix_file_path(self, file_path):
        # with open(file_path, "r") as f:
        f = open(file_path)
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            row = line.strip().split()
            self.add_vertex(int(i))
            if len(row) != len(lines):
                raise ValueError("Invalid adjacency matrix.")
            for j, weight in enumerate(row, 1):
                self.add_vertex(int(j))
                if weight != "0":
                    self.add_edge(int(i), int(j), int(weight))

    # Загрузка из списка cмежности
    def load_from_adjacency_list_file_path(self, file_path):
        f = open(file_path)
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            row = line.strip().split()
            for j in row:
                self.add_edge(int(i), int(j), 1)

    # Загрузка из списка ребер
    def load_from_edges_list_file_path(self, file_path):
        f = open(file_path)
        lines = f.readlines()
        for number, line in enumerate(lines):
            vertex_vertex_weight = line.split()  # Разделение строки на Vi, Vj, W(Vi, Vj)
            if len(vertex_vertex_weight) == 2:  # Если только заданы вершины и не задан вес
                self.add_edge(int(vertex_vertex_weight[0]),
                              int(vertex_vertex_weight[1]),
                              1)
            else:  # Если заданы и вершины и вес между ними
                self.add_edge(int(vertex_vertex_weight[0]),
                              int(vertex_vertex_weight[1]),
                              int(vertex_vertex_weight[2]))

    # Возвращает матрицу смежности графа
    def adjacency_matrix(self):
        Size = len(self.vertices)

        matrix = [[0] * Size for _ in range(Size)]
        for each_start_vertex in self.vertices:
            each_start_vertex = int(each_start_vertex)
            for each_end_vertex in self.vertices[each_start_vertex]:
                each_end_vertex = int(each_end_vertex)

                matrix[each_start_vertex-1][(each_end_vertex-1)] = self.vertices[each_start_vertex][each_end_vertex]
        return matrix

    # Возвращает список ребер графа (при None) или для определенной вершине
    def list_of_edges(self, Vi=None):
        incident_edges = []
        if Vi is None:
            for each_start_vertex in self.vertices:
                for each_end_vertex in self.vertices[each_start_vertex]:
                    incident_edges.append((each_start_vertex, each_end_vertex))
        else:
            for each in self.vertices[Vi]:
                incident_edges.append((Vi, each))
        return incident_edges

    # Возвращает список смежных вершин (при None) или смежных Vi вершин
    def adjacency_list(self, Vi=None):
        if Vi is None:
            return self.vertices
        else:
            adjacent_vertex = []
            for each_end_vertex in self.vertices[Vi]:
                adjacent_vertex.append(each_end_vertex)
            return adjacent_vertex

    # Проверяет на наличие заданного ребра в графе
    def is_edge(self, Vi, Vj):
        return Vj in self.vertices.get(Vi, {})

    # Возвращает вес ребра между двумя вершинами
    def weight(self, Vi, Vj):
        raise self.vertices[Vi][Vj]

    # Проверка на ориентированность графа (true - ориантированный, false - неориентированный)
    def is_directed(self):
        for each_start_vertex in self.vertices:
            for each_end_vertex in self.vertices[each_start_vertex]:
                if not self.is_edge(each_end_vertex, each_start_vertex):
                    return True
        return False


"""---------------------------------------------
                 З А Д А Н И Я
---------------------------------------------"""
"""
Задание №1
Программа, рассчитывающая следующие характеристики графа/орграфа:
вектор степеней вершин, матрицу расстояний, диаметр, радиус,
множество центральных вершин (для графа), множество периферийных
вершин (для графа). Расчёт производится алгоритмом Флойда-Уоршелла

DONE
"""
def Floyd_Warshall(Graph):
    FW_matrix = Graph.adjacency_matrix()    # Инициализация матрицы расстояний
    N = len(Graph.vertices)                 # Размерность графа
    vert_degree = [0] * N                   # Список степеней вершин
    vert_plus = [0] * N
    for i in range(N):
        for j in range(N):
            if FW_matrix[i][j] == 0:        # Замена 0 на inf
                FW_matrix[i][j] = float("inf")
            else:
                vert_degree[i] += 1         # Подсчет степени вершины
                vert_plus[j]+=1


    # Заполнение матрицы расстояний
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if (i != j and FW_matrix[i][k] != float("inf") and FW_matrix[k][j] != float("inf")):
                    FW_matrix[i][j] = min(FW_matrix[i][j], FW_matrix[i][k] + FW_matrix[k][j]);

    # Обратная замена inf на 0
    for i in range(N):
        for j in range(N):
            if FW_matrix[i][j] == float("inf"):
                FW_matrix[i][j] = 0


    print('\nTASK №1')
    if Graph.is_directed():
        print('d+ = ', vert_plus)
        print('d- = ', vert_degree)
    else:
        print("degree = ", vert_degree)
    print('Matrix (Floyd-Warshall):', pd.DataFrame(FW_matrix, index=range(1, N+1), columns = range (1, N+1)), sep='\n')

    # Найдем диаметр графа (максимальное расстояние между любыми двумя вершинами)
    diameter = 0
    for i in range(N):
        for j in range(i + 1, N):
            if FW_matrix[i][j] > diameter:
                diameter = FW_matrix[i][j]
    print("Диаметр графа: ", diameter)

    # Найдем радиус графа (минимальное расстояние от одной вершины до самой удаленной)
    radius = float('inf')
    for i in range(N):
        max_distance = 0
        for j in range(N):
            if FW_matrix[i][j] > max_distance:
                max_distance = FW_matrix[i][j]
        if max_distance < radius:
            radius = max_distance
    print("Радиус графа: ", radius)

    # Найдем центральные вершины графа (вершины, от которых расстояние до других вершин минимально)
    central_vertices = []
    for i in range(N):
        max_distance = 0
        for j in range(N):
            if FW_matrix[i][j] > max_distance:
                max_distance = FW_matrix[i][j]
        if max_distance == radius:
            central_vertices.append(i+1)
    print("Центральные вершины графа: ", central_vertices)

    # Найдем периферийные вершины графа (вершины, от которых расстояние до других вершин максимально)
    peripheral_vertices = []
    for i in range(N):
        min_distance = 0
        for j in range(N):
            if FW_matrix[i][j] > min_distance:
                min_distance = FW_matrix[i][j]
        if min_distance == diameter:
            peripheral_vertices.append(i+1)
    print("Периферийные вершины графа: ", peripheral_vertices)


"""
Задание №2
Программа, определяющая связность. Для графа – связность, а также
количество и состав компонент связности. Для орграфа – сильную, слабую
связность, или несвязность. А также количество и состав компонент
связности и сильной связности. Для определения используется поиск в
ширину.

DONE
"""
def connectivity_of_graph(Graph):


    def is_strongly_connected(graph):
        # Проверяем граф на связность, начиная с первой вершины
        visited = set()
        dfss(graph, next(iter(graph)), visited)
        if len(visited) != len(graph):
            return False

        # Транспонируем граф
        transposed_graph = transpose_graph(graph)

        # Проверяем граф на связность, начиная с первой вершины
        visited = set()
        dfss(transposed_graph, next(iter(transposed_graph)), visited)
        if len(visited) != len(transposed_graph):
            return False

        # Граф сильно связный
        return True

    def is_connected(graph):
        start_vertex = next(iter(graph))
        visited = set()
        dfss(graph, start_vertex, visited)
        return len(visited) == len(graph)

    def find_connected_components(graph):

        # Обходим все вершины графа, запуская поиск в глубину из каждой непосещенной вершины
        visited = set()
        components = []
        for vertex in graph:
            if vertex not in visited:
                component = set()
                dfss(graph, vertex, visited, component)
                components.append(component)

        return components

    def dfss(graph, start_vertex, visited, component=None):
        visited.add(start_vertex)
        if component is not None:
             component.add(start_vertex)

        for neighbor in graph.get(start_vertex, []):
            if neighbor not in visited:
                dfss(graph, neighbor, visited, component)

    def transpose_graph(graph):
        transposed_graph = {vertex: {} for vertex in graph}
        for vertex in graph:
            for neighbor, weight in graph[vertex].items():
                transposed_graph.setdefault(neighbor, {})[vertex] = weight
        return transposed_graph

    def find_strongly_connected_components(graph):
        # Сначала запускаем поиск в глубину из каждой вершины графа,
        # чтобы найти все компоненты связности.
        visited = set()
        components = []
        for vertex in graph:
            if vertex not in visited:
                component = set()
                dfss(graph, vertex, visited, component)
                components.append(component)

        # Теперь транспонируем граф.
        transposed_graph = transpose_graph(graph)

        # Затем запускаем поиск в глубину из каждой вершины транспонированного графа,
        # в порядке обратном порядку обхода компонент связности,
        # чтобы найти все сильно связанные компоненты.
        visited.clear()
        strongly_connected_components = []
        for component in reversed(components):
            strongly_connected_component = set()
            for vertex in component:
                if vertex not in visited:
                    dfss(transposed_graph, vertex, visited, strongly_connected_component)
            if strongly_connected_component:
                strongly_connected_components.append(strongly_connected_component)

        return strongly_connected_components

    def find_strongly_in_weakly(graph):
        # Строим обратный граф.
        reversed_graph = reverse_graph(graph)

        # Выполняем первый обход в глубину на обратном графе.
        visited = set()
        stack = []
        for vertex in reversed_graph:
            if vertex not in visited:
                dfs_reverse(reversed_graph, vertex, visited, stack)

        # Выполняем второй обход в глубину на исходном графе, в порядке,
        # определенном в результате первого обхода.
        visited.clear()
        strongly_connected_components = []
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                strongly_connected_component = set()
                dfss(graph, vertex, visited, strongly_connected_component)
                strongly_connected_components.append(strongly_connected_component)

        return strongly_connected_components

    def reverse_graph(graph):
        reversed_graph = {}
        for vertex in graph:
            for neighbor in graph[vertex]:
                if neighbor not in reversed_graph:
                    reversed_graph[neighbor] = {}
                if vertex not in reversed_graph:
                    reversed_graph[vertex] = {}
                reversed_graph[neighbor][vertex] = graph[vertex][neighbor]
        return reversed_graph

    def dfs_reverse(graph, vertex, visited, stack):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs_reverse(graph, neighbor, visited, stack)
        stack.append(vertex)

    def find_weakly_connected_components(graph):
        # Сначала строим неориентированный граф, игнорируя направления ребер.
        undirected_graph = {}
        for vertex in graph:
            for neighbor in graph[vertex]:
                if neighbor not in undirected_graph:
                    undirected_graph[neighbor] = {}
                if vertex not in undirected_graph:
                    undirected_graph[vertex] = {}
                undirected_graph[vertex][neighbor] = graph[vertex][neighbor]
                undirected_graph[neighbor][vertex] = graph[vertex][neighbor]

        # Затем находим все компоненты связности неориентированного графа.
        visited = set()
        components = []
        for vertex in undirected_graph:
            if vertex not in visited:
                component = set()
                dfss(undirected_graph, vertex, visited, component)
                components.append(component)

        # Наконец, для каждой компоненты связности неориентированного графа
        # находим все соответствующие компоненты связности ориентированного графа.
        weakly_connected_components = []
        for component in components:
            weakly_connected_component = set()
            for vertex in component:
                if vertex in graph:
                    weakly_connected_component.add(vertex)
            if weakly_connected_component:
                weakly_connected_components.append(weakly_connected_component)

        return weakly_connected_components

    #-------------------------------------------------------------------------

    if Graph.is_directed():
        print("Граф ориентированный")
        if is_connected(Graph.adjacency_list()):
            print("Граф связный")
        else:
            print("Граф несвязный")
        if is_strongly_connected(Graph.adjacency_list()):
            print("Граф сильно связен")
            components = find_strongly_connected_components(Graph.adjacency_list())
            print("Количество компонент связности:", len(components))
            print("Состав компонент связности:")
            for component in components:
                print(component)
        else:
            print("Граф слабо связен")
            components = find_weakly_connected_components(Graph.adjacency_list())
            print("Количество компонент связности:", len(components))
            print("Состав компонент связности:")
            for component in components:
                print(component)

            components = find_strongly_in_weakly(Graph.adjacency_list())
            print("Количество компонент связности:", len(components))
            print("Состав компонент связности:")
            for component in components:
                print(component)


    else:
        print("Граф неориентированный")
        # определение количества компонент связности и состава каждой компоненты
        if is_connected(Graph.adjacency_list()):
            print("Граф связный")
        else:
            print("Граф несвязный")
        components = find_connected_components(Graph.adjacency_list())
        print("Количество компонент связности:", len(components))
        print("Состав компонент связности:")
        for component in components:
            print(component)

"""
Задание №3
Программа, находящая мосты и шарниры в графе. Для орграфа находятся 
мосты и шарниры в соотнесённом графе

DONE
"""

"            " \
"   DONE     " \
"            "

"""
Задание №4
Программа, находящая остовное дерево графа. Для орграфа находится
остовное дерево соотнесённого графа. Результатом является список рёбер
графа, входящих в остовное дерево и суммарный вес дерева.

DONE
"""
def Find_minimum_spanning_tree(Graph, type_of_algo):
    def prima_mst(graph):
        mst = []    # Остовное дерево MST
        # Создаем словарь, чтобы отслеживать, была ли вершина посещена
        visited = {vertex: False for vertex in graph}
        # Устанавливаем начальную вершину в качестве посещенной
        start_vertex = 1
        visited[start_vertex] = True
        mst_weight = 0
        while False in visited.values():
            min_edge = None
            # Ищем минимальный вес среди всех ребер, исходящих из посещенных вершин
            for vertex in visited:
                if visited[vertex]:
                    for neighbor in graph[vertex]:
                        if not visited[neighbor]:
                            weight = graph[vertex][neighbor]
                            if min_edge is None or weight < min_edge[2]:
                                min_edge = (vertex, neighbor, weight)   # Образуем минимальное ребро MST
            mst_weight += min_edge[2]       # Считаем вес MST
            mst.append(min_edge)            # Добавляем минимальное ребро в MST
            visited[min_edge[1]] = True     # Помечаем конечную вершину как посещенную
        return mst, mst_weight

    def kruskal_mst(graph):
        # Список ребер остовного дерева.
        mst = []

        # Множество вершин графа.
        vertices = set(graph.keys())

        # Множество множеств, каждое из которых представляет вершину графа.
        subsets = [{v} for v in vertices]

        # Сортируем все ребра графа по возрастанию весов.
        edges = []
        for v in graph:
            for w, weight in graph[v].items():
                edges.append((v, w, weight))
        edges.sort(key=lambda edge: edge[2])

        for edge in edges:
            # Находим множества, к которым принадлежат вершины ребра.
            v_subset = next((subset for subset in subsets if edge[0] in subset))
            w_subset = next((subset for subset in subsets if edge[1] in subset))

            # Если вершины не принадлежат одному множеству, добавляем ребро в остовное дерево.
            if v_subset != w_subset:
                mst.append((edge[0], edge[1], edge[2]))

                # Объединяем множества.
                v_subset.update(w_subset)
                subsets.remove(w_subset)

            # Если все вершины принадлежат одному множеству, остовное дерево найдено.
            if len(subsets) == 1:
                break

        # Вычисляем суммарный вес остовного дерева.
        total_weight = sum(weight for _, _, weight in mst)

        return mst, total_weight

    def boruvka_mst(gg):
        def adjacency_list(matrix):
            graph = defaultdict(list)
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    if matrix[i][j] > 0:
                        graph[i].append((j, matrix[i][j]))
                        graph[j].append((i, matrix[i][j]))
            return graph
        graph = adjacency_list(gg)
        parent = {}
        rank = {}
        mst = set()
        weight = 0

        # Функция для нахождения корня дерева
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        # Функция для объединения двух деревьев
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                else:
                    parent[root_u] = root_v
                    if rank[root_u] == rank[root_v]:
                        rank[root_v] += 1

        # Цикл до тех пор, пока есть несколько компонент связности
        while len(mst) < len(graph) - 1:
            # Инициализация для каждой компоненты связности
            for node in graph:
                parent[node] = node
                rank[node] = 0

            # Для каждой компоненты связности выбираем ребро минимального веса
            for node in graph:
                min_edge = float('inf')
                min_node = None
                for neighbor, weight in graph[node]:
                    if find(node) != find(neighbor) and weight < min_edge:
                        min_edge = weight
                        min_node = neighbor
                if min_node:
                    mst.add((node+1, min_node+1))
                    weight += gg[node][min_node]
                    union(node, min_node)

        return mst, weight

    match type_of_algo:
        case 'k':
            minimum_spanning_tree, weight_of_minimum_spanning_tree = kruskal_mst(Graph.adjacency_list())
        case 'p':
            minimum_spanning_tree, weight_of_minimum_spanning_tree = prima_mst(Graph.adjacency_list())
        case 'b':
            print();
            minimum_spanning_tree, weight_of_minimum_spanning_tree = boruvka_mst(Graph.adjacency_matrix())
        case 's':
            start = time.time()
            minimum_spanning_tree, weight_of_minimum_spanning_tree = kruskal_mst(Graph.adjacency_list())
            end = time.time()
            print("Kruskala timer: ", end-start)

            start = time.time()
            prima_mst(Graph.adjacency_list())
            end = time.time()
            print("Prima timer: ", end - start)

            start = time.time()
            boruvka_mst(Graph.adjacency_matrix())
            end = time.time()
            print("Boruvka timer: ", end - start)

        case _:
            raise "Unknown type of algorithm"
    print(f"Minimum spanning tree:\n{minimum_spanning_tree}\nWeight of spanning tree: {weight_of_minimum_spanning_tree}")

"""
Задание 5
Программа, находящая геодезическую цепь между двумя вершинами в 
графе. Поиск производится алгоритмом Дейкстры. Результатом работы 
является маршрут между вершинами, заданный списком рёбер, и длина 
маршрута.

DONE
"""
"            " \
"   DONE     " \
"            "

"""
Задание 6
Программа, рассчитывающая расстояние от указанной вершины до всех 
остальных вершин в графе. Результатом работы является перечисление 
пар вершин, и соответствующих расстояний между ними.

ALGO UNRIGHT
"""

def distances_from_vertex_to_other(Graph, start_vertex, type_of_algo):
    # Алгоритм Дейкстра
    def Dijkstra(Graph, start):
        Adj_List = Graph.adjacency_list()
        # инициализируем таблицу расстояний
        distances = {node: float('inf') for node in Adj_List}
        # расстояние до стартовой вершины равно 0
        distances[start] = 0
        # создаем множество посещенных вершин
        visited = set()

        while True:
            # находим ближайшую непосещенную вершину
            current_node = None
            for node in Adj_List:
                if node not in visited:
                    if current_node is None:
                        current_node = node
                    elif distances[node] < distances[current_node]:
                        current_node = node
            if current_node is None:
                break

            # обновляем расстояния до соседних вершин
            for neighbor, weight in Adj_List[current_node].items():
                if weight < 0:
                    return None
                distance = distances[current_node] + weight
                if distances[current_node] < 0:
                    return None
                if distance < distances[neighbor]:
                    distances[neighbor] = distance

            # помечаем вершину как посещенную
            visited.add(current_node)

        return distances

    # Алгоритм Беллмана-Форда-Мура
    def Bellman_Ford_Mur(graph, start):
        # инициализируем расстояния до всех вершин как бесконечность
        distances = {vertex: math.inf for vertex in graph}
        distances[start] = 0

        # проходим по графу V-1 раз, где V - количество вершин в графе
        for _ in range(len(graph) - 1):
            # проходим по каждому ребру и обновляем расстояния до соседних вершин,
            # если новый путь короче старого
            for vertex in graph:
                for neighbor in graph[vertex]:
                    if distances[vertex] + graph[vertex][neighbor] < distances[neighbor]:
                        distances[neighbor] = distances[vertex] + graph[vertex][neighbor]

        # проверяем наличие отрицательного цикла
        for vertex in graph:
            for neighbor in graph[vertex]:
                if distances[vertex] + graph[vertex][neighbor] < distances[neighbor]:
                    return None

        return distances

    # Алгоритм Левита
    def Levit(graph, start):
        # инициализация расстояний
        dist = {vertex: math.inf for vertex in graph}
        dist[start] = 0

        # инициализация очереди
        queue = [start]

        # инициализация множества посещенных вершин
        visited = set()

        # основной цикл алгоритма
        while queue:
            vertex = queue.pop(0)
            visited.add(vertex)

            # рассмотрение всех соседей текущей вершины
            for neighbor, weight in graph[vertex].items():
                # проверка на отрицательный цикл
                if dist[neighbor] > dist[vertex] + weight:
                    dist[neighbor] = dist[vertex] + weight

                    if neighbor not in queue:
                        queue.append(neighbor)

            # обработка всех вершин с минимальным расстоянием
            min_dist = math.inf
            for v in queue + list(visited):
                if dist[v] < min_dist and v not in visited:
                    min_dist = dist[v]
                    vertex = v

            # если не найдено более достижимых вершин, прерываем цикл
            if min_dist == math.inf:
                break

        return dist


    flag_neg_cycle = Bellman_Ford_Mur(Graph.adjacency_list(), start_vertex)



    # Определяем тип алгоритма
    match type_of_algo:
        case 'd':
            result = Dijkstra(Graph, start_vertex)
        case 'b':
            result = Bellman_Ford_Mur(Graph.adjacency_list(), start_vertex)
        case 't':
            result = Levit(Graph.adjacency_list(), start_vertex)
        case _:
            raise "Unknown type of algorithm"

    if flag_neg_cycle == None:
        print("Graph contains negative weight cycle")
    elif result == None:
        print("Graph contains negative weights")
    else:
        print("Graph does not contain edges with negative weight.\nShortest paths lengths:")
        print(result)
        for index in range(1, len(Graph.vertices) + 1):
            if index == start_vertex:
                continue
            else:
                print(f"{start_vertex} - {index}: {result[index]}")



"""
Задание №7
Программа, рассчитывающая расстояние между всеми парами вершин в 
графе. Поиск производится алгоритмом Джонсона. Результатом работы 
является перечисление пар вершин, и соответствующих расстояний между 
ними
"""
import heapq


def johnson_algorithm(graph):
    # добавление фиктивной вершины
    new_vertex = -1
    graph[new_vertex] = {}
    for vertex in graph.keys():
        graph[new_vertex][vertex] = 0

    # запуск алгоритма Беллмана-Форда
    distances = bellman_ford_algorithm(graph, new_vertex)
    if distances == None:
        print("neg cycle")
        return None

    # пересчет весов ребер
    for vertex in graph.keys():
        for neighbor in graph[vertex].keys():
            graph[vertex][neighbor] += distances[vertex] - distances[neighbor]

    # запуск алгоритма Дейкстры
    shortest_paths = {}
    for vertex in graph.keys():
        shortest_paths[vertex] = dijkstra_algorithm(graph, vertex)
        for neighbor in shortest_paths[vertex]:
            shortest_paths[vertex][neighbor] += distances[neighbor] - distances[vertex]

    # удаление фиктивной вершины
    del shortest_paths[new_vertex]
    for vertex in graph.keys():
        if vertex != new_vertex:

            del shortest_paths[vertex][new_vertex]
    #del shortest_paths[new_vertex]
    print("shortest path", shortest_paths)
    for vertex_from, item in shortest_paths.items():
        for vertex_to in item:
            if vertex_to != vertex_from and shortest_paths[vertex_from][vertex_to] != math.inf:
                print(f" {vertex_from} - {vertex_to}: {shortest_paths[vertex_from][vertex_to]}")
    return shortest_paths


def bellman_ford_algorithm(graph, start):
    distances = {vertex: float("inf") for vertex in graph.keys()}
    distances[start] = 0

    for _ in range(len(graph.keys()) - 1):
        for vertex in graph.keys():
            for neighbor, weight in graph[vertex].items():
                if distances[vertex] + weight < distances[neighbor]:
                    distances[neighbor] = distances[vertex] + weight

    # проверка на наличие отрицательных циклов
    for vertex in graph.keys():
        for neighbor, weight in graph[vertex].items():
            if distances[vertex] + weight < distances[neighbor]:
                return None



    return distances


def dijkstra_algorithm(graph, start):
    distances = {vertex: float("inf") for vertex in graph}
    distances[start] = 0

    heap = [(0, start)]
    while heap:
        (current_distance, current_vertex) = heapq.heappop(heap)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    return distances


def johnson(graph):
    def dijkstra(graph, start):
        distances = {vertex: math.inf for vertex in graph}
        distances[start] = 0
        heap = [(0, start)]
        while heap:
            (current_distance, current_vertex) = heapq.heappop(heap)
            if current_distance > distances[current_vertex]:
                continue
            for neighbor, weight in graph[current_vertex].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))
        return distances

    def bellman_ford(graph, start):
        distances = {vertex: float("inf") for vertex in graph.keys()}
        distances[start] = 0

        for _ in range(len(graph.keys()) - 1):
            for vertex in graph.keys():
                for neighbor, weight in graph[vertex].items():
                    if distances[vertex] + weight < distances[neighbor]:
                        distances[neighbor] = distances[vertex] + weight

        # проверка на наличие отрицательных циклов
        for vertex in graph.keys():
            for neighbor, weight in graph[vertex].items():
                if distances[vertex] + weight < distances[neighbor]:
                    return None

        # поиск предшественников
        predecessors = {vertex: None for vertex in graph.keys()}
        for _ in range(len(graph.keys())):
            for vertex in graph.keys():
                for neighbor, weight in graph[vertex].items():
                    if distances[vertex] != float("inf") and distances[vertex] + weight < distances[neighbor]:
                        predecessors[neighbor] = vertex

        return distances

    # добавляем дамми вершину ко всем и связываем ее
    new_vertex = "s"
    for vertex in graph:
        graph[vertex][new_vertex] = 0
    graph[new_vertex] = {vertex: 0 for vertex in graph}
    # находим растояния белманом-фордом из дамми вершины
    potentials = bellman_ford(graph, new_vertex)

    # удаляем дамми вершину и ребра
    del graph[new_vertex]
    for vertex in graph:
        del graph[vertex][new_vertex]
    # изменяем расстояния с пмоощью формулы вес + растояние U - расстояние V
    for u in graph:
        for v, w in graph[u].items():
            graph[u][v] = w + potentials[u] - potentials[v]
    # Теперь запускаем алгоритм дейкстра для поиска кратчайшего
    shortest_paths = {}
    for vertex in graph:
        shortest_paths[vertex] = dijkstra(graph, vertex)
        for v in shortest_paths[vertex]:
            shortest_paths[vertex][v] = shortest_paths[vertex][v] - potentials[vertex] + potentials[v]

    print("shortest path", shortest_paths)
    for vertex_from, item in shortest_paths.items():
        for vertex_to in item:
            if vertex_to != vertex_from and shortest_paths[vertex_from][vertex_to] != math.inf:
                print(f" {vertex_from} - {vertex_to}: {shortest_paths[vertex_from][vertex_to]}")



"""
Задание №10
Программа, рассчитывающая максимальный поток в сети. Расчёт 
выполняется алгоритмом Форда-Фалкерсона. Источник и сток 
определяются автоматически.

DONE
"""

def ford_fulkerson(Graph):
    # Функция поиска увеличивающего пути в графе
    def bfs(graph, source, sink, flow):
        queue = [(source, [source], float('inf'))]
        while queue:
            (vertex, path, capacity) = queue.pop(0)
            for next_vertex in graph[vertex]:
                if next_vertex not in path:
                    residual_capacity = graph[vertex][next_vertex] - flow[vertex][next_vertex]
                    if residual_capacity > 0:
                        new_capacity = min(capacity, residual_capacity)
                        new_path = path + [next_vertex]
                        if next_vertex == sink:
                            return (new_path, new_capacity)
                        else:
                            queue.append((next_vertex, new_path, new_capacity))
        return None
    # Функция определени истоков и стоков
    def find_source_and_sink(graph):
        in_edges = {vertex: set() for vertex in graph}
        out_edges = {vertex: set() for vertex in graph}

        for source in graph:
            for target, weight in graph[source].items():
                in_edges[target].add(source)
                out_edges[source].add(target)

        sources = [vertex for vertex in graph if not in_edges[vertex]]
        sinks = [vertex for vertex in graph if not out_edges[vertex]]

        return sources, sinks

    graph = Graph.adjacency_list()
    # Определение источника и стока
    source, sink = find_source_and_sink(graph)
    source, sink = source[0], sink[0]
    # Инициализируем поток нулевыми значениями
    flow = defaultdict(dict)
    for u in graph:
        for v in graph[u]:
            flow[u][v] = 0



    # Поиск увеличивающих путей в графе до тех пор, пока такие пути существуют
    while True:
        path = bfs(graph, source, sink, flow)
        if not path:
            break
        (vertices, capacity) = path
        for i in range(len(vertices) - 1):
            u = vertices[i]
            v = vertices[i + 1]
            if v not in flow:
                flow[v] = {}
            if u not in flow[v]:
                flow[v][u] = 0  # добавляем отсутствующий ключ в словарь
            flow[u][v] += capacity
            flow[v][u] -= capacity


    max_flow = 0
    for v in graph[source]:
        max_flow += flow[source][v]
    print(f'{max_flow} - maximum flow from {source} to {sink}')

    # Составляем список ребер и величин потоков по ним
    edges = []
    for u in graph:
        for v in graph[u]:
                edges.append((u, v, flow[u][v]))
                print(u, v, flow[u][v], "/", graph[u][v])





"""
Задание №11
Программа, находящая максимальные паросочетания. Перед этим
определяется, является ли граф двудольным. Результатом является список 
рёбер (паросочетаний). Для орграфа находятся паросочетания 
соотнесённого графа
"""
def couple_finder(Graph):
    def is_bipartite(graph):
        colors = {}  # словарь цветов вершин
        visited = set()  # множество посещенных вершин

        def dfs(node, color):
            colors[node] = color  # раскрасить вершину в цвет
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if not dfs(neighbor, 1 - color):
                        return False  # граф не является двудольным
                elif colors[neighbor] == color:
                    return False  # граф не является двудольным
            return True  # граф является двудольным

        for node in graph:
            if node not in visited:
                if not dfs(node, 0):
                    return False  # граф не является двудольным
        comp1=[]
        comp2=[]
        print('colors:', colors)
        for each in colors:
            if colors[each] == 1:
                comp1.append(each)
            else:
                comp2.append(each)
        print(f"component1:\n{comp1}\ncomponent2\n{comp2}")
        return True, comp1, comp2  # граф является двудольным

    bip_flag, comp1, comp2 = is_bipartite(Graph.adjacency_list())
    if bip_flag:
        print("Двудольный")
        if not Graph.is_directed():
            print("Граф неориентированный")
            adj_list_for_pairs = Graph.adjacency_list()
        else:
            print("Граф ориентированный")
            print("\tсписок смежности ориент")
            adj_list_for_pairs = Graph.adjacency_list()
            print(adj_list_for_pairs)
            # Создаем пустой список смежности соответственного графа
            corresponding_adjacency_list = {}

            # Перебираем каждую вершину и исходящие из нее ребра
            for vertex, edges in adj_list_for_pairs.items():
                # Проверяем, есть ли вершина в списке смежности соответственного графа
                if vertex not in corresponding_adjacency_list:
                    corresponding_adjacency_list[vertex] = {}

                # Перебираем каждое ребро и добавляем его в список смежности соответственного графа
                for destination, weight in edges.items():
                    # Добавляем прямое ребро
                    if destination not in corresponding_adjacency_list[vertex]:
                        corresponding_adjacency_list[vertex][destination] = weight
                    # Добавляем обратное ребро
                    if vertex not in corresponding_adjacency_list[destination]:
                        corresponding_adjacency_list[destination][vertex] = weight
            adj_list_for_pairs = corresponding_adjacency_list

            print("список смежности сооот\n", adj_list_for_pairs)

        start_vertex = 's'
        end_vertex = 't'

        adj_list_for_pairs[start_vertex] = {vertex: 1 for vertex in comp1}
        for each in comp2:
            adj_list_for_pairs[each][end_vertex] = 1
        print(adj_list_for_pairs)

        for vertex in comp1:
            for neighbour in comp2:
                if Graph.is_edge(vertex, neighbour):
                    adj_list_for_pairs[neighbour][vertex] = -1 * adj_list_for_pairs[vertex][neighbour]
        print(adj_list_for_pairs)

        # после предыдущих шагов получается граф, ориентирован следующим образом
        # s -> comp1 -> comp2 -> t
        def bfs(graph, source, sink, flow):
            queue = [(source, [source], float('inf'))]
            while queue:
                (vertex, path, capacity) = queue.pop(0)
                for next_vertex in graph[vertex]:
                    if next_vertex not in path:
                        residual_capacity = graph[vertex][next_vertex] - flow[vertex][next_vertex]
                        if residual_capacity > 0:
                            new_capacity = min(capacity, residual_capacity)
                            new_path = path + [next_vertex]
                            if next_vertex == sink:
                                return (new_path, new_capacity)
                            else:
                                queue.append((next_vertex, new_path, new_capacity))
            return None

        source = start_vertex
        sink = end_vertex
        graph = adj_list_for_pairs
        # Инициализируем поток нулевыми значениями
        flow = defaultdict(dict)
        for u in graph:
            for v in graph[u]:
                flow[u][v] = 0

        # Поиск увеличивающих путей в графе до тех пор, пока такие пути существуют
        while True:
            path = bfs(graph, source, sink, flow)
            if not path:
                break
            (vertices, capacity) = path
            for i in range(len(vertices) - 1):
                u = vertices[i]
                v = vertices[i + 1]
                if v not in flow:
                    flow[v] = {}
                if u not in flow[v]:
                    flow[v][u] = 0  # добавляем отсутствующий ключ в словарь
                flow[u][v] += capacity
                flow[v][u] -= capacity

        max_flow = 0
        for v in graph[source]:
            max_flow += flow[source][v]
        print(f'{max_flow} - maximum flow from {source} to {sink}')
        print(flow)

        del flow[start_vertex]
        del flow[end_vertex]

        for each in flow.keys():
            for vertex in flow[each]:
                if vertex != 't' and vertex != 's':
                    if flow[each][vertex] == 1:
                        print(f"{each}-{vertex}")



    else:
        print("Не двудольный")




# DONE: 1 2 3 4 5 6 7 10 11

g=Graph('TestsFiles/task9/matrix_t9_005.txt', 'm')

print(pd.DataFrame(g.adjacency_matrix(), index=range(1, len(g.vertices)+1), columns=range(1, len(g.vertices)+1)))
print(g.adjacency_list())



#distances_from_vertex_to_other(g, 5, 'b')
#couple_finder(g)

#distances_from_vertex_to_other(g, 2, 't')

#print(Floyd_Warshall(g))
#johnson_algorithm(g.adjacency_list())

#johnson(g.adjacency_list())






















#print(Hamilton_Way(g, 'a'))

##print(johnson(g.adjacency_list()))
#ford_fulkerson(g)


#connectivity_of_graph(g)
#distances_from_vertex_to_other(g, 6, 'b')
#Find_minimum_spanning_tree(g, 's')

#Find_minimum_spanning_tree(g, 's')

#print(Hamilton_Way(g, 'b'))




