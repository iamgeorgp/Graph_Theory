import math
maxsize = float('inf')
# Первый мин. ребро из вершины V
def firstMin(adj, V):
    min = maxsize
    for k in range(N):
        if adj[V][k] < min and V != k:
            min = adj[V][k]
    return min
# Второе мин ребро из вершины V
def secondMin(adj, V):
    first, second = maxsize, maxsize
    for j in range(N):
        if V == j:
            continue
        if adj[V][j] <= first:
            second = first
            first = adj[V][j]
        elif (adj[V][j] <= second and
              adj[V][j] != first):
            second = adj[V][j]
    return second
# Запись в итоговый ответ текущего результата
def copyToFinal(curr_path):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]
def PathRecording(
        adj,            # матрица смежности
        curr_bound,     # нижняя граница корневого узла
        curr_weight,    # текуший вес
        level,          # уровень перемещения
        curr_path,      # путь решения
        visited         # посещеные узлы
        ):
    global final_res
    if level == len(adj):  # проверка на полное прохождение
        # проверка, если есть ребро от последней до первой
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            #cur res - теперь общий вес решения
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res: # проверка на мин цикл
                copyToFinal(curr_path)
                final_res = curr_res
        return
    # построение решения
    for i in range(N):
        # рассматриваем непосещеную вершину + она имеет вес
        if (adj[curr_path[level - 1]][i] != 0 and visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]
            # оценка нижней границы узла
            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)
            # фактическая нижняя оценка curr_bound + curr_weight
            # для посещенного узла;
            # соответсвенно если текущая нижняя оценка
            # меньше наим стоимости, то проходим дальше.
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True
                # Запускаем для следующей вершины, увеличивая уровень
                PathRecording(adj,
                       curr_bound,
                       curr_weight,
                       level + 1,
                       curr_path,
                       visited
                       )
            # иначе возвращаем предыдущие значения
            curr_weight -= adj[curr_path[level - 1]][i]; curr_bound = temp
            # также сбрасываем состояние посещенных вершин
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True
def Branch_and_Bound(adj):
    N = len(adj); curr_bound = 0; curr_path = [-1] * (N + 1); visited = [False] * N
    # вычислить начальную границу
    for i in range(N):
        curr_bound += (firstMin(adj, i) + secondMin(adj, i))
    # Округление нижней границы до целого числа
    curr_bound = math.ceil(curr_bound / 2)
    # иницализация начала
    visited[0] = True; curr_path[0] = 0
    # текущий вес = 0, а уровень = 1
    PathRecording(adj, curr_bound, 0, 1, curr_path, visited)
# Пример графа в виде матрицы смежности
adj = []
def load_map(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(height) for height in line.strip().split()]
            adj.append(row)
load_map('TestsFiles/task9/matrix_t9_008.txt')
N = len(adj); final_path = [None] * (N + 1); visited = [False] * N; final_res = maxsize
Branch_and_Bound(adj)
print("Minimum cost :", final_res)
print("Path Taken : ")
for i in range(len(final_path)-1):
    print(f'{final_path[i]+1} - {final_path[i+1]+1}: ({adj[final_path[i]][final_path[i+1]]})')