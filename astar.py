import pandas as pd
import math
class Cell:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position    # Координаты (x, y)
        self.g = 0  # расстояние между текущим узлом и начальным узлом
        self.h = 0  # эвристически оцененное расстояние от текущего узла до конечного узла
        self.f = 0  # общая стоимость узла f = g + h
    def __eq__(self, other): # eq = equality
        return self.position == other.position
class Map:
    def __init__(self, file_path):
        self.map = []
        self.load_map(file_path)

    def load_map(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                row = [int(height) for height in line.strip().split()]
                self.map.append(row)

    def neighbors(self, cell):
        neighbors_list = []
        if cell.position[0] > 0:
            neighbors_list.append(Cell(cell, (cell.position[0] - 1, cell.position[1])))
        if cell.position[0] < len(self.map[0]) - 1:
            neighbors_list.append(Cell(cell, (cell.position[0] + 1, cell.position[1])))
        if cell.position[1] > 0:
            neighbors_list.append(Cell(cell, (cell.position[0], cell.position[1] - 1)))
        if cell.position[1] < len(self.map) - 1:
            neighbors_list.append(Cell(cell, (cell.position[0], cell.position[1] + 1)))
        return neighbors_list


def astar(Map, start, end, type_of_heu):
    map = Map.map
    # создаем начальную и конечную вершины
    start_node = Cell(None, start); start_node.g = start_node.h = start_node.f = 0
    end_node = Cell(None, end);     end_node.g = end_node.h = end_node.f = 0
    # инициализация списков закрытых и открытх узлов
    open_list = [];    closed_list = []

    def Manhattan(x,y,X,Y):
        return abs(x-X) + abs(y-Y)
    def Chebyshev(x,y,X,Y):
        return max(abs(x-X), abs(y-Y))
    def Evklid(x,y,X,Y):
        return math.sqrt(pow((x-X), 2) + pow((y-Y), 2))

    open_list.append(start_node) # начинаем с начальной вершины
    while len(open_list) > 0:

        # получаем текущую открытую вершину
        current_node = open_list[0]
        current_index = 0

        # выбор вершины с минимальным значением f из спсика открытых узлов
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # выбранная вершина становится закрытой
        open_list.pop(current_index);   closed_list.append(current_node)

        # проверка на конец
        if current_node == end_node:
            path = []
            current = current_node

            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] , (len(closed_list)/(pow(len(map), 2)) * 100) # возвращает реверснутый путь

        # # поиск соседей клетки (слева/справа/сверху/снизу)
        # neighbours = []
        # for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        #     # получение соседей по координатам
        #     #       (0,-1)
        #     #          |
        #     #(-1,0)--(0,0)--(1,0)
        #     #          |
        #     #        (0,1)
        #     #(NewX, NewY) = (X+Смещение, Y+Смещение)
        #     node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
        #
        #     # проверка на не выход за границы
        #     if node_position[0] > (len(map) - 1) or node_position[0] < 0 or node_position[1] > (len(map[len(map)-1]) -1) or node_position[1] < 0:
        #         continue
        #     new_node = Cell(current_node, node_position)    # создание новой вершины
        #     neighbours.append(new_node)
        # проходимся по соседям
        neighboors = Map.neighbors(current_node)
        for neigh in neighboors:
            m = []
            for each in neighboors:
                m.append(each.position)
            # сосед - закрытая вершина
            for closed_neigh in closed_list:
                if neigh == closed_neigh:
                    continue
            # определение значений g, h, f
            neigh.g = current_node.g + abs(map[current_node.position[0]][current_node.position[1]] - map[neigh.position[0]][neigh.position[1]]) +1
            # neigh.h = abs(neigh.position[0] - end_node.position[0]) + (abs(neigh.position[1] - end_node.position[1]))
            match type_of_heu:
                case 'M':
                    neigh.h = Manhattan(neigh.position[0], neigh.position[1], end_node.position[0], end_node.position[1])
                case 'CH':
                    neigh.h = Chebyshev(neigh.position[0], neigh.position[1], end_node.position[0], end_node.position[1])
                case 'E':
                    neigh.h = Evklid(neigh.position[0], neigh.position[1], end_node.position[0], end_node.position[1])
                case 'D':
                    neigh.h = 0
            neigh.f = neigh.g + neigh.h

            if neigh not in [open_node for open_node in open_list]:
                open_list.append(neigh)
            # добавление в список откртых узлов
            #open_list.append(neigh)


def main():

    mmap = Map('TestsFiles/task8/map_001.txt')
    print(pd.DataFrame(mmap.map))
    start = (9, 0)
    end = (0, 9)
    path, perc = astar(mmap, start, end, 'CH')
    # for each_type_of_heu in ['M', 'E', 'CH', 'D']:
    #     path, perc = astar(mmap, start, end, each_type_of_heu)
    #     print(f'Path:\n{path}\n{perc}%')




if __name__ == '__main__':
    main()
