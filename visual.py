import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation


def reconstruct_path(cameFrom, current):
    total_path = [current]
    while sum(cameFrom[current[0]][current[1]]) > -1:
        current = cameFrom[current[0]][current[1]]
        total_path.append(current)
    total_path.reverse()
    return total_path


def astar(start, goal, M, web):
    if M[start[0]][start[1]]:
        return []
    N = len(M)
    if (start[0] < 0) or (start[1] < 0) or (start[0] >= N) or (start[1] >= N):
        return []
    if (goal[0] < 0) or (goal[1] < 0) or (goal[0] >= N) or (goal[1] >= N):
        return []

    ### Steps Right, Left, Down, Up
    g = lambda x: abs(x[0]) + abs(x[1])
    Steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    ### Steps Right, Left, Down, Up, DR, DL, UR, UL
    #     g = lambda x: abs(x[0]) + abs(x[1]) + 0.5* abs(x[0]*x[1])
    #     Steps = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    ### Heuristics:
    #     h = lambda y:0 ### No heuristic - Dijkstra
    h = lambda y: np.sqrt((goal[0] - y[0]) ** 2 + (goal[1] - y[1]) ** 2)  ### Euclid heuristic
    #     h = lambda y: max(abs(goal[0] - y[0]),abs(goal[1] - y[1])) ### Chebyshev heuristic
    #     h = lambda y: (abs(goal[0] - y[0]) + abs(goal[1] - y[1])) ### Manhattan heuristic

    openSet = [start]
    cameFrom = [[(-1, -1)] * N for i in range(N)]
    gscore = [[10 ** 10] * N for i in range(N)]
    gscore[start[0]][start[1]] = 0
    fscore = [[10 ** 10] * N for i in range(N)]
    fscore[start[0]][start[1]] = h(start)
    while openSet:
        spider = []
        current = min([(x, fscore[x[0]][x[1]]) for x in openSet], key=lambda x_fscore: x_fscore[1])[0]
        spider.append((current[0], current[1]))
        #         plt.plot(current[1], current[0], 'y s')
        #         print("------------------Next-----------------------")
        #         print(f'Current = {current}')
        #         print(f'gscore = {gscore[current[0]][current[1]]}')
        if (current[0] == goal[0]) and (current[1] == goal[1]):
            return reconstruct_path(cameFrom, current)
        openSet.remove(current)
        for step in Steps:
            neighbor = (current[0] + step[0], current[1] + step[1])
            if (neighbor[0] >= N) or (neighbor[1] >= N) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            if M[neighbor[0]][neighbor[1]]:
                continue

            spider.append((neighbor[0], neighbor[1]))
            #             print(f'Neghtbor = {neighbor}')
            #             print(f'gscore = {gscore[current[0]][current[1]]}')
            tentative_gscore = gscore[current[0]][current[1]] + g(step)
            #             print(f'tentative_gscore = {tentative_gscore}')
            if tentative_gscore < gscore[neighbor[0]][neighbor[1]]:
                cameFrom[neighbor[0]][neighbor[1]] = current
                gscore[neighbor[0]][neighbor[1]] = tentative_gscore
                fscore[neighbor[0]][neighbor[1]] = gscore[neighbor[0]][neighbor[1]] + h(neighbor)
                if neighbor not in openSet:
                    openSet.append(neighbor)
        web.append(spider)
    return []


def show_path(path, start, end, M, web):
    fig = plt.figure(figsize=[12., 12.])
    ax = fig.add_subplot(1, 1, 1)
    plt.spy(M)

    ax.minorticks_on()
    positions = np.arange(0, len(M)).tolist()

    @ticker.FuncFormatter
    def ticks_formatter(x, pos):
        return f'{x}'

    ax.xaxis.set_minor_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_minor_formatter(ticks_formatter)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_minor_formatter(ticks_formatter)

    ax.set_xticks(np.arange(-0.5, len(M), 1))
    ax.set_yticks(np.arange(-0.5, len(M), 1))
    ax.tick_params(axis='both',
                   which='minor',
                   length=0.0,
                   width=0.0,
                   labelbottom=True,
                   labeltop=True,
                   labelleft=True,
                   labelright=True)
    ax.tick_params(axis='both',  # Применяем параметры к обеим осям
                   which='major',  # Применяем параметры к основным делениям
                   length=0.0,  # Длина делений
                   width=0.0,  # Ширина делений
                   labelbottom=False,  # Рисуем подписи снизу
                   labeltop=False,  # сверху
                   labelleft=False,  # слева
                   labelright=False)

    ax.grid(which='major',
            color='k',
            linestyle=':',
            lw=1)

    xdata, ydata = [], []  # Для анимации просмотренных точек
    graph, = plt.plot(xdata, ydata, 'y s')

    def path_find(step):
        for i in range(1, len(step)):
            xdata.append(step[0][1])
            xdata.append(step[i][1])
            ydata.append(step[0][0])
            ydata.append(step[i][0])
        graph.set_data(xdata, ydata)
        return graph,

    ani = animation.FuncAnimation(fig, path_find, frames=web, interval=1, blit=False, repeat=False)
    if path:
        for k in range(len(path) - 1):
            plt.plot([path[k][1], path[k + 1][1]], [path[k][0], path[k + 1][0]], 'r:,')
        plt.plot(start[1], start[0], 'g:o')
        plt.plot(end[1], end[0], 'r:X')
    else:
        plt.plot(start[1], start[0], 'r:x')

    plt.show()
    return


def a_maze(M, start, aim):
    web = []
    path = astar(start, aim, M, web)
    print(path)
    print(f'Length of path: {len(path)}')
    show_path(path, start, aim, M, web)


M = [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
     [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
     [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]]


start = (1, 1)
aim = (4, 14)
a_maze(M, start, aim)