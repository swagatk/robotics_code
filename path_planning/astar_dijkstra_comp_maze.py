"""
visualizing and comparing A* and Dijkstra algorithms
"""
from astar_grid import AStar 
from dijkstra_grid import Dijkstra, Node 
import numpy as np
import matplotlib.pyplot as plt
from graph_visualization import plot_maze
import networkx as nx


# grid map
maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

start = (0,0)
end = (3,8)


map_grid = np.array(maze)
print('Grid shape =', map_grid.shape)

start_node = Node(None, start)
goal_node = Node(None, end)

dijkstra = Dijkstra(map_grid)
path_d = dijkstra.search(start_node, goal_node)
print(path_d)

# visualize the path
plt = plot_maze(map_grid, path=path_d, algo='Dijkstra')

astar = AStar(map_grid)
path_a = astar.search(start_node, goal_node)

plt.plot([x[1] for x in path_a], [x[0] for x in path_a], 'b', label='A*')
plt.legend(loc='best')

plt.show()
