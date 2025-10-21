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
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

start = (1,1)
end = (8,8)


map_grid = np.array(maze)
print('Grid shape =', map_grid.shape)

plt.imshow(map_grid, cmap='binary')
plt.plot(start[0], start[1], 'go', markersize=10, label='start')
plt.plot(end[0], end[1], 'ro', markersize=10, label='goal')# start and end points
plt.grid(visible='both', which='both')
plt.xticks(range(0,10,1))
plt.yticks(range(0,10,1))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.figure()
start_node = Node(None, start)
goal_node = Node(None, end)

dijkstra = Dijkstra(map_grid)
path_d, cost_d = dijkstra.search(start_node, goal_node)
print('Dijkstra Path:', path_d)
print('Dijkstra cost:', cost_d)

# visualize the path
plt = plot_maze(map_grid, path=path_d, algo='Dijkstra')

astar = AStar(map_grid, heuristic_type='manhattan')
path_a, cost_a = astar.search(start_node, goal_node)
print('A* Path:', path_a)
print('A* cost:', cost_a)

# visualize the path

plt.plot([x[1] for x in path_a], [x[0] for x in path_a], 'b', label='A*')
plt.legend(loc='best')

plt.show()
