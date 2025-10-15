import numpy as np
import matplotlib.pyplot as plt
from graph_visualization import plot_maze

class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # (x,y) position in the grid

        self.g = 0  # actual cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):        # used for sorting
        return self.g < other.g

class Dijkstra:
    def __init__(self, map_grid):
        self.open = []
        self.closed = []
        self.map_grid = map_grid

    def search(self, start_node, goal_node):
        # add start node to the open list
        self.open.append(start_node)

        while self.open:
            # find the node with the lowest g value
            self.open.sort()
            current_node = self.open.pop(0)

            # add the current node to the closed list
            self.closed.append(current_node)

            if current_node == goal_node:
                # reached the goal node
                return self.reconstruct_path(current_node)

            # check every neigbor
            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                # skip the neighbor if it is already visited
                if neighbor in self.closed:
                    continue

                # compute cost to travel to this node from current node
                g_cost = current_node.g + 1

                # check if we found a cheaper path
                if neighbor in self.open:
                    if neighbor.g > g_cost:
                        self.update_node(neighbor, g_cost)
                        continue
                else:
                    self.update_node(neighbor, g_cost)
                    self.open.append(neighbor)

        # no path found
        return None

    def get_neighbors(self, node):
        neighbors = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (node.position[0] + new_position[0],
                             node.position[1] + new_position[1])

            # check if node_position is valid
            if 0 <= node_position[0] < self.map_grid.shape[0] and \
                    0 <= node_position[1] < self.map_grid.shape[1]:

                # check if new node is traversable
                if self.map_grid[node_position] == 0: # no obstacle
                    neighbors.append(Node(parent=node,
                                        position=node_position))
        return neighbors # list of nodes

    def reconstruct_path(self, goal_node):
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1] # reverse the path

    def update_node(self, node, g_cost):
        node.g = g_cost


if __name__ == "__main__":
    
    # grid map
    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (0,0)
    end = (8,8)


    map_grid = np.array(maze)
    print('Grid shape =', map_grid.shape)

    start_node = Node(None, start)
    goal_node = Node(None, end)

    dijkstra = Dijkstra(map_grid)
    path = dijkstra.search(start_node, goal_node)
    print(path)

    # visualize the path
    plot_maze(map_grid, path)
    plt.show()
