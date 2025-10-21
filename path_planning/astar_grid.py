import numpy as np
import matplotlib.pyplot as plt
from graph_visualization import plot_maze
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # (x,y) position in the grid

        self.g = 0  # actual cost
        self.h = 0  # heuristic cost
        self.f = self.g + self.h # total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):        # used for sorting
        return self.f < other.f
    
class AStar:
    def __init__(self, map_grid, heuristic_type='euclidean'):
        self.open = []
        self.closed = []
        self.map_grid = map_grid
        self.heuristic_type = heuristic_type

    def search(self, start_node, goal_node):
        # add start node to the open list
        self.open.append(start_node)

        while self.open:
            # find the node with the lowest f value
            self.open.sort()
            current_node = self.open.pop(0)

            # add the current node to the closed list
            self.closed.append(current_node)

            if current_node == goal_node:
                # reached the goal node
                return self.reconstruct_path(current_node) # Returns path and cost

            # check every neigbor
            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                # skip the neighbor if it is already visited
                if neighbor in self.closed:
                    continue

                # compute cost to travel to this node from current node
                g_cost = current_node.g + 1
                h_cost = self.heuristic(neighbor, goal_node, type=self.heuristic_type)
                f_cost = g_cost + h_cost

                # check if we found a cheaper path
                if neighbor in self.open:
                    if neighbor.f > f_cost:
                        self.update_node(neighbor, g_cost, h_cost)
                        continue
                else:
                    self.update_node(neighbor, g_cost, h_cost)
                    self.open.append(neighbor)

        # no path found
        return None, float('inf')

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

    def heuristic(self, start, end, type='euclidean'): # Using squared Euclidean distance
        if type == 'euclidean':
            return np.linalg.norm(np.array(start.position) - np.array(end.position))
        elif type == 'manhattan':
            return abs(start.position[0] - end.position[0]) + abs(start.position[1] - end.position[1])
        else:
            raise ValueError("Unknown heuristic type")

    def reconstruct_path(self, goal_node):
        cost = goal_node.g
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1], cost # reverse the path and return with cost

    def update_node(self, node, g_cost, h_cost):
        node.g = g_cost
        node.h = h_cost
        node.f = g_cost + h_cost


if __name__ == "__main__":
    
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

    a_star = AStar(map_grid, heuristic_type='manhattan')
    path, cost = a_star.search(start_node, goal_node)
    
    if path:
        print("Path found:", path)
        print("Total cost:", cost)
        # visualize the path
        plot_maze(map_grid, path, title="A* Path (Grid)")
        plt.show()
    else:
        print("No path found.")
