"""
A* Algorithm implementation using a graph class. 
"""
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heapify, heappush, heappop
from graph_visualization import show_wgraph, show_wpath

class Graph:
    def __init__(self, graph_dict: dict = None):
        """
        Initializes a graph object.
        If no dictionary is provided, an empty graph is created.
        """
        self.graph = graph_dict if graph_dict is not None else {}

    def add_edge(self, node1, node2, weight):
        """ Adds an edge to the graph. """
        if node1 not in self.graph:
            self.graph[node1] = {}
        self.graph[node1][node2] = weight

    def astar_path(self, source: str, target: str, heuristic: dict):
        """
        Finds the shortest path from source to target using A* algorithm.

        Args:
            source (str): The starting node.
            target (str): The target node.
            heuristic (dict): A dictionary of heuristic costs for each node to the target.

        Returns:
            tuple: A tuple containing the path (list) and the total cost (float).
                   Returns (None, float('inf')) if no path is found.
        """
        # Priority queue: (f_score, node). f_score = g_score + h_score
        pq = [(heuristic[source], source)]
        heapify(pq)

        # g_score: cost from source to the current node
        g_scores = {node: float("inf") for node in self.graph}
        g_scores[source] = 0

        # came_from: to reconstruct the path
        predecessors = {node: None for node in self.graph}

        while pq:
            _, current_node = heappop(pq)

            if current_node == target:
                # Reconstruct path
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = predecessors[current_node]
                path.reverse()
                return path, g_scores[target]

            if current_node not in self.graph: # Handle nodes that are destinations but not sources
                continue

            for neighbor, weight in self.graph[current_node].items():
                tentative_g_score = g_scores[current_node] + weight

                if tentative_g_score < g_scores[neighbor]:
                    predecessors[neighbor] = current_node
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic[neighbor]
                    heappush(pq, (f_score, neighbor))

        return None, float('inf') # Path not found

    def get_nodes(self):
        return list(self.graph.keys())

    def get_edges(self):
        edges = []
        for node in self.graph:
            for neighbor, weight in self.graph[node].items():
                edges.append((node, neighbor, weight))
        return edges

if __name__ == "__main__":

    # Using the same graph structure as astar_dict.py for consistency
    my_graph_dict = {
        'A': {'B':2, 'C':3},
        'B': {'A':2, 'D':3, 'E':1},
        'C': {'A':3, 'F':2},
        'D': {'B':3, 'E':1},
        'E': {'B':1, 'D':1, 'F':1},
        'F': {'C':2, 'E':1}
    }

    heuristic = { 'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 1, 'F': 0 }

    # Create graph object
    g = Graph(my_graph_dict)

    # Find shortest path
    path, dist = g.astar_path("A", "F", heuristic)
    print('Shortest Path using A*: ', path)
    print('Total cost: ', dist)

    # --- Visualization ---
    nxG = nx.Graph()
    nodes = g.get_nodes()
    weighted_edges = g.get_edges()
    nxG.add_nodes_from(nodes)
    nxG.add_weighted_edges_from(weighted_edges)

    show_wgraph(nxG, node_labels=heuristic)
    show_wpath(nxG, path, node_labels=heuristic)
    plt.show()
