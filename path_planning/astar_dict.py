"""
Implement A* algorithm for a graph represented as a dictionary.
"""
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from graph_visualization import show_wgraph, show_wpath

def astar(graph, heuristic, start_node, end_node):
    f_distance = {node: float('inf') for node in graph}
    f_distance[start_node] = 0
    g_distance = {node: float('inf') for node in graph}
    g_distance[start_node] = 0

    came_from = {node: None for node in graph}
    came_from[start_node] = None
    open_set = [(0, start_node)] # priority Q
    while open_set:
        current_f_distance, current_node = heapq.heappop(open_set)

        if current_node == end_node:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            return path, g_distance[end_node]

        for next_node, weight in graph[current_node].items():
            temp_g_distance = g_distance[current_node] + weight

            if temp_g_distance < g_distance[next_node]:
                g_distance[next_node] = temp_g_distance
                f_distance[next_node] = g_distance[next_node] + heuristic[next_node]
                heapq.heappush(open_set, (f_distance[next_node], next_node))
                came_from[next_node] = current_node
    return f_distance, came_from


if __name__ == "__main__":
    my_graph = {
        'A': {'B':2, 'C':3},
        'B': {'A':2, 'D':3, 'E':1},
        'C': {'A':3, 'F':2},
        'D': {'B':3, 'E':1},
        'E': {'B':1, 'D':1, 'F':1},
        'F': {'C':2, 'E':1}
    }

    heuristic = {
        'A': 7,
        'B': 6,
        'C': 5,
        'D': 4,
        'E': 1,
        'F': 0
    }

    path, dist = astar(my_graph, heuristic, 'A', 'F')
    print('Shortest Path using A*: ', path)

    # visualize the graph
    G = nx.Graph()

nodes = list(my_graph.keys())
weighted_edges = []
for node, neighbors in my_graph.items():
    for neighbor, weight in neighbors.items():
        weighted_edges.append((node, neighbor, weight))

G.add_nodes_from(nodes)
G.add_weighted_edges_from(weighted_edges)

# Show original graph
show_wgraph(G, node_labels=heuristic)

# Show A* path
show_wpath(G, path, node_labels=heuristic)
plt.show()

