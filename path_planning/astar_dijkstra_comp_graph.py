"""
Comparing A* and Dijkstra algorithms on the same graph
"""
import numpy as np
import matplotlib.pyplot as plt
from graph_visualization import show_wgraph, show_wpath
from dijkstra_graph import Graph as DGraph
from astar_graph import Graph as AGraph

import networkx as nx

my_graph = {
        'A': {'B':2, 'C':3},
        'B': {'A':2, 'D':3, 'E':1, 'F':5},
        'C': {'A':3, 'D':1, 'E':2, 'F':2},
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

# Create graph object
dg = DGraph(my_graph)
ag = AGraph(my_graph)

# Find shortest path using A*
path_a, dist_a = ag.astar_path("A", "F", heuristic)
print('Shortest Path using A*: ', path_a)
print('Total cost: ', dist_a)

# Find shortest path using Dijkstra
path_d, dist_d = dg.shortest_path("A", "F")
print('Shortest Path using Dijkstra: ', path_d)
print('Total cost: ', dist_d)


# --- Visualization ---
nxG = nx.Graph()
nodes = dg.get_nodes()
weighted_edges = dg.get_edges()
nxG.add_nodes_from(nodes)
nxG.add_weighted_edges_from(weighted_edges)
node_pos = {'A':(1, 2), 'B':(2, 3), 'C':(2, 1), 'D':(3, 3), 'E':(4, 2), 'F':(4, 1)}

show_wgraph(nxG, custom_node_positions=node_pos, node_labels=heuristic)
show_wpath(nxG, path_a, custom_node_positions=node_pos, node_labels=heuristic)
show_wpath(nxG, path_d, custom_node_positions=node_pos)
plt.show()

