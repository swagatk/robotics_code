"""
Dijkstra's algorithm implementation using dictionary input

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_visualization import show_wgraph, show_wpath_d

# 

# minimum function for dictionary,
# it will return the key who have the smallest value
def minimum(dict):
    min_key = list(dict.keys())[0]
    for i in list(dict.keys())[1:]:
        if dict[i] < dict[min_key]:
            min_key = i
    return(min_key)

def dijkstra(nodes, edges, start, end):
    unexplored = {node : float('inf') for node in nodes}
    unexplored[start] = 0
    shortest_path = []
    while len(unexplored) != 0:
        explore = minimum(unexplored)
        shortest_path.append(explore)
        if explore == end:
            break
        else:
            for path in edges.items():
                if path[0][0] == explore:
                    if path[0][1] in unexplored.keys():
                        check_time = unexplored[path[0][0]] + path[1]
                        if check_time < unexplored[path[0][1]]:
                            unexplored[path[0][1]] = check_time
                elif path[0][1] == explore:
                    if path[0][0] in unexplored.keys():
                        check_time = unexplored[path[0][1]] + path[1]
                        if check_time < unexplored[path[0][0]]:
                            unexplored[path[0][0]] = check_time
            del unexplored[explore]
    return(unexplored[explore], shortest_path)


if __name__ == "__main__":

    # list to represent the Airports
    airports = ['A', 'B', 'C', 'D', 'E']
    # dictionary to represent the lines between Airports
    # and the time it will take
    lines = {
        ('A', 'B') : 4,
        ('A', 'C') : 2,
        ('B', 'C') : 1,
        ('B', 'D') : 2,
        ('C', 'D') : 4,
        ('C', 'E') : 5,
        ('E', 'D') : 1,
    }

    # we choose Airport A as the starting airport
    # and Airport D as the destination
    start = 'A'
    end = 'D'

    # Apply dijkstra algorithm to find optimal path
    cum_cost, wp = dijkstra(airports, lines, start, end)
    print("Total Cost: ", cum_cost)
    print("Shortest Path: ", wp)

    # visualize original graph
    # create a networkx graph from the dictionary    
    G = nx.Graph()
    weighted_edges = [(*key, value) for (key, value) in zip(lines.keys(), lines.values())]
    node_pos = {'A':(1, 2), 'B':(2, 3), 'C':(2, 1), 'D':(3, 3), 'E':(3, 1)}


    G.add_nodes_from(airports)
    #G.add_edges_from(lines)
    G.add_weighted_edges_from(weighted_edges)

    show_wgraph(G, custom_node_positions=node_pos)

    # show shortest path
    show_wpath_d(G, start, end, node_pos)

    plt.show()