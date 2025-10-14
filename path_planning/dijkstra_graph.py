import networkx as nx
import matplotlib.pyplot as plt
from heapq import heapify, heappush, heappop
from graph_visualization import show_wgraph, show_wpath_d
class Graph:
    def __init__(self, graph: dict = {}):
        self.graph = graph  # A dictionary for the adjacency list

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph:  # Check if the node is already added
            self.graph[node1] = {}  # If not, create the node as a dict
        self.graph[node1][node2] = weight  # Else, add a connection to its neighbor

    def __str__(self) -> str:
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")
        return ""

    def shortest_distance(self, source: str):
        # Initialize the values of all nodes with infinity
        distances = {node: float("inf") for node in self.graph}
        distances[source] = 0 # set the source value to 0

        #Initialize priority queues
        pq = [(0, source)]
        heapify(pq)

        # create a set to hold visited nodes
        visited = set()

        while pq:
            current_distance, current_node = heappop(pq) # node with shortest distance
            if current_node in visited:
                continue # skip already visited nodes
            visited.add(current_node) # Else, add the node to visited node

            for neighbor, weight in self.graph[current_node].items():
                # calculate the distance from the current_node to the neighbor
                tentative_distance = current_distance + weight
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    heappush(pq, (tentative_distance, neighbor))

        predecessors = {node: None for node in self.graph}
        for node, distance in distances.items():
            for neighbor, weight in self.graph[node].items():
                if distances[neighbor] == distance + weight:
                    predecessors[neighbor] = node
        return distances, predecessors

    def shortest_path(self, source: str, target: str):
        # generate predecessors dict
        distances, predecessors = self.shortest_distance(source)
        # Initialize the path with the target node
        path = []
        current_node = target
        # Backtrack from the target node using predecessors
        while current_node:
            path.append(current_node)
            current_node = predecessors[current_node]
        # Reverse the path and return it
        path.reverse()
        shortest_distance_to_target = distances[target]
        return path, shortest_distance_to_target

    def get_nodes(self):
        return list(self.graph.keys())

    def get_edges(self):
        edges = []
        for node in self.graph:
            for neighbor, weight in self.graph[node].items():
                edges.append((node, neighbor, weight))
        return edges
    
if __name__ == "__main__":
    G = Graph()

    # Add A and its neighbors
    G.add_edge("A", "B", 3)
    G.add_edge("A", "C", 3)

    # Add B and its neighbors
    G.add_edge("B", "A", 3)
    G.add_edge("B", "D", 3.5)
    G.add_edge("B", "E", 2.8)

    G.add_edge("C", "A", 3)
    G.add_edge("C", "E", 2.8)
    G.add_edge("C", "F", 3.5)

    G.add_edge("D", "B", 3.5)
    G.add_edge("D", "E", 3.1)
    G.add_edge("D", "G", 10)

    G.add_edge("E", "B", 2.8)
    G.add_edge("E", "C", 2.8)
    G.add_edge("E", "D", 3.1)
    G.add_edge("E", "G", 7)

    G.add_edge("F", "C", 3.5)
    G.add_edge("F", "G", 2.5)

    G.add_edge("G", "D", 10)
    G.add_edge("G", "E", 7)
    G.add_edge("G", "F", 2.5)

    print(G)

    path, dist_to_target = G.shortest_path("B", "F")
    print('Shortest path from node B to F:', path)
    print('Distance to target: ', dist_to_target)


    path, dist_to_target = G.shortest_path("A", "G")
    print('shortest path from A to G: ', path)
    print('Distance to target: ', dist_to_target)


    # visualize orginal graph 
    nxG = nx.Graph()

    nodes = G.get_nodes()
    weighted_edges = G.get_edges()

    nxG.add_nodes_from(nodes)
    nxG.add_weighted_edges_from(weighted_edges)
    show_wgraph(nxG)
    

    # visualize shortest path from A to G
    show_wpath_d(nxG, 'A', 'G')

    plt.show()