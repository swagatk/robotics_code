"""
Dijkstra algorithm takes nodes and edges separately.
Edges are in the form of dictionary with tuple keys.
"""
from graph_visualization import show_wpath_ne
import matplotlib.pyplot as plt

Example_1 = False
Example_2 = True

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
    path_trace = {}

    while len(unexplored) != 0:
        explore = minimum(unexplored)

        if explore == end:
            break
        else:
            for path in edges.items():
                if path[0][0] == explore:
                    neighbor = path[0][1]
                    if neighbor in unexplored.keys():
                        check_time = unexplored[explore] + path[1]
                        if check_time < unexplored[neighbor]:
                            unexplored[neighbor] = check_time
                            path_trace[neighbor] = explore
                elif path[0][1] == explore:
                    neighbor = path[0][0]
                    if neighbor in unexplored.keys():
                        check_time = unexplored[explore] + path[1]
                        if check_time < unexplored[neighbor]:
                            unexplored[neighbor] = check_time
                            path_trace[neighbor] = explore
            del unexplored[explore]

    # Reconstruct the shortest path
    current = end
    path = [current]
    while current != start:
        current = path_trace[current]
        path.append(current)
    path.reverse()

    return(unexplored[explore], path)

if __name__ == "__main__":

    if Example_1:
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

        # visualize the graph with path
        show_wpath_ne(airports, lines, wp)
        plt.show()

    if Example_2:

        nodes = ['A', 'B', 'C', 'D', 'E', 'F']
        edges = {

                ('A', 'B') : 2,
                ('A', 'C') : 4,
                ('B', 'C') : 1,
                ('B', 'D') : 4,
                ('B', 'E') : 2,
                ('C', 'E') : 3,
                ('D', 'F') : 2,
                ('E', 'D') : 3,
                ('E', 'F') : 2,

        }

        start = 'A'
        end = 'F'

        # Apply dijkstra algorithm to find optimal path
        cum_cost, wp = dijkstra(nodes, edges, start, end)
        print("Total Cost: ", cum_cost)
        print("Shortest Path: ", wp)

        # visualize the graph with path
        show_wpath_ne(nodes, edges, wp)
        plt.show()

