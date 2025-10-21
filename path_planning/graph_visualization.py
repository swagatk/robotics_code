import networkx as nx
import matplotlib.pyplot as plt



# provide options to display additional node values
def show_wgraph(G, custom_node_positions=None, node_labels:dict=None):
    plt.figure()

    if custom_node_positions==None:
        pos = nx.spring_layout(G)
    else:
        pos=custom_node_positions

    weight_labels = nx.get_edge_attributes(G,'weight')
    nx.draw(G,pos,font_color = 'white', node_shape = 's', with_labels = True,)
    output = nx.draw_networkx_edge_labels(G,pos,edge_labels=weight_labels)
    if node_labels!=None:
        pos2 = {k:(v[0]+0.1,v[1]+0.1) for (k,v) in pos.items()}
        nx.draw_networkx_labels(G, pos2, labels=node_labels, font_color='r')
    return plt

# provide options to display additional node values
# shows dijkstra shortest path
def show_wpath_d(G, from_node, to_node,custom_node_positions=None, node_labels=None):
    fig, ax = plt.subplots()

    if custom_node_positions==None:
        pos = nx.spring_layout(G)
    else:
        pos=custom_node_positions

    weight_labels = nx.get_edge_attributes(G,'weight')

    path = nx.dijkstra_path(G, source = from_node, target = to_node)

    edges_path = list(zip(path,path[1:]))
    edges_path_reversed = [(y,x) for (x,y) in edges_path]
    edges_path = edges_path + edges_path_reversed
    edge_colors = ['black' if not edge in edges_path else 'red' for edge in G.edges()]

    nodecol = ['steelblue' if not node in path else 'red' for node in G.nodes()]
    nx.draw(G, pos, with_labels = True, font_color = 'white', edge_color= edge_colors, node_shape = 's', node_color = nodecol, ax=ax)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=weight_labels, ax=ax)

    if node_labels != None:
        pos2 = {k:(v[0]+0.1,v[1]+0.1) for (k,v) in pos.items()}
        nx.draw_networkx_labels(G, pos2, labels=node_labels, font_color='r')

    return fig, ax


# shows the specified path in red color
def show_wpath(G, path, custom_node_positions=None, node_labels=None, title=None):
    """
    Highlights a specified path on a networkx graph.

    Args:
        G (nx.Graph): The graph to display.
        path (list): A list of nodes representing the path.
        custom_node_positions (dict, optional): Pre-defined positions for nodes. Defaults to None.
        node_labels (dict, optional): Additional labels for nodes. Defaults to None.
    """
    fig, ax = plt.subplots()
    pos = custom_node_positions if custom_node_positions is not None else nx.spring_layout(G)
    weight_labels = nx.get_edge_attributes(G, 'weight')
    edges_in_path = list(zip(path, path[1:]))
    edge_colors = ['red' if (u, v) in edges_in_path or (v, u) in edges_in_path else 'black' for u, v in G.edges()]
    node_colors = ['red' if node in path else 'steelblue' for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, font_color='white', edge_color=edge_colors, node_shape='s', node_color=node_colors, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weight_labels, ax=ax)

    if node_labels is not None:
        pos2 = {k: (v[0] + 0.1, v[1] + 0.1) for (k, v) in pos.items()}
        nx.draw_networkx_labels(G, pos2, labels=node_labels, font_color='r', ax=ax)

    if title is not None:
        ax.set_title(title)
        
    return fig, ax

# plot a maze and the path found
def plot_maze(maze, path, algo='path', title=None):
    plt.imshow(maze, cmap='binary')
    plt.plot([x[1] for x in path], [x[0] for x in path], 'r', label=algo)
    start = path[0]
    end = path[-1]
    plt.plot(start[1], start[0], 'go', markersize=10, label='start')
    plt.plot(end[1], end[0], 'ro', markersize=10, label='goal')
    plt.xticks(range(0,10,1))
    plt.yticks(range(0,10,1))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(visible='both', which='both')
    plt.legend(loc='best')
    if title != None:
        plt.title(title)
    return plt


def show_wpath_ne(nodes: list, edges: dict, path: list, 
                  custom_node_positions=None, node_labels=None,
                  title=None):
    """
    It takes an input of nodes, edges (in dictionary form) and a path (list of nodes)
    and visualizes the graph with the specified path highlighted.
    """
    fig, ax = plt.subplots()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    weighted_edges = []
    for edge, weight in edges.items():
        weighted_edges.append((edge[0], edge[1], weight))
    G.add_weighted_edges_from(weighted_edges)

    if custom_node_positions==None:
        pos = nx.spring_layout(G)
    else:
        pos=custom_node_positions

    weight_labels = nx.get_edge_attributes(G,'weight')

    edges_path = list(zip(path,path[1:]))
    edges_path_reversed = [(y,x) for (x,y) in edges_path]
    edges_path = edges_path + edges_path_reversed
    edge_colors = ['black' if not edge in edges_path else 'red' for edge in G.edges()]

    nodecol = ['steelblue' if not node in path else 'red' for node in G.nodes()]
    nx.draw(G, pos, with_labels = True, font_color = 'white', edge_color= edge_colors, node_shape = 's', node_color = nodecol, ax=ax)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=weight_labels, ax=ax)

    if node_labels != None:
        pos2 = {k:(v[0]+0.1,v[1]+0.1) for (k,v) in pos.items()}
        nx.draw_networkx_labels(G, pos2, labels=node_labels, font_color='r')

    if title != None:
        ax.set_title(title)

    return fig, ax


def show_wgraph_ne(nodes: list, edges: dict, 
                   custom_node_positions=None, node_labels:dict=None,
                   title=None):
    """
    Shows a graph given nodes and edges in dictionary form.
    """
    fig, ax = plt.subplots()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    weighted_edges = []
    for edge, weight in edges.items():
        weighted_edges.append((edge[0], edge[1], weight))
    G.add_weighted_edges_from(weighted_edges)

    if custom_node_positions==None:
        pos = nx.spring_layout(G)
    else:
        pos=custom_node_positions

    weight_labels = nx.get_edge_attributes(G,'weight')
    nx.draw(G,pos,font_color = 'white', node_shape = 's', with_labels = True, ax=ax)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=weight_labels, ax=ax)
    if node_labels!=None:
        pos2 = {k:(v[0]+0.1,v[1]+0.1) for (k,v) in pos.items()}
        nx.draw_networkx_labels(G, pos2, labels=node_labels, font_color='r', ax=ax)

    if title != None:
        ax.set_title(title)
    return fig, ax