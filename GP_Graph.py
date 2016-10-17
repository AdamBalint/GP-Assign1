import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

l = None
g = nx.Graph();
pos = graphviz_layout(g, prog="neato")

# Graphs the program tree
def graph(nodes, edges, labels):
    g = nx.Graph();
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    l = labels
    pos = graphviz_layout(g, prog='dot')
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, l)
    plt.show();
