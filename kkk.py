import os, pprint
import networkx as nx
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import matplotlib.pyplot as plt
import numpy as np


pp = pprint.PrettyPrinter() # we'll use this later.
#file = os.path.join(os.path.dirname(simple.__file__), 'Net2.inp') # open an example
file = os.path.join(os.path.dirname(simple.__file__), 'EPANET Net 3.inp') # open an example
es = EPANetSimulation(file)
G = nx.Graph()
#for i in range(1, len(es.network.nodes)+1):
#    G.add_node(i-1, name=es.network.nodes[i].id)


network_nodes = [es.network.nodes[x].id for x in list(es.network.nodes)]
num_nodes = len(network_nodes)
for item in network_nodes:
    G.add_node(str(item))

network_links = [[es.network.links[i].start.id, es.network.links[i].end.id] for i in list(es.network.links)]
for item in network_links:
    G.add_edge(str(item[0]), str(item[1]))

G_numpy = nx.convert_matrix.to_numpy_matrix(G)


nx.draw(G, with_labels=True, font_weight='bold')
plt.show()






