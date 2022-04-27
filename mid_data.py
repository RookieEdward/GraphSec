import os, pprint
import networkx as nx
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import matplotlib.pyplot as plt
import numpy as np
import csv

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


#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()


es.run()

p = Node.value_type['EN_PRESSURE']
k_p = es.network.nodes[1].results[p]
sample_length = len(k_p)

pressure = Node.value_type['EN_PRESSURE']
head = Node.value_type['EN_HEAD']
elevation = Node.value_type['EN_ELEVATION']
demand = Node.value_type['EN_DEMAND']

data_pressure = np.zeros((num_nodes, sample_length))
data_head = np.zeros((num_nodes, sample_length))
data_elevation = np.zeros((num_nodes, sample_length))
data_demand = np.zeros((num_nodes, sample_length))

for n in range(num_nodes):
    for step in range(sample_length):
        data_pressure[n][step] = es.network.nodes[n+1].results[pressure][step]
        data_head[n][step] = es.network.nodes[n + 1].results[head][step]
        data_demand[n][step] = es.network.nodes[n + 1].results[demand][step]
np.save('mid_pressure.npy', data_pressure)
np.save('mid_head.npy', data_head)
np.save('mid_demand.npy', data_demand)

with open('Mid_data.csv', 'w', newline='') as f:
    spamwriter = csv.writer(f)
    for n in range(1, num_nodes+1):
        head = ['Node index:' + network_nodes[n-1]]
        spamwriter.writerow(head)
        head2 = ['Pressure (psi):']
        spamwriter.writerow(head2)
        spamwriter.writerow(data_pressure[n-1])
        head3 = ['Head (ft):']
        spamwriter.writerow(head3)
        spamwriter.writerow(data_head[n - 1])
        head4 = ['Demand (GPM):']
        spamwriter.writerow(head4)
        spamwriter.writerow(data_demand[n - 1])
        spamwriter.writerow('')

    '''head = ['Node index']
    head2 = ['Data index', 'Pressure (psi)', 'Head (ft)', 'Demand (GPM)']
    spamwriter = csv.writer(f)
    spamwriter.writerow(head)
    spamwriter.writerow(head2)
    spamwriter.writerow(' ')'''


'''with open('Mid_data.csv', 'w', newline='') as f:
    head = ['Node index']
    head2 = ['Data index', 'Pressure (psi)', 'Head (ft)', 'Demand (GPM)']
    spamwriter = csv.writer(f)
    spamwriter.writerow(head)
    spamwriter.writerow(head2)
    spamwriter.writerow(' ')
    spamwriter.writerow(head)
    spamwriter.writerow(head2)'''



'''np.save('mid_pressure.csv', data_pressure)
np.save('mid_head.csv', data_head)
np.save('mid_demand.csv', data_demand)'''
