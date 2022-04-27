import matplotlib.pyplot as plt
import numpy as np
import os
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples

def smooth(data, weight = 0.9):
    last = data[0]
    res = []
    for point in data:
        smoothed_val = last * weight + (1-weight) * point
        res.append(smoothed_val)
        last = smoothed_val
    return res

if __name__ == '__main__':
    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 16,
             }

    file = os.path.join(os.path.dirname(simple.__file__), 'EPANET Net 3.inp')  # open an example
    es = EPANetSimulation(file)
    network_nodes = [es.network.nodes[x].id for x in list(es.network.nodes)]

    pressure = np.load('mid_pressure.npy')#/300

    p1_id, p2_id = '10', '159'
    p3_id = '215'
    p4_id = '229'
    p1_idx = network_nodes.index(p1_id)
    p2_idx = network_nodes.index(p2_id)
    p3_idx = network_nodes.index(p3_id)
    p4_idx = network_nodes.index(p4_id)

    pressure = pressure[:, 0:120]
    p1 = pressure[p1_idx, :]
    p2 = pressure[p2_idx, :]
    p3 = pressure[p3_idx, :]
    p4 = pressure[p4_idx, :]
    x = np.arange(len(p1))
    plt.plot(x, p1, label='Node '+p1_id)
    #plt.plot(x, p2, label='Node '+p2_id)
    #plt.plot(x, p3, label='Node ' + p3_id)
    #plt.plot(x, p4, label='Node ' + p4_id)
    plt.xlabel('Time (minute)', fontdict=font1)
    plt.ylabel('Pressure (psi)', fontdict=font1)
    plt.grid()

    #plt.grid(True)
    #plt.grid(color='grey', linestyle='--')  # 修改网r't格颜色，类型为虚线

    plt.legend()
    plt.show()






