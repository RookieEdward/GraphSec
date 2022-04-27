import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from MADDPGModelNewV1 import Agent, Buffer, AgentRestore
from nistrng import *
import os
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import csv


def get_score(alice_act, bob_act):
    score, p_value = 0, 0
    a_act = np.zeros(len(alice_act), dtype=np.int)
    b_act = np.zeros(len(bob_act), dtype=np.int)
    for i in range(len(alice_act)):
        if alice_act[i] >= 0:
            a_act[i] = 1
        if bob_act[i] >= 0:
            b_act[i] = 1
        if a_act[i] == b_act[i]:
            score += 1
    if np.sum(a_act) ==0 or np.sum(a_act)==len(alice_act):
        a_value = 0
    else:
        eligible_battery: dict = check_eligibility_all_battery(a_act, SP800_22R1A_BATTERY)
        results = run_all_battery(a_act, eligible_battery, False)
        a_value = results[0][0].score
    if np.sum(b_act) ==0 or np.sum(b_act)==len(bob_act):
        b_value = 0
    else:
        eligible_battery: dict = check_eligibility_all_battery(b_act, SP800_22R1A_BATTERY)
        results = run_all_battery(b_act, eligible_battery, False)
        b_value = results[0][0].score
    p_value = (a_value + b_value) / 2
    if p_value >= 0.01:
        p_value = 0.01
    return p_value, score/len(b_act), a_act, b_act



def get_reward(a_act, b_act):
    score = np.sqrt(np.square(a_act - b_act)).mean()
    return -score

def get_correction(alice_act, bob_act):
    a_act = np.zeros(len(alice_act))
    b_act = np.zeros(len(bob_act))
    for i in range(len(alice_act)):
        if alice_act[i] >= 0:
            a_act[i] = 1
        if bob_act[i] >= 0:
            b_act[i] = 1
    return a_act, b_act

def get_check(step, current_key, binary_sequence):
    count = 0
    for i in range(step):
        if (current_key[i,:]==binary_sequence).all() == True:
            count += 1
    return count

def run():
    file = os.path.join(os.path.dirname(simple.__file__), 'EPANET Net 3.inp')  # open an example
    es = EPANetSimulation(file)
    network_nodes = [es.network.nodes[x].id for x in list(es.network.nodes)]
    tf.random.set_seed(1)
    np.random.seed(1)
    memory_size = 100000
    training_epoch = 1
    node_attribute_all = np.load('mid_pressure.npy')#/300
    node_attribute_all = node_attribute_all[:, 1:]
    #node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 45, 5))
    #node_attribute_all = np.reshape(node_attribute_all, (36, 12, 60))
    #node_attribute_all = np.random.sample(size=(11, 45, 5))*150+50
    attribute_length = 60
    key_length = 20
    num_agent = 2
    reward_all = []
    num_node = len(network_nodes)
    step_length = 20
    alice = AgentRestore(name='Alice', num_obs=attribute_length+2, num_act=key_length, num_agent=num_agent, num_node=num_node)
    bob = AgentRestore(name='Bob', num_obs=attribute_length+2, num_act=key_length, num_agent=num_agent, num_node=num_node)
    alice_id = '10'
    bob_id = '159'
    alice_idx = network_nodes.index(alice_id)
    bob_idx = network_nodes.index(bob_id)
    # alice_idx = 9
    # bob_idx = 11
    for epoch in range(training_epoch):
        reward_epoch = 0
        alice_current_key = np.zeros((step_length, key_length), dtype=np.int)
        bob_current_key = np.zeros((step_length, key_length), dtype=np.int)
        index = np.zeros(step_length)
        for i in range(step_length):
            index[i] = i
        np.random.shuffle(index)
        for step in range(step_length):
            kkk = int(index[step])
            node_attribute = node_attribute_all[:, step:step+attribute_length]
            alice_attribute = (node_attribute[alice_idx]-min(node_attribute[alice_idx]))
            bob_attribute = (node_attribute[bob_idx]-min(node_attribute[bob_idx]))
            obs_alice = np.concatenate((alice_attribute, [alice_idx, bob_idx]))
            obs_bob = np.concatenate((bob_attribute, [alice_idx, bob_idx]))
            state = np.vstack((obs_alice, obs_bob))
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            p_value, score, a_binary_sequence, b_binary_sequence = get_score(act_alice, act_bob)

            alice_pen = get_check(step, alice_current_key, a_binary_sequence)
            bob_pen = get_check(step, bob_current_key, b_binary_sequence)
            alice_current_key[step, :] = a_binary_sequence
            bob_current_key[step, :] = b_binary_sequence
            reward = -(alice_pen + bob_pen)/2 + score + p_value
            reward_epoch += reward
            print('alice:%s, ind:%s' % (a_binary_sequence, kkk))
            print('bob:%s, ind:%s' % (b_binary_sequence, kkk))
        print('reward:', reward_epoch/(step_length))

    with open('Keys.csv', 'w', newline='') as f:
        spamwriter = csv.writer(f)
        head = ['Node index:' + alice_id]
        spamwriter.writerow(head)
        head2 = ['Keys:']
        spamwriter.writerow(head2)
        spamwriter.writerows(alice_current_key)
        spamwriter.writerow('')
        spamwriter = csv.writer(f)
        head = ['Node index:' + bob_id]
        spamwriter.writerow(head)
        head2 = ['Keys:']
        spamwriter.writerow(head2)
        spamwriter.writerows(bob_current_key)













if __name__ == '__main__':
    run()