import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from MADDPGModelNewV1 import Agent, Buffer, AgentRestore
from nistrng import *


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
    tf.random.set_seed(1)
    np.random.seed(1)
    memory_size = 100000
    training_epoch = 1
    batch_size = 128
    adj_matrix = np.load('adj_matrix.npy')
    node_attribute_all = np.load('test.npy')#/300
    node_attribute_all = node_attribute_all[:, 1:]
    #node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 45, 5))
    #node_attribute_all = np.reshape(node_attribute_all, (36, 12, 60))
    #node_attribute_all = np.random.sample(size=(11, 45, 5))*150+50
    attribute_length = 60
    key_length = 20
    num_agent = 2
    reward_all = []
    num_node = len(adj_matrix)
    step_length = 10
    alice = AgentRestore(name='Alice', num_obs=attribute_length+2, num_act=key_length, num_agent=num_agent, num_node=num_node)
    bob = AgentRestore(name='Bob', num_obs=attribute_length+2, num_act=key_length, num_agent=num_agent, num_node=num_node)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 0.6
    memory = Buffer(buffer_capacity=memory_size, batch_size=batch_size, num_agent=num_agent, num_obs=attribute_length+2, num_node=num_node, num_act=key_length)
    alice_idx = 35
    bob_idx = 32
    eve_idx = 35
    for epoch in range(training_epoch):
        reward_epoch = 0
        alice_current_key = np.zeros((step_length, key_length))
        bob_current_key = np.zeros((step_length, key_length))
        for step in range(step_length):
            node_attribute = node_attribute_all[:, step:step+attribute_length]

            alice_attribute = (node_attribute[eve_idx]-min(node_attribute[eve_idx]))
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
            print('alice:', a_binary_sequence)
            print('bob:', b_binary_sequence)
        print('reward:', reward_epoch/(step_length))












if __name__ == '__main__':
    run()