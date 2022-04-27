import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from MADDPGModelV5 import OUActionNoise, Agent, Buffer, AgentRestore
from nistrng import *


def get_score(alice_act, bob_act):
    score, p_value = 0, 0
    a_act = np.zeros(len(alice_act))
    b_act = np.zeros(len(bob_act))
    for i in range(len(alice_act)):
        if alice_act[i]>=0:
            a_act[i]=1
        if bob_act[i]>=0:
            b_act[i]=1
        if a_act[i]==b_act[i]:
            score+=1
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
    training_epoch = 5
    batch_size = 128
    adj_matrix = np.load('adj_matrix.npy')
    node_attribute_all = np.load('training_data.npy')/600
    node_attribute_all = node_attribute_all[:, 1:]
    node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 45, 5))
    #node_attribute_all = np.reshape(node_attribute_all, (11, 18, 5))
    attribute_length = 5
    key_length = 7
    num_agent = 2
    reward_all = []
    num_node = len(adj_matrix)
    step_length = 40
    alice = AgentRestore(name='Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    bob = AgentRestore(name='Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    #alice = Agent(name='Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    #bob = Agent(name='Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    alice_idx = 1
    bob_idx = 4
    for epoch in range(training_epoch):
        reward_epoch = 0
        alice_current_key = np.zeros((step_length, key_length))
        bob_current_key = np.zeros((step_length, key_length))
        index = np.zeros(step_length)
        for i in range(step_length):
            index[i] = i

        np.random.shuffle(index)

        for step in range(step_length):

            kkk = int(index[step])

            node_attribute = node_attribute_all[:, step, :]
            alice_attribute = node_attribute[alice_idx]
            bob_attribute = node_attribute[bob_idx]
            obs_alice = alice_attribute
            obs_bob = bob_attribute
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            p_value, score, a_binary_sequence, b_binary_sequence = get_score(act_alice, act_bob)
            alice_pen = get_check(step, alice_current_key, a_binary_sequence)
            bob_pen = get_check(step, bob_current_key, b_binary_sequence)
            alice_current_key[step, :] = a_binary_sequence
            bob_current_key[step, :] = b_binary_sequence
            reward = -(alice_pen+bob_pen)/2 + score
            reward_epoch += reward
            print('alice:', a_binary_sequence)
            print('bob:', b_binary_sequence)
        print('reward:', reward_epoch / (step_length))















if __name__ == '__main__':
    run()