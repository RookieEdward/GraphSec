import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from MADDPGModelNewV1 import Agent, Buffer
from nistrng import *
import os
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples

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
    if np.sum(a_act) == 0 or np.sum(a_act) == len(alice_act):
        a_value = 0
    else:
        eligible_battery: dict = check_eligibility_all_battery(a_act, SP800_22R1A_BATTERY)
        results = run_all_battery(a_act, eligible_battery, False)
        a_value = np.sum([results[i][0].score for i in range(len(results))])/len(results)

    if np.sum(b_act) ==0 or np.sum(b_act)==len(bob_act):
        b_value = 0
    else:
        eligible_battery: dict = check_eligibility_all_battery(b_act, SP800_22R1A_BATTERY)
        results = run_all_battery(b_act, eligible_battery , False)
        b_value = np.sum([results[i][0].score for i in range(len(results))])/len(results)
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
    file = os.path.join(os.path.dirname(simple.__file__), 'EPANET Net 3.inp')  # open an example
    es = EPANetSimulation(file)
    network_nodes = [es.network.nodes[x].id for x in list(es.network.nodes)]
    memory_size = 100000
    training_epoch = 15000
    batch_size = 128
    node_attribute_all = np.load('mid_pressure.npy')#/300
    node_attribute_all = node_attribute_all[:, 1:]
    #node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 45, 5))
    #node_attribute_all = np.reshape(node_attribute_all, (36, 24, 30))
    #node_attribute_all = np.random.sample(size=(11, 45, 5))*150+50
    attribute_length = 60
    key_length = 20
    num_agent = 2
    reward_all = []
    num_node = len(network_nodes)
    step_length = 20
    alice = Agent(name='Alice', num_obs=attribute_length+2, num_act=key_length, num_agent=num_agent, num_node=num_node)
    bob = Agent(name='Bob', num_obs=attribute_length+2, num_act=key_length, num_agent=num_agent, num_node=num_node)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 0.6
    memory = Buffer(buffer_capacity=memory_size, batch_size=batch_size, num_agent=num_agent, num_obs=attribute_length+2, num_node=num_node, num_act=key_length)


    alice_id = '10'
    bob_id = '159'
    alice_idx = network_nodes.index(alice_id)
    bob_idx = network_nodes.index(bob_id)
    #alice_idx = 9
    #bob_idx = 11
    for epoch in range(training_epoch):
        reward_epoch = 0
        alice_current_key = np.zeros((step_length, key_length))
        bob_current_key = np.zeros((step_length, key_length))
        for step in range(step_length):
            node_attribute = node_attribute_all[:, step:step+attribute_length]

            alice_attribute = (node_attribute[alice_idx]-min(node_attribute[alice_idx]))
            bob_attribute = (node_attribute[bob_idx]-min(node_attribute[bob_idx]))
            #obs_alice = alice_attribute
            #obs_bob = bob_attribute
            obs_alice = np.concatenate((alice_attribute, [alice_idx, bob_idx]))
            obs_bob = np.concatenate((bob_attribute, [alice_idx, bob_idx]))
            state = np.vstack((obs_alice, obs_bob))
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            act_alice = np.clip(np.random.normal(act_alice, std_dev), -1, 1)
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            act_bob = np.clip(np.random.normal(act_bob, std_dev), -1, 1)
            p_value, score, a_binary_sequence, b_binary_sequence = get_score(act_alice, act_bob)

            alice_pen = get_check(step, alice_current_key, a_binary_sequence)
            bob_pen = get_check(step, bob_current_key, b_binary_sequence)
            alice_current_key[step, :] = a_binary_sequence
            bob_current_key[step, :] = b_binary_sequence
            reward = -(alice_pen + bob_pen)/2 + score + p_value
            reward_epoch += reward
            node_attribute_ = node_attribute_all[:, step+1:step+1+attribute_length]
            alice_attribute_ = (node_attribute_[alice_idx] - min(node_attribute_[alice_idx]))
            bob_attribute_ = (node_attribute_[bob_idx] - min(node_attribute_[bob_idx]))
            #obs_alice_ = alice_attribute_
            #obs_bob_ = bob_attribute_
            obs_alice_ = np.concatenate((alice_attribute_, [alice_idx, bob_idx]))
            obs_bob_ = np.concatenate((bob_attribute_, [alice_idx, bob_idx]))
            state_ = np.vstack((obs_alice_, obs_bob_))

            action = np.concatenate((act_alice, act_bob), axis=0)
            memory.store(state, action, reward, state_)
            if memory.buffer_counter >= batch_size*30:
                start_flag = True
                std_dev *= 0.99995
                # Get sampling range
                record_range = min(memory.buffer_counter, memory.buffer_capacity)
                # Randomly sample indices
                idx = np.random.choice(record_range, batch_size)
                bt_state = tf.convert_to_tensor(memory.state_buffer[idx])
                bt_action = tf.convert_to_tensor(memory.action_buffer[idx])
                bt_reward = tf.convert_to_tensor(memory.reward_buffer[idx])
                bt_state_ = tf.convert_to_tensor(memory.next_state_buffer[idx])

                bt_alice_obs, bt_bob_obs = bt_state[:, 0, :], bt_state[:, 1, :]
                bt_alice_obs_, bt_bob_obs_ = bt_state_[:, 0, :], bt_state_[:, 1, :]
                bt_alice_act, bt_bob_act = bt_action[:, 0:alice.num_act], bt_action[:, -alice.num_act:]
                bt_alice_act_, bt_bob_act_ = alice.actor_target(bt_alice_obs_, training=True), bob.actor_target(bt_bob_obs_, training=True)
                bt_reward = tf.cast(bt_reward, tf.float32)

                with tf.GradientTape() as tape:
                    Q = alice.critic([bt_state, alice.actor(bt_alice_obs), bt_bob_act], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, alice.actor.trainable_variables)
                alice.actor_optimizer.apply_gradients(zip(actor_grad, alice.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    Q = bob.critic([bt_state, bob.actor(bt_bob_obs), bt_alice_act], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, bob.actor.trainable_variables)
                bob.actor_optimizer.apply_gradients(zip(actor_grad, bob.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * alice.critic_target([bt_state_, bt_alice_act_, bt_bob_act_], training=True)
                    critic_value = alice.critic([bt_state, bt_alice_act, bt_bob_act], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, alice.critic.trainable_variables)
                alice.critic_optimizer.apply_gradients(zip(critic_grad, alice.critic.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * bob.critic_target([bt_state_, bt_bob_act_, bt_alice_act_], training=True)
                    critic_value = bob.critic([bt_state, bt_bob_act, bt_alice_act], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, bob.critic.trainable_variables)
                bob.critic_optimizer.apply_gradients(zip(critic_grad, bob.critic.trainable_variables))


                alice.target_update()
                bob.target_update()
        reward_all.append(reward_epoch/step_length)
        #print('epoch:%s, start:%s, reward:[p_value:%s, score:%s, k_a:%s, k_b:%s]' % (epoch, start_flag, p_value_epoch/step_length, score_epoch/step_length, k_a, k_b))
        print('epoch:%s, start:%s, reward:%s' % (epoch, start_flag, reward_epoch/step_length))
        if (epoch + 1) % 1000 == 0 and start_flag is True:
            np.save("reward_new.npy", reward_all)
            alice.save_model()
            bob.save_model()

            x = np.arange(len(reward_all))
            plt.plot(x, reward_all, label='reward')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()
            alice_current_key = np.zeros((step_length, key_length))
            bob_current_key = np.zeros((step_length, key_length))
            reward_epoch = 0
            for step in range(1, step_length):
                node_attribute = node_attribute_all[:, step:step+attribute_length]

                alice_attribute = (node_attribute[alice_idx] - min(node_attribute[alice_idx]))
                bob_attribute = (node_attribute[bob_idx] - min(node_attribute[bob_idx]))
                #obs_alice = alice_attribute
                #obs_bob = bob_attribute
                obs_alice = np.concatenate((alice_attribute, [alice_idx, bob_idx]))
                obs_bob = np.concatenate((bob_attribute, [alice_idx, bob_idx]))
                act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
                act_alice = np.clip(np.random.normal(act_alice, std_dev), -1, 1)
                act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
                act_bob = np.clip(np.random.normal(act_bob, std_dev), -1, 1)
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