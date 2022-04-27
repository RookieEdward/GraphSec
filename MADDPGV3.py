import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from MADDPGModelV3 import OUActionNoise, Agent, Buffer
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

def run():
    tf.random.set_seed(1)
    np.random.seed(1)
    memory_size = 100000
    training_epoch = 80000
    batch_size = 128
    adj_matrix = np.load('adj_matrix.npy')
    node_attribute_all = np.load('training_data.npy')/600
    node_attribute_all[:, 1:]
    node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 90, 10))
    attribute_length = 10
    key_length = 5
    num_agent = 2
    reward_all = []
    num_node = len(adj_matrix)
    step_length = node_attribute_all.shape[1]-88
    alice = Agent(name='Alice', num_obs=key_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    bob = Agent(name='Bob', num_obs=key_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 1
    memory = Buffer(buffer_capacity=memory_size, batch_size=batch_size, num_agent=num_agent, num_obs=key_length, num_node=num_node, num_act=key_length)
    for epoch in range(training_epoch):
        reward_epoch = 0
        p_value_epoch = 0
        score_epoch = 0
        alice_current_key = np.zeros(key_length)
        bob_current_key = np.zeros(key_length)
        for step in range(step_length):
            obs_alice = alice_current_key
            obs_bob = bob_current_key
            state = np.vstack((obs_alice, obs_bob))
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            act_alice = np.clip(np.random.normal(act_alice, std_dev), -1, 1)
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            act_bob = np.clip(np.random.normal(act_bob, std_dev), -1, 1)
            p_value, score, a_binary_sequence, b_binary_sequence = get_score(act_alice, act_bob)

            alice_pen = np.sum(np.abs(alice_current_key-a_binary_sequence))/(key_length)
            bob_pen = np.sum(np.abs(bob_current_key-b_binary_sequence))/(key_length)

            reward = (alice_pen+bob_pen)/2
            reward_epoch += reward

            alice_current_key = a_binary_sequence
            bob_current_key = b_binary_sequence
            obs_alice_ = a_binary_sequence
            obs_bob_ = b_binary_sequence
            state_ = np.vstack((obs_alice_, obs_bob_))

            action = np.concatenate((act_alice, act_bob), axis=0)
            #action = np.vstack((act_alice, act_bob))
            memory.store(state, action, reward, state_)
            if memory.buffer_counter >= batch_size*10:
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
                    Q = alice.critic([bt_state, bt_alice_act, bob.actor(bt_bob_obs)], training=True)
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
                    y = bt_reward + gamma * bob.critic_target([bt_state_, bt_alice_act_, bt_bob_act_], training=True)
                    critic_value = bob.critic([bt_state, bt_alice_act, bt_bob_act], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, bob.critic.trainable_variables)
                bob.critic_optimizer.apply_gradients(zip(critic_grad, bob.critic.trainable_variables))

                alice.target_update()
                bob.target_update()
        reward_all.append(reward_epoch/step_length)
        #print('epoch:%s, start:%s, reward:[p_value:%s, score:%s, k_a:%s, k_b:%s]' % (epoch, start_flag, p_value_epoch/step_length, score_epoch/step_length, k_a, k_b))
        print('epoch:%s, start:%s, reward:%s' % (epoch, start_flag, reward_epoch/step_length))
        if (epoch + 1) % 500 == 0 and start_flag is True:
            np.save("reward_new.npy", reward_all)
            #alice.save_model()
            #bob.save_model()

            x = np.arange(len(reward_all))
            plt.plot(x, reward_all, label='reward')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()
            alice_current_key = np.zeros(key_length)
            bob_current_key = np.zeros(key_length)
            reward_epoch = 0
            for step in range(step_length):
                obs_alice = alice_current_key
                obs_bob = bob_current_key
                act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
                act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
                p_value, score, a_binary_sequence, b_binary_sequence = get_score(act_alice, act_bob)
                alice_pen = np.sum(np.abs(alice_current_key - a_binary_sequence)) / (key_length)
                bob_pen = np.sum(np.abs(bob_current_key - b_binary_sequence)) / (key_length)
                reward = (alice_pen + bob_pen) / 2
                reward_epoch += reward
                alice_current_key = a_binary_sequence
                bob_current_key = b_binary_sequence
                print('alice:', a_binary_sequence)
                print('bob:', b_binary_sequence)
            print('reward:', reward_epoch/step_length)













if __name__ == '__main__':
    run()