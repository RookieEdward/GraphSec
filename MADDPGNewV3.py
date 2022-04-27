import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from MADDPGModelNewV1 import OUActionNoise, Agent, Buffer
from nistrng import *


'''def get_score(alice_act, bob_act):
    a_t = np.zeros(len(alice_act), dtype=np.int)
    b_t = np.zeros(len(bob_act), dtype=np.int)
    for i in range(len(alice_act)):
        if alice_act[i]>=0:
            a_t[i] = 1
        if bob_act[i]>= 0:
            b_t[i] = 1
    eligible_battery: dict = check_eligibility_all_battery(a_t, SP800_22R1A_BATTERY)
    results = run_all_battery(a_t, eligible_battery, False)
    a_value = results[0][0].score

    eligible_battery: dict = check_eligibility_all_battery(b_t, SP800_22R1A_BATTERY)
    results = run_all_battery(b_t, eligible_battery, False)
    b_value = results[0][0].score
    p_value = (a_value+b_value)/2
    score = 0
    for i in range(len(a_t)):
        if a_t[i]==b_t[i]:
            score+=1
    return p_value, score/len(a_t)'''

def get_score(alice_act, bob_act):
    #a_act = np.round(alice_act*64)
    #b_act = np.round(bob_act*64)
    score, p_value = 0, 0
    a_act = alice_act
    b_act = bob_act
    #q_act = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0001])
    #q_act = np.arange(0, 1.001, 1/16)
    q_act = np.linspace(0, 1.000000001, num=16, endpoint=True, retstep=False, dtype=None)
    #q_act = np.linspace(0, 1.00000000001, num=8, endpoint=True, retstep=False, dtype=None)
    a_act = np.digitize(a_act, q_act)-1
    b_act = np.digitize(b_act, q_act)-1
    def bin_array(act):
        m = 4
        """Convert a positive integer num into an m-bit bit vector"""
        kk = []
        for i in range(len(act)):
            k = np.array(list(np.binary_repr(act[i]).zfill(m))).astype(np.int8)
            kk.append(k)
        return np.reshape(np.array(kk), (1, -1))[0]
    a_binary_sequence = bin_array(a_act)
    b_binary_sequence = bin_array(b_act)
    #a_binary_sequence: np.ndarray = np.unpackbits(np.array(a_act, dtype=np.uint8)).astype(np.uint8)
    eligible_battery: dict = check_eligibility_all_battery(a_binary_sequence, SP800_22R1A_BATTERY)
    results = run_all_battery(a_binary_sequence, eligible_battery, False)
    a_value = results[0][0].score
    #b_binary_sequence: np.ndarray = np.unpackbits(np.array(b_act, dtype=np.uint8)).astype(np.uint8)
    eligible_battery: dict = check_eligibility_all_battery(b_binary_sequence, SP800_22R1A_BATTERY)
    results = run_all_battery(b_binary_sequence, eligible_battery, False)
    b_value = results[0][0].score
    p_value = (a_value + b_value) / 2
    for i in range(len(a_binary_sequence)):
        if a_binary_sequence[i] == b_binary_sequence[i]:
            score += 1
    #if p_value>=0.1:
    #    p_value = 0.1
    return p_value, score/len(a_binary_sequence)



def get_reward(a_act, b_act):
    score = np.sqrt(np.square(a_act - b_act)).mean()
    return -score

def get_correction(alice_act, bob_act):
    score, p_value = 0, 0
    a_act = alice_act
    b_act = bob_act
    # q_act = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0001])
    # q_act = np.arange(0, 1.001, 1/16)
    q_act = np.linspace(0, 1.000000001, num=16, endpoint=True, retstep=False, dtype=None)
    # q_act = np.linspace(0, 1.00000000001, num=8, endpoint=True, retstep=False, dtype=None)
    a_act = np.digitize(a_act, q_act) - 1
    b_act = np.digitize(b_act, q_act) - 1

    def bin_array(act):
        m = 4
        """Convert a positive integer num into an m-bit bit vector"""
        kk = []
        for i in range(len(act)):
            k = np.array(list(np.binary_repr(act[i]).zfill(m))).astype(np.int8)
            kk.append(k)
        return np.reshape(np.array(kk), (1, -1))[0]

    a_binary_sequence = bin_array(a_act)
    b_binary_sequence = bin_array(b_act)
    return a_binary_sequence, b_binary_sequence

def run():
    tf.random.set_seed(1)
    np.random.seed(1)
    memory_size = 100000
    training_epoch = 80000
    batch_size = 128
    adj_matrix = np.load('adj_matrix.npy')
    node_attribute_all = np.load('training_data.npy')/600
    node_attribute_all[:, 1:]
    node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 90))
    attribute_length = 10
    key_length = 8
    num_agent = 2
    reward_all = []
    num_node = len(adj_matrix)
    step_length = node_attribute_all.shape[1]-50
    alice = Agent(name='Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    bob = Agent(name='Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, num_node=num_node)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 0.4
    memory = Buffer(buffer_capacity=memory_size, batch_size=batch_size, num_agent=num_agent, num_obs=attribute_length, num_node=num_node, num_act=key_length)
    alice_idx = 1
    bob_idx = 4
    for epoch in range(training_epoch):
        reward_epoch = 0
        p_value_epoch = 0
        score_epoch = 0
        for step in range(step_length):
            node_attribute = node_attribute_all[:, step:step+attribute_length]
            state = node_attribute
            alice_attribute = node_attribute[alice_idx]
            bob_attribute = node_attribute[bob_idx]
            obs_alice = alice_attribute
            obs_bob = bob_attribute
            act_alice = np.clip(tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))) + np.random.randn(key_length) * std_dev, 0, 1)
            act_bob = np.clip(tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))) + np.random.randn(key_length) * std_dev, 0, 1)
            p_value, score = get_score(act_alice, act_bob)
            reward = p_value + score
            p_value_epoch +=p_value
            score_epoch += score
            reward_epoch += reward
            state_ = node_attribute_all[:, step+1:step+1+attribute_length]
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

                alice.target_update()
                bob.target_update()
        reward_all.append(reward_epoch/step_length)
        print('epoch:%s, start:%s, reward:[p_value:%s, score:%s]' % (epoch, start_flag, p_value_epoch/step_length, score_epoch/step_length))
        if (epoch + 1) % 100 == 0 and start_flag is True:
            np.save("reward_new.npy", reward_all)
            alice.save_model()
            bob.save_model()
        if (epoch+1) % 100 == 0 and start_flag is True:
            reward_epoch = 0
            p_value_epoch = 0
            score_epoch = 0
            for step in range(node_attribute_all.shape[1]-10):
                node_attribute = node_attribute_all[:, step:step + attribute_length]
                alice_attribute = node_attribute[alice_idx]
                bob_attribute = node_attribute[bob_idx]
                obs_alice = alice_attribute
                obs_bob = bob_attribute
                act_alice = np.clip(tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))), 0, 1)
                act_bob = np.clip(tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))), 0, 1)
                p_value, score = get_score(act_alice, act_bob)
                a_t, b_t = get_correction(act_alice, act_bob)
                reward = p_value + score
                p_value_epoch += p_value
                score_epoch += score
                reward_epoch += reward
                print('step:%s, reward:[p_value:%s, score:%s]' % (step, p_value, score))
                print('alice:%s', a_t)
                print('bob:%s', b_t)

            x = np.arange(len(reward_all))
            plt.plot(x, reward_all, label='reward')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()










if __name__ == '__main__':
    run()