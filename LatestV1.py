import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from LatestModelV1 import Agent, Buffer
from nistrng import *


def get_score(alice_act, bob_act, cal_act):
    alice_act = np.reshape(alice_act, (2, int(alice_act.shape[0] / 2)))
    bob_act = np.reshape(bob_act, (2, int(bob_act.shape[0] / 2)))
    cal_act = np.reshape(cal_act, (2, int(cal_act.shape[0] / 2)))
    k_a_b = np.zeros(alice_act.shape[1], dtype=np.int)
    k_a_c = np.zeros(alice_act.shape[1], dtype=np.int)
    k_b_a = np.zeros(bob_act.shape[1], dtype=np.int)
    k_b_c = np.zeros(bob_act.shape[1], dtype=np.int)
    k_c_a = np.zeros(cal_act.shape[1], dtype=np.int)
    k_c_b = np.zeros(cal_act.shape[1], dtype=np.int)
    s_ab, s_ac, s_bc = 0, 0, 0
    for i in range(alice_act.shape[1]):
        if alice_act[0][i] >= 0:
            k_a_b[i] = 1
        if alice_act[1][i] >= 0:
            k_a_c[i] = 1
        if bob_act[0][i] >= 0:
            k_b_a[i] = 1
        if bob_act[1][i] >= 0:
            k_b_c[i] = 1
        if cal_act[0][i] >= 0:
            k_c_a[i] = 1
        if cal_act[1][i] >= 0:
            k_c_b[i] = 1

        if k_a_b[i] == k_b_a[i]:
            s_ab+=1
        if k_a_c[i] == k_c_a[i]:
            s_ac+=1
        if k_b_c[i] == k_c_b[i]:
            s_bc +=1
    def get_p_value(act):
        p_t = 0
        if np.sum(act) == 0 or np.sum(act) == len(act):
            p_t = 0
        else:
            eligible_battery: dict = check_eligibility_all_battery(act, SP800_22R1A_BATTERY)
            results = run_all_battery(act, eligible_battery, False)
            p_t = results[0][0].score
            if p_t >= 0.01:
                p_t = 0.01
        return p_t
    p_ab = get_p_value(k_a_b)
    p_ac = get_p_value(k_a_c)
    p_ba = get_p_value(k_b_a)
    p_bc = get_p_value(k_b_c)
    p_ca = get_p_value(k_c_a)
    p_cb = get_p_value(k_c_b)
    p_ave = (p_ab+p_ac+p_bc+p_ba+p_ca+p_cb)/6
    s_ave = (s_ab/alice_act.shape[1] + s_ac/alice_act.shape[1] + s_bc/alice_act.shape[1])/3
    return s_ave, p_ave, k_a_b, k_a_c, k_b_a, k_b_c, k_c_a, k_c_b


def get_check(step, current_key, k_0, k_1):
    count = 0
    key_all = np.vstack((k_0, k_1))
    for i in range(step):
        kk = current_key[i, :, :]
        kl = key_all
        if (kk[0]==kl[0]).all() == True:
            count += 1
        if (kk[1]==kl[1]).all() == True:
            count += 1
    return count

def run():
    tf.random.set_seed(1)
    np.random.seed(1)
    memory_size = 50000
    training_epoch = 50000
    batch_size = 128
    adj_matrix = np.load('adj_matrix.npy')
    node_attribute_all = np.load('test.npy')#/300
    node_attribute_all = node_attribute_all[:, 1:]
    #node_attribute_all = np.random.normal(loc=0, scale=1, size=(11, 45, 5))
    node_attribute_all = np.reshape(node_attribute_all, (36, 6, 20))
    #node_attribute_all = np.random.sample(size=(11, 45, 5))*150+50
    attribute_length = 20
    key_length = 8
    num_agent = 3
    reward_all = []
    num_node = len(adj_matrix)
    step_length = 5
    alice = Agent(name='Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent)
    bob = Agent(name='Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent)
    cal = Agent(name='Cal', num_obs=attribute_length, num_act=key_length, num_agent=num_agent)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 1
    #memory = Buffer(buffer_capacity=memory_size, num_agent=num_agent, num_obs=attribute_length, num_act=key_length)
    memory = Buffer(buffer_capacity=memory_size, num_agent=num_agent, num_obs=attribute_length, num_act=key_length)
    alice_idx = 1
    bob_idx = 10
    cal_idx = 17
    for epoch in range(training_epoch):
        reward_epoch = 0
        alice_current_key = np.zeros((step_length, num_agent-1, key_length), dtype=np.int)
        bob_current_key = np.zeros((step_length, num_agent-1, key_length), dtype=np.int)
        cal_current_key = np.zeros((step_length, num_agent-1, key_length), dtype=np.int)
        for step in range(step_length):
            node_attribute = node_attribute_all[:, step, :]
            alice_attribute = (node_attribute[alice_idx]-min(node_attribute[alice_idx]))
            bob_attribute = (node_attribute[bob_idx]-min(node_attribute[bob_idx]))
            cal_attribute = (node_attribute[cal_idx] - min(node_attribute[cal_idx]))
            obs_alice = alice_attribute
            obs_bob = bob_attribute
            obs_cal = cal_attribute
            state = np.vstack((obs_alice, obs_bob, obs_cal))
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            act_alice = np.clip(np.random.normal(act_alice, std_dev), -1, 1)
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            act_bob = np.clip(np.random.normal(act_bob, std_dev), -1, 1)
            act_cal = tf.squeeze(cal.actor(tf.expand_dims(tf.convert_to_tensor(obs_cal), 0))).numpy()
            act_cal = np.clip(np.random.normal(act_cal, std_dev), -1, 1)

            s_ave, p_ave, k_a_b, k_a_c, k_b_a, k_b_c, k_c_a, k_c_b = get_score(act_alice, act_bob, act_cal)

            alice_pen = get_check(step, alice_current_key, k_a_b, k_a_c)
            bob_pen = get_check(step, bob_current_key, k_b_a, k_b_c)
            cal_pen = get_check(step, cal_current_key, k_c_a, k_c_b)

            alice_current_key[step, :, :] = np.vstack((k_a_b, k_a_c))
            bob_current_key[step, :, :] = np.vstack((k_b_a, k_b_c))
            cal_current_key[step, :, :] = np.vstack((k_c_a, k_c_b))

            reward = s_ave + p_ave - (alice_pen + bob_pen + cal_pen)/3
            reward_epoch += reward
            node_attribute_ = node_attribute_all[:, step+1, :]

            alice_attribute_ = (node_attribute_[alice_idx] - min(node_attribute_[alice_idx]))
            bob_attribute_ = (node_attribute_[bob_idx] - min(node_attribute_[bob_idx]))
            cal_attribute = (node_attribute_[cal_idx] - min(node_attribute_[cal_idx]))
            obs_alice_ = alice_attribute_
            obs_bob_ = bob_attribute_
            obs_cal_ = cal_attribute
            state_ = np.vstack((obs_alice_, obs_bob_, obs_cal_))
            #action = np.vstack((act_alice, act_bob, act_cal))
            #action = np.concatenate((act_alice,act_bob,act_cal), axis=1)
            action = [act_alice, act_bob, act_cal]
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

                bt_alice_obs, bt_bob_obs, bt_cal_obs = bt_state[:, 0, :], bt_state[:, 1, :], bt_state[:, 2, :]
                bt_alice_obs_, bt_bob_obs_, bt_cal_obs_ = bt_state_[:, 0, :], bt_state_[:, 1, :], bt_state_[:, 2, :]
                bt_alice_act, bt_bob_act, bt_cal_act = bt_action[:, 0, :], bt_action[:, 1, :], bt_action[:, 2, :]
                bt_alice_act_, bt_bob_act_, bt_cal_act_ = alice.actor_target(bt_alice_obs_, training=True), bob.actor_target(bt_bob_obs_, training=True), cal.actor_target(bt_cal_obs_, training=True)
                bt_reward = tf.cast(bt_reward, tf.float32)

                with tf.GradientTape() as tape:
                    Q = alice.critic([bt_state, alice.actor(bt_alice_obs), tf.concat([bt_bob_act, bt_cal_act], 1)], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, alice.actor.trainable_variables)
                alice.actor_optimizer.apply_gradients(zip(actor_grad, alice.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    Q = bob.critic([bt_state, bob.actor(bt_bob_obs), tf.concat([bt_alice_act, bt_cal_act], 1)], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, bob.actor.trainable_variables)
                bob.actor_optimizer.apply_gradients(zip(actor_grad, bob.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    Q = cal.critic([bt_state, cal.actor(bt_cal_obs), tf.concat([bt_alice_act, bt_bob_act], 1)], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, cal.actor.trainable_variables)
                cal.actor_optimizer.apply_gradients(zip(actor_grad, cal.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * alice.critic_target([bt_state_, bt_alice_act_, tf.concat([bt_bob_act_, bt_cal_act_], 1)], training=True)
                    critic_value = alice.critic([bt_state, bt_alice_act, tf.concat([bt_bob_act, bt_cal_act], 1)], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, alice.critic.trainable_variables)
                alice.critic_optimizer.apply_gradients(zip(critic_grad, alice.critic.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * bob.critic_target([bt_state_, bt_bob_act_, tf.concat([bt_alice_act_, bt_cal_act_], 1)], training=True)
                    critic_value = bob.critic([bt_state, bt_bob_act, tf.concat([bt_alice_act, bt_cal_act], 1)], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, bob.critic.trainable_variables)
                bob.critic_optimizer.apply_gradients(zip(critic_grad, bob.critic.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * cal.critic_target([bt_state_, bt_cal_act_, tf.concat([bt_alice_act_, bt_bob_act_], 1)], training=True)
                    critic_value = cal.critic([bt_state, bt_cal_act, tf.concat([bt_alice_act, bt_bob_act], 1)], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, cal.critic.trainable_variables)
                cal.critic_optimizer.apply_gradients(zip(critic_grad, cal.critic.trainable_variables))


                alice.target_update()
                bob.target_update()
                cal.target_update()
        reward_all.append(reward_epoch/step_length)
        print('epoch:%s, start:%s, reward:%s' % (epoch, start_flag, reward_epoch/step_length))
        if (epoch + 1) % 1000 == 0 and start_flag is True:
            np.save("reward_new.npy", reward_all)
            alice.save_model()
            bob.save_model()
            cal.save_model()

            x = np.arange(len(reward_all))
            plt.plot(x, reward_all, label='reward')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()

            alice_current_key = np.zeros((step_length, num_agent - 1, key_length), dtype=np.int)
            bob_current_key = np.zeros((step_length, num_agent - 1, key_length), dtype=np.int)
            cal_current_key = np.zeros((step_length, num_agent - 1, key_length), dtype=np.int)
            reward_epoch = 0
            for step in range(step_length):
                node_attribute = node_attribute_all[:, step, :]
                alice_attribute = (node_attribute[alice_idx] - min(node_attribute[alice_idx]))
                bob_attribute = (node_attribute[bob_idx] - min(node_attribute[bob_idx]))
                cal_attribute = (node_attribute[cal_idx] - min(node_attribute[cal_idx]))
                obs_alice = alice_attribute
                obs_bob = bob_attribute
                obs_cal = cal_attribute
                act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
                act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
                act_cal = tf.squeeze(cal.actor(tf.expand_dims(tf.convert_to_tensor(obs_cal), 0))).numpy()

                s_ave, p_ave, k_a_b, k_a_c, k_b_a, k_b_c, k_c_a, k_c_b = get_score(act_alice, act_bob, act_cal)

                alice_pen = get_check(step, alice_current_key, k_a_b, k_a_c)
                bob_pen = get_check(step, bob_current_key, k_b_a, k_b_c)
                cal_pen = get_check(step, cal_current_key, k_c_a, k_c_b)

                alice_current_key[step, :, :] = np.vstack((k_a_b, k_a_c))
                bob_current_key[step, :, :] = np.vstack((k_b_a, k_b_c))
                cal_current_key[step, :, :] = np.vstack((k_c_a, k_c_b))

                reward = -(alice_pen + bob_pen + cal_pen) / 3 + s_ave + p_ave
                reward_epoch += reward
                print('step:%s,a_b:%s'% (step, [k_a_b, k_b_a]))
                print('step:%s,a_c:%s'% (step, [k_a_c, k_c_a]))
                print('step:%s,b_c:%s'% (step, [k_b_c, k_c_b]))
            print('reward:', reward_epoch/(step_length))












if __name__ == '__main__':
    run()