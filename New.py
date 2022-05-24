import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from LatestModelV1 import Agent, Buffer
from nistrng import *
import os
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples

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
            p_t = np.sum([results[i][0].score for i in range(len(results))])/len(results)
            p_t = min(p_t, 0.01)
        return p_t
    p_ab = get_p_value(k_a_b)
    p_ac = get_p_value(k_a_c)
    p_ba = get_p_value(k_b_a)
    p_bc = get_p_value(k_b_c)
    p_ca = get_p_value(k_c_a)
    p_cb = get_p_value(k_c_b)
    p_ave = (p_ab+p_ac+p_bc+p_ba+p_ca+p_cb)/6
    s_ave = (s_ab/alice_act.shape[1] + s_ac/alice_act.shape[1] + s_bc/alice_act.shape[1])/3

    s_a, s_b, s_c = (s_ab+s_ac)/(2*alice_act.shape[1]), (s_ab+s_bc)/(2*alice_act.shape[1]), (s_ac+s_bc)/(2*alice_act.shape[1])
    p_a, p_b, p_c = (p_ab+p_ac)/2, (p_ab+p_bc)/2, (p_ac+p_bc)/2
    #return s_a, s_b, s_c, p_a, p_b, p_c, k_a_b, k_a_c, k_b_a, k_b_c, k_c_a, k_c_b
    return s_ave, p_ave, k_a_b, k_a_c, k_b_a, k_b_c, k_c_a, k_c_b


def get_check(step, current_key, k_0, k_1):
    count = 0
    key_all = np.vstack((k_0, k_1))
    for i in range(step):
        kk = current_key[i, :, :]
        kl = key_all
        kkk = kl[0]
        kkkk = kl[1]
        if (kk[0]==kl[0]).all() == True:
            count += 1
        if (kk[1]==kl[1]).all() == True:
            count += 1
    return count

def get_data(es, network_nodes):
    num_nodes = len(network_nodes)
    es.run()
    sample_length = len(es.network.nodes[1].results[Node.value_type['EN_PRESSURE']])

    tyep_pressure = Node.value_type['EN_PRESSURE']
    tyep_head = Node.value_type['EN_HEAD']
    tyep_demand = Node.value_type['EN_DEMAND']
    tyep_elevation = Node.value_type['EN_ELEVATION']
    tyep_max_level = Node.value_type['EN_MAXLEVEL']
    tyep_min_level = Node.value_type['EN_MINLEVEL']
    pressure_data = np.zeros((num_nodes, sample_length))
    head_data = np.zeros((num_nodes, sample_length))
    demand_data = np.zeros((num_nodes, sample_length))
    elevation_data = np.zeros(num_nodes)
    max_level_data = np.zeros(num_nodes)
    min_level_data = np.zeros(num_nodes)
    for n in range(num_nodes):
        elevation_data[n] = es.network.nodes[n + 1].results[tyep_elevation][0]
        max_level_data[n] = es.network.nodes[n + 1].results[tyep_max_level][0]
        min_level_data[n] = es.network.nodes[n + 1].results[tyep_min_level][0]

        for step in range(sample_length):
            pressure_data[n][step] = es.network.nodes[n + 1].results[tyep_pressure][step]
            demand_data[n][step] = es.network.nodes[n + 1].results[tyep_demand][step]
            head_data[n][step] = es.network.nodes[n + 1].results[tyep_head][step]
    return pressure_data, head_data, demand_data

def run():
    tf.random.set_seed(1)
    np.random.seed(1)
    training_epoch = 80000
    memory_size = 100000
    batch_size = 256
    file = os.path.join(os.path.dirname(simple.__file__), 'FinalV1.inp')  # open an example
    es = EPANetSimulation(file)
    network_nodes = [es.network.nodes[x].id for x in list(es.network.nodes)]
    pressure_data_all, head_data_all, demand_data_all = get_data(es, network_nodes)
    pressure_data_all, _, _ = pressure_data_all[:, 2:], head_data_all[:, 2:], demand_data_all[:, 2:]
    network_nodes = [es.network.nodes[x].id for x in list(es.network.nodes)]
    node_attribute_all = pressure_data_all
    attribute_length = 60
    key_length = 10
    num_agent = 3
    reward_all = []
    step_length = 5
    action_bound = 1
    alice = Agent(name='Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, action_bound=action_bound)
    bob = Agent(name='Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, action_bound=action_bound)
    cal = Agent(name='Cal', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, action_bound=action_bound)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 0.4
    #memory = Buffer(buffer_capacity=memory_size, num_agent=num_agent, num_obs=attribute_length, num_act=key_length)
    memory = Buffer(buffer_capacity=memory_size, num_agent=num_agent, num_obs=attribute_length, num_act=key_length)
    alice_id = '121'
    bob_id = '141'
    cal_id = '189'
    alice_idx = network_nodes.index(alice_id)
    bob_idx = network_nodes.index(bob_id)
    cal_idx = network_nodes.index(cal_id)
    time_dura = 5
    for epoch in range(training_epoch):
        reward_epoch = 0
        alice_current_key = np.zeros((step_length, num_agent-1, key_length), dtype=np.int)
        bob_current_key = np.zeros((step_length, num_agent-1, key_length), dtype=np.int)
        cal_current_key = np.zeros((step_length, num_agent-1, key_length), dtype=np.int)
        for step in range(step_length):
            node_attribute = node_attribute_all[:, step*time_dura:step*time_dura+attribute_length]
            alice_attribute = (node_attribute[alice_idx] - min(node_attribute[alice_idx])) / (
                    max(node_attribute[alice_idx]) - min(node_attribute[alice_idx]))
            bob_attribute = (node_attribute[bob_idx] - min(node_attribute[bob_idx])) / (
                    max(node_attribute[bob_idx]) - min(node_attribute[bob_idx]))
            cal_attribute = (node_attribute[cal_idx] - min(node_attribute[cal_idx])) / (
                    max(node_attribute[cal_idx]) - min(node_attribute[cal_idx]))
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

            reward = s_ave + p_ave-(alice_pen + bob_pen + cal_pen)/3
            reward_epoch += reward

            node_attribute_ = node_attribute_all[:, (step + 1)*time_dura:(step + 1)*time_dura + attribute_length]
            alice_attribute_ = (node_attribute_[alice_idx] - min(node_attribute_[alice_idx])) / (
                    max(node_attribute_[alice_idx]) - min(node_attribute_[alice_idx]))
            bob_attribute_ = (node_attribute_[bob_idx] - min(node_attribute_[bob_idx])) / (
                    max(node_attribute_[bob_idx]) - min(node_attribute_[bob_idx]))
            cal_attribute_ = (node_attribute_[cal_idx] - min(node_attribute_[cal_idx])) / (
                    max(node_attribute_[cal_idx]) - min(node_attribute_[cal_idx]))
            obs_alice_ = alice_attribute_
            obs_bob_ = bob_attribute_
            obs_cal_ = cal_attribute_
            state_ = np.vstack((obs_alice_, obs_bob_, obs_cal_))
            #action = np.vstack((act_alice, act_bob, act_cal))
            #action = np.concatenate((act_alice,act_bob,act_cal), axis=1)
            action = [act_alice, act_bob, act_cal]
            memory.store(state, action, [reward], state_)
            if memory.buffer_counter >= batch_size*30:
                start_flag = True
                #std_dev *= 0.9999995
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
                    y = bt_reward+ gamma * alice.critic_target([bt_state_, bt_alice_act_, tf.concat([bt_bob_act_, bt_cal_act_], 1)], training=True)
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
        print('epoch:%s, start:%s, reward:%s' % (epoch, start_flag, reward_epoch / step_length))
        if (epoch + 1) % 500 == 0 and start_flag is True:
            np.save("reward_new.npy", reward_all)
            alice.save_model(epoch+1)
            bob.save_model(epoch+1)
            cal.save_model(epoch+1)
            alice_current_key = np.zeros((step_length, num_agent - 1, key_length), dtype=np.int)
            bob_current_key = np.zeros((step_length, num_agent - 1, key_length), dtype=np.int)
            cal_current_key = np.zeros((step_length, num_agent - 1, key_length), dtype=np.int)
            reward_epoch = 0
            for step in range(step_length):
                node_attribute = node_attribute_all[:, step * time_dura:step * time_dura + attribute_length]
                alice_attribute = (node_attribute[alice_idx] - min(node_attribute[alice_idx]))
                bob_attribute = (node_attribute[bob_idx] - min(node_attribute[bob_idx]))
                cal_attribute = (node_attribute[cal_idx] - min(node_attribute[cal_idx]))
                obs_alice = alice_attribute
                obs_bob = bob_attribute
                obs_cal = cal_attribute
                act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
                #act_alice = np.clip(act_alice, -1, 1)
                act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
                #act_bob = np.clip(act_bob, -1, 1)
                act_cal = tf.squeeze(cal.actor(tf.expand_dims(tf.convert_to_tensor(obs_cal), 0))).numpy()
                #act_cal = np.clip(act_cal, -1, 1)

                s_ave, p_ave, k_a_b, k_a_c, k_b_a, k_b_c, k_c_a, k_c_b = get_score(act_alice, act_bob, act_cal)

                alice_pen = get_check(step, alice_current_key, k_a_b, k_a_c)
                bob_pen = get_check(step, bob_current_key, k_b_a, k_b_c)
                cal_pen = get_check(step, cal_current_key, k_c_a, k_c_b)

                alice_current_key[step, :, :] = np.vstack((k_a_b, k_a_c))
                bob_current_key[step, :, :] = np.vstack((k_b_a, k_b_c))
                cal_current_key[step, :, :] = np.vstack((k_c_a, k_c_b))

                reward = s_ave + p_ave - (alice_pen + bob_pen + cal_pen) / 3
                reward_epoch += reward
                print(reward)
                print('a-b:%s', (k_a_b, k_b_a))
                print('a-c:%s', (k_a_c, k_c_a))
                print('b-c:%s', (k_b_c, k_c_b))
                print('s:%s', (s_ave))
                print('p:%s', (p_ave))

            reward_t = np.array(reward_all)
            # np.save("reward_new.npy", reward_t)
            x = np.arange(len(reward_t))
            plt.plot(x, reward_t, label='Alice reward')
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()














if __name__ == '__main__':
    run()