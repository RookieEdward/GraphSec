import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from keras import backend as k

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity, batch_size, num_agent, num_obs, num_node, num_act):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((buffer_capacity, num_agent, num_obs))
        self.action_buffer = np.zeros((buffer_capacity, num_agent*num_act))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_agent, num_obs))

    # Takes (s,a,r,s') obervation tuple as input
    def store(self, state, action, reward, state_):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = state_
        self.buffer_counter += 1


class Agent:
    def __init__(self, name, num_obs, num_act, num_agent, num_node):
        self.name=name
        self.num_obs=num_obs
        self.num_act=num_act
        self.num_agent=num_agent
        self.tau=0.01

        def get_actor():
            # Initialize weights between -3e-3 and 3-e3
            last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

            inputs = layers.Input(shape=(num_obs,))
            out = layers.Dense(100, activation="elu")(inputs)
            #out = layers.Dropout(0.2)(out)
            out = layers.Dense(100, activation="elu", kernel_initializer=last_init)(out)
            #out = layers.Dropout(0.2)(out)
            out = layers.Dense(100, activation="elu", kernel_initializer=last_init)(out)
            #out = layers.Dropout(0.2)(out)
            out = layers.Dense(100, activation="elu", kernel_initializer=last_init)(out)
            out = layers.Dense(100, activation="elu", kernel_initializer=last_init)(out)
            #outputs = layers.Dense(num_act, activation="sigmoid", kernel_initializer=last_init)(out)
            outputs = layers.Dense(num_act, activation="tanh", kernel_initializer=last_init)(out)
            # Our upper bound is 2.0 for Pendulum.
            outputs = outputs
            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic():
            # State as input
            state_input = layers.Input(shape=(num_agent, num_obs))
            state_output = layers.Dense(100, activation="elu")(state_input)
            state_output = layers.Dense(100, activation="elu")(state_output)
            state_output = layers.Flatten()(state_output)
            #state_output = state_output[:, :, 0]

            # Action as input
            self_act_input = layers.Input(shape=(num_act))
            other_act_input = layers.Input(shape=(num_act))
            action_input = layers.Concatenate()([self_act_input, other_act_input])
            action_out = layers.Dense(100, activation="elu")(action_input)
            # Both are passed through seperate layer before concatenating
            concat = layers.Concatenate()([state_output, action_out])

            action_out = layers.Dense(100, activation="elu")(concat)
            action_out = layers.Dense(100, activation="elu")(action_out)
            action_out = layers.Dense(100, activation="elu")(action_out)
            outputs = layers.Dense(1)(action_out)
            # Outputs single value for give state-action
            model = tf.keras.Model([state_input, self_act_input, other_act_input], outputs)
            return model

        self.actor = get_actor()
        self.critic = get_critic()
        self.actor_target = get_actor()
        self.critic_target = get_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())


    def save_model(self):
        self.actor.save(self.name+'_actor')
        self.critic.save(self.name+'_critic')


    def target_update(self):
        for (a, b) in zip(self.actor_target.variables, self.actor.variables):
            a.assign(b * self.tau + a * (1 - self.tau))
        for (a, b) in zip(self.critic_target.variables, self.critic.variables):
            a.assign(b * self.tau + a * (1 - self.tau))



class AgentRestore:
    def __init__(self, name, num_obs, num_act, num_agent, num_node):
        self.name=name
        self.num_obs = num_obs
        self.num_act = num_act
        self.num_agent = num_agent
        self.tau = 0.01

        def restore_actor():
            model = keras.models.load_model(self.name+'_actor')
            return model

        def restore_critic():
            model = keras.models.load_model(self.name + '_critic')
            return model

        self.actor = restore_actor()
        self.critic = restore_critic()
        self.actor_target = restore_actor()
        self.critic_target = restore_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())




    def target_update(self):
        for (a, b) in zip(self.actor_target.variables, self.actor.variables):
            a.assign(b * self.tau + a * (1 - self.tau))
        for (a, b) in zip(self.critic_target.variables, self.critic.variables):
            a.assign(b * self.tau + a * (1 - self.tau))
















