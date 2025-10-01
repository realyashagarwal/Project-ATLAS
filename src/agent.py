# src/agent.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, obs_shape=(10, 2), num_actions=2):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.obs_shape = obs_shape

        # The state buffers store the flattened observation
        num_states = np.prod(obs_shape)
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0].flatten()
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3].flatten()
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # <<< MODIFIED: Reshape using the stored obs_shape
        state_batch = tf.reshape(state_batch, [self.batch_size] + list(self.obs_shape))
        next_state_batch = tf.reshape(next_state_batch, [self.batch_size] + list(self.obs_shape))

        return state_batch, action_batch, reward_batch, next_state_batch

def create_actor_network(observation_space, action_space):
    """
    Creates the Actor network.
    Takes the state (observation) as input and outputs an action.
    """
    # The last layer's activation is tanh to keep outputs between -1 and 1.
    # We will then scale this to our farm's dimensions.
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=observation_space.shape)
    # Flatten the input for the dense layers
    net = layers.Flatten()(inputs)
    net = layers.Dense(256, activation="relu")(net)
    net = layers.Dense(256, activation="relu")(net)
    # The output layer has 2 nodes (x, y) with a tanh activation
    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(net)

    # Scale the output from [-1, 1] to [0, farm_dim]
    farm_dimensions = action_space.high
    outputs = outputs * farm_dimensions

    model = tf.keras.Model(inputs, outputs)
    return model


def create_critic_network(observation_space, action_space):
    """
    Creates the Critic network.
    Takes state and action as input and outputs a Q-value.
    """
    # State pathway
    state_input = layers.Input(shape=observation_space.shape)
    state_net = layers.Flatten()(state_input)
    state_net = layers.Dense(16, activation="relu")(state_net)
    state_net = layers.Dense(32, activation="relu")(state_net)

    # Action pathway
    action_input = layers.Input(shape=action_space.shape)
    action_net = layers.Dense(32, activation="relu")(action_input)

    # Combine both pathways
    concat = layers.Concatenate()([state_net, action_net])
    net = layers.Dense(256, activation="relu")(concat)
    net = layers.Dense(256, activation="relu")(net)
    outputs = layers.Dense(1)(net) # Output is a single Q-value

    model = tf.keras.Model([state_input, action_input], outputs)
    return model