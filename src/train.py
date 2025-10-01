# src/train.py

import yaml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from environment import WindFarmEnv
from agent import create_actor_network, create_critic_network, Buffer
import os
from datetime import datetime

# --- 1. Load Configuration ---
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# --- 2. Initialize Environment and Models ---
TERRAIN_FILE = 'data/terrain.tif'
WIND_DATA_FILE = 'data/wind_data.lib'
# <<< MODIFIED: Pass the config dictionary to the environment
env = WindFarmEnv(
    terrain_path=TERRAIN_FILE, 
    wind_data_path=WIND_DATA_FILE, 
    config=config
)

actor_model = create_actor_network(env.observation_space, env.action_space)
critic_model = create_critic_network(env.observation_space, env.action_space)
target_actor = create_actor_network(env.observation_space, env.action_space)
target_critic = create_critic_network(env.observation_space, env.action_space)
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# <<< MODIFIED: Learning rates from config
critic_lr = config['training']['critic_lr']
actor_lr = config['training']['actor_lr']
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# --- 3. Initialize Hyperparameters from Config ---
# <<< MODIFIED: All hyperparameters now come from the config file
total_episodes = config['training']['total_episodes']
gamma = config['training']['gamma']
tau = config['training']['tau']
num_random_steps = config['exploration']['num_random_steps']
noise_std_dev = config['exploration']['noise_std_dev_factor'] * env.action_space.high
buffer_config = config['buffer']

buffer = Buffer(
    buffer_capacity=buffer_config['capacity'], 
    batch_size=buffer_config['batch_size'],
    obs_shape=env.observation_space.shape, 
    num_actions=env.action_space.shape[0]
)

@tf.function
def update_networks(state_batch, action_batch, reward_batch, next_state_batch):
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))
    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        actor_loss = -tf.math.reduce_mean(critic_value)
    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
        
# --- 4. The Main Training Loop ---
ep_reward_list = []
best_layout = None
best_aep = 0.0

for ep in range(total_episodes): 
    prev_state, _ = env.reset()
    episodic_reward = 0
    done = False
    truncated = False

    while not done and not truncated:
        if buffer.buffer_counter < num_random_steps:
            action = env.action_space.sample()
        else:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = actor_model(tf_prev_state)[0].numpy()
            noise = np.random.normal(0, noise_std_dev, size=action.shape) 
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)

        state, reward, done, truncated, info = env.step(action)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        if buffer.buffer_counter > buffer.batch_size:
            state_batch, action_batch, reward_batch, next_state_batch = buffer.sample()
            update_networks(state_batch, action_batch, reward_batch, next_state_batch)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-20:])
    
    current_aep = env.calculate_aep(env.turbine_coords)
    if current_aep > best_aep:
        best_aep = current_aep
        best_layout = env.turbine_coords.copy()

    print(f"Episode {ep+1}/{total_episodes} | Turbines: {len(env.turbine_coords)} | AEP: {current_aep:.2f} | Avg AEP: {avg_reward:.2f}")

print("\n--- Archiving Results ---")

# Create a unique timestamp string for this run, e.g., "2025-10-01_21-30-05"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_folder = os.path.join('outputs', timestamp)
os.makedirs(run_folder, exist_ok=True) # Create the unique folder for this run

# 1. Save the final plot to the new folder
plot_path = os.path.join(run_folder, 'final_layout.png')
print(f"Generating final layout plot...")

plt.figure(figsize=(10, 10))
if best_layout:
    x_coords = [coord[0] for coord in best_layout]
    y_coords = [coord[1] for coord in best_layout]
    plt.scatter(x_coords, y_coords, c='blue', s=100, label='Turbines')
    for i, (x, y) in enumerate(best_layout):
        plt.annotate(f'T{i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.title(f'Best Layout Found (AEP: {best_aep:.2f} MWh/year)')
plt.xlabel('X-coordinate (m)')
plt.ylabel('Y-coordinate (m)')
plt.xlim(0, env.farm_dims[0])
plt.ylim(0, env.farm_dims[1])
plt.grid(True)
plt.legend()
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()

# 2. Save the config file used for this run, for reproducibility
config_path = os.path.join(run_folder, 'config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f)
print(f"Config used for this run saved to {config_path}")

# 3. Save the training rewards log
log_path = os.path.join(run_folder, 'training_log.txt')
with open(log_path, 'w') as f:
    f.write("Episodic Rewards:\n")
    for reward in ep_reward_list:
        f.write(f"{reward}\n")
print(f"Training rewards log saved to {log_path}")
