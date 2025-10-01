# src/environment.py

import rasterio
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WindFarmEnv(gym.Env):
    def __init__(self, terrain_path, wind_data_path, config):
        super().__init__()
        
        print("Initializing Wind Farm Environment from config...")
        self.config = config # Store the config
        self.terrain_elevation = self._load_terrain(terrain_path)
        self.wind_data = self._parse_wind_data(wind_data_path)
        
        # --- Turbine Specifications (from config) ---
        turbine_config = self.config['turbine'] 
        self.hub_height = turbine_config['hub_height']
        self.rotor_diameter = turbine_config['rotor_diameter']
        self.rated_power = turbine_config['rated_power']
        self.cut_in_speed = turbine_config['cut_in_speed']
        self.rated_speed = turbine_config['rated_speed']
        self.cut_out_speed = turbine_config['cut_out_speed']
        
        # --- RL Environment State (from config) ---
        env_config = self.config['environment']
        self.max_turbines = turbine_config['max_turbines']
        self.farm_dims = tuple(env_config['farm_dims'])
        self.min_distance = env_config['min_distance_rotors'] * self.rotor_diameter

        self.turbine_coords = []
        self.current_step = 0

        # --- Define Spaces (from config) ---
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array(self.farm_dims), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1, 
            high=max(self.farm_dims), 
            shape=(self.max_turbines, 2), 
            dtype=np.float32
        )
        print("Environment initialized successfully.")
    
    # ... (The _load_terrain and _parse_wind_data methods are unchanged) ...
    def _load_terrain(self, file_path):
        """Loads the terrain elevation data from a GeoTIFF file."""
        print(f"Loading terrain data from {file_path}...")
        with rasterio.open(file_path) as dataset:
            elevation = dataset.read(1)
            elevation[elevation < -1000] = np.nan
        return elevation

    def _parse_wind_data(self, file_path):
        """Custom parser for GWA (.lib/.gwc) files."""
        print(f"Loading wind data from {file_path}...")
        with open(file_path, 'r') as f:
            lines = f.readlines()

        dims = [int(d) for d in lines[1].strip().split()]
        num_heights, num_sectors = dims[1], dims[2]
        
        heights = [float(h) for h in lines[3].strip().split()]
        
        try:
            target_height = 100.0
            height_index = heights.index(target_height)
        except ValueError:
            target_height = heights[-1]
            height_index = len(heights) - 1
        
        frequencies = [float(x) for x in lines[4].strip().split()]
        
        num_roughnesses = dims[0]
        roughness_index = 0
        start_of_data_lines = 5
        
        speed_line_index = start_of_data_lines + (height_index * num_roughnesses) + roughness_index
        mean_speeds = [float(s) for s in lines[speed_line_index].strip().split()]
        
        wind_df = pd.DataFrame({
            'Sector': np.arange(1, num_sectors + 1),
            'Direction': np.arange(0, 360, 30),
            'Frequency': frequencies,
            f'Mean_Wind_Speed_{int(target_height)}m': mean_speeds
        })
        return wind_df

    def reset(self, seed=None, options=None):
        # (This method is unchanged)
        super().reset(seed=seed)
        self.current_step = 0
        self.turbine_coords = []
        observation = np.full(self.observation_space.shape, -1, dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        # <<< MODIFIED: The min_distance logic is now cleaner
        is_too_close = False
        for coord in self.turbine_coords:
            dist = np.linalg.norm(np.array(action) - np.array(coord))
            if dist < self.min_distance: # <<< MODIFIED: Uses self.min_distance
                is_too_close = True
                break
        
        if is_too_close:
            observation = np.full(self.observation_space.shape, -1, dtype=np.float32)
            if self.turbine_coords:
                observation[:len(self.turbine_coords)] = np.array(self.turbine_coords)
            
            reward = -10000.0
            truncated = True
            done = True
            info = {'status': 'invalid_placement'}
            return observation, reward, done, truncated, info

        prev_aep = self.calculate_aep(self.turbine_coords)
        self.turbine_coords.append(action)
        self.current_step += 1
        new_aep = self.calculate_aep(self.turbine_coords)
        reward = new_aep - prev_aep
        
        done = self.current_step >= self.max_turbines
        truncated = False
        
        observation = np.full(self.observation_space.shape, -1, dtype=np.float32)
        if self.turbine_coords:
            observation[:len(self.turbine_coords)] = np.array(self.turbine_coords)
            
        info = {'status': 'valid_placement'}
        return observation, reward, done, truncated, info
        
    # ... (The physics simulation methods _get_power, _jensen_wake_model, and calculate_aep are unchanged) ...
    def _get_power(self, wind_speed):
        """Calculates power output using a simplified power curve."""
        if self.cut_in_speed <= wind_speed < self.rated_speed:
            return self.rated_power * ((wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed))**3
        elif self.rated_speed <= wind_speed < self.cut_out_speed:
            return self.rated_power
        else:
            return 0.0

    def _jensen_wake_model(self, dist_downwind, dist_crosswind):
        """Calculates wind speed deficit using the Jensen wake model."""
        wake_expansion_coeff = 0.05
        rotor_radius = self.rotor_diameter / 2.0
        wake_radius = rotor_radius + wake_expansion_coeff * dist_downwind
        
        if abs(dist_crosswind) < wake_radius:
            velocity_deficit = (2/3) * (rotor_radius / wake_radius)**2
            return velocity_deficit
        else:
            return 0.0
        
    def calculate_aep(self, turbine_coords):
        """Calculates the Annual Energy Production for a given layout."""
        if not turbine_coords:
            return 0.0
        
        n_turbines = len(turbine_coords)
        total_energy = 0.0
        hours_in_year = 8760.0
        coords = np.array(turbine_coords)
        wind_speed_col = self.wind_data.columns[-1]

        for _, row in self.wind_data.iterrows():
            wind_dir_deg = row['Direction']
            wind_speed = row[wind_speed_col]
            frequency = row['Frequency'] / 100.0
            wind_dir_rad = np.deg2rad(wind_dir_deg)
            cos_dir, sin_dir = np.cos(-wind_dir_rad), np.sin(-wind_dir_rad)
            rotation_matrix = np.array([[cos_dir, -sin_dir], [sin_dir, cos_dir]])
            rotated_coords = coords @ rotation_matrix.T
            
            total_power_for_dir = 0.0
            for i in range(n_turbines):
                velocity_deficits_sq = 0.0
                for j in range(n_turbines):
                    if i == j: continue
                    dist_downwind = rotated_coords[i, 0] - rotated_coords[j, 0]
                    if dist_downwind > 0:
                        dist_crosswind = rotated_coords[i, 1] - rotated_coords[j, 1]
                        deficit = self._jensen_wake_model(dist_downwind, dist_crosswind)
                        velocity_deficits_sq += deficit**2
                
                total_deficit = np.sqrt(velocity_deficits_sq)
                waked_wind_speed = wind_speed * (1 - total_deficit)
                total_power_for_dir += self._get_power(waked_wind_speed)
            
            total_energy += total_power_for_dir * frequency * hours_in_year

        return total_energy / 1000.0