import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from functions import *

def plot_pursuit_trajectory(r_t: np.ndarray, r_p: np.ndarray, 
                          min_r: float, maneuver_time: float, 
                          model_name: str) -> None:
    """Plot the pursuit trajectory showing predator and target positions."""
    plt.figure(figsize=(10, 8))
    
    # Plot full trajectories
    plt.plot(r_t[:, 0], r_t[:, 1], 'gray', label='Target', alpha=0.5)
    plt.plot(r_p[:, 0], r_p[:, 1], 'blue', label='Predator', alpha=0.5)
    
    # Mark start positions
    plt.plot(r_t[0, 0], r_t[0, 1], 'ko', label='Target start')
    plt.plot(r_p[0, 0], r_p[0, 1], 'bo', label='Predator start')
    
    # Add points every 100ms to show timing
    for i in range(0, len(r_t),1):  # dt=0.0167 (60Hz)
        plt.plot(r_t[i, 0], r_t[i, 1], 'k.')
        plt.plot(r_p[i, 0], r_p[i, 1], 'b.')
    
    plt.title(f'{model_name} - Minimum Distance: {min_r:.2f} cm')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()

# Run simulation with these parameters
params = {
    'model': 'pPP',        # Predictive pursuit model
    'lag': 0.042,         # 42ms sensory-motor delay
    'dt': 0.0167,         # 60Hz update rate
    's_p': 100,           # Predator speed (cm/s)
    's_t': 80,            # Target speed (cm/s)
    'maneuver_time': 0.2, # Target maneuvers when predator is 0.2s away
    'maneuver_angle': np.pi/4,  # 45-degree turn
    'k':43.0,           # Gain parameter
    'tMax': 1.0,         # Maximum simulation time (s)
    'f_time': 0.042   # Forecast time (equal to sensory-motor delay)
}

# Run simulation
r_t, v_t, r_p, v_p, min_r, h, t = run_sim(**params)

# Plot results of the simulation
plot_pursuit_trajectory(r_t, r_p, min_r, params['maneuver_time'], 
                       f"Predictive Pursuit (k={params['k']})")

# Print  statistics of the simulations
print(f"Simulation Statistics:")
print(f"Minimum distance to target: {min_r:.2f} cm")
print(f"Simulation duration: {t*params['dt']:.2f} seconds")
print(f"Final distance to target: {np.sqrt(np.sum((r_t[-1] - r_p[-1])**2)):.2f} cm")

plt.show()