import numpy as np
import matplotlib.pyplot as plt

def calculate_visual_angle(distance, predator_width=5):
    """Calculate visual angle (in degrees) based on predator width and distance"""
    return 2 * np.arctan(predator_width / (2 * distance)) * 180 / np.pi

def calculate_threshold_distance(threshold_angle, predator_width=2.5):
    """Calculate distance at which visual angle reaches threshold"""
    return predator_width / (2 * np.tan(threshold_angle * np.pi / 360))

def calculate_los_angle(predator_pos, prey_pos):
    """Calculate line of sight angle between predator and prey"""
    dx = prey_pos[0] - predator_pos[0]
    dy = prey_pos[1] - predator_pos[1]
    return np.arctan2(dy, dx)

def calculate_relative_velocity(predator_pos, predator_vel, prey_pos, prey_vel):
    """Calculate relative velocity between predator and prey"""
    return [prey_vel[0] - predator_vel[0], prey_vel[1] - predator_vel[1]]

def choose_target_prey(predator_pos, prey_positions, prey_velocities, predator_width=5):
    """Choose which prey to pursue based on distance and visual angle"""
    distances = []
    visual_angles = []
    
    for prey_pos in prey_positions:
        dist = np.sqrt(np.sum((prey_pos - predator_pos)**2))
        distances.append(dist)
        angle = calculate_visual_angle(dist, predator_width)
        visual_angles.append(angle)
    
    # Create a simple scoring system based on distance and visual angle
    scores = np.array(visual_angles) / np.array(distances)
    return np.argmax(scores)  # Return index of prey with highest score

def simulate_pursuit(predator_speed, escape_angle, 
                    prey_speed=8, threshold_angle=14.0,
                    predator_width=5, latency=0.42,
                    navigation_gain=3.0, time_step=0.005, 
                    total_time=4.0, min_distance=1.0):
    """
    Simulate predator-prey encounter with proportional navigation
    
    Args:
        predator_speed (float): Initial speed of predator in cm/s
        escape_angle (float): Initial escape angle of prey in radians
        prey_speed (float): Speed of prey escape in cm/s
        threshold_angle (float): Visual angle threshold in degrees
        predator_width (float): Width of predator in cm
        latency (float): Prey response latency in seconds
        navigation_gain (float): Proportional navigation gain
        time_step (float): Time step for simulation in seconds
        total_time (float): Total simulation time in seconds
        min_distance (float): Minimum distance for simulation to continue
    """
    # Initialize time array
    time = np.arange(0, total_time, time_step)
    n_steps = len(time)
    
    # Initialize arrays for storing simulation data
    predator_pos = np.zeros((n_steps, 2))  # x, y positions
    predator_vel = np.zeros((n_steps, 2))  # x, y velocities
    prey_pos = np.zeros((n_steps, 2))
    prey_vel = np.zeros((n_steps, 2))
    distances = np.zeros(n_steps)
    los_angles = np.zeros(n_steps)
    visual_angles = np.zeros(n_steps)
    
    # Set initial conditions
    threshold_distance = calculate_threshold_distance(threshold_angle, predator_width)
    predator_pos[0] = [-threshold_distance, 0]
    predator_vel[0] = [predator_speed, 0]  # Initially moving along x-axis
    
    # Initialize predator heading
    predator_heading = 0  # Initial heading along x-axis
    
    # Response index for prey
    response_idx = int(latency / time_step)
    escape_triggered = False
    
    # Simulation loop
    for i in range(1, n_steps):
        # Calculate current distance and LOS angle
        distances[i-1] = np.sqrt(np.sum((prey_pos[i-1] - predator_pos[i-1])**2))
        los_angles[i-1] = calculate_los_angle(predator_pos[i-1], prey_pos[i-1])
        visual_angles[i-1] = calculate_visual_angle(distances[i-1], predator_width)
        
        # Break if predator catches prey
        if distances[i-1] < min_distance:
            break
            
        # Check if prey should start escaping
        if distances[i-1] <= threshold_distance and not escape_triggered:
            escape_triggered = True
            escape_start_idx = i + response_idx
            
        # Update prey position and velocity
        if escape_triggered and i >= escape_start_idx:
            prey_vel[i] = [prey_speed * np.cos(escape_angle),
                          prey_speed * np.sin(escape_angle)]
        prey_pos[i] = prey_pos[i-1] + prey_vel[i] * time_step
        
        # Update predator using proportional navigation
        if i > 0:
            # Calculate line-of-sight rate
            los_rate = (los_angles[i-1] - los_angles[i-2]) / time_step
            
            # Update predator heading using PN law
            heading_rate = navigation_gain * los_rate
            predator_heading += heading_rate * time_step
            
            # Update predator velocity and position
            predator_vel[i] = [predator_speed * np.cos(predator_heading),
                             predator_speed * np.sin(predator_heading)]
            predator_pos[i] = predator_pos[i-1] + predator_vel[i] * time_step
    
    # Calculate final distance
    distances[-1] = np.sqrt(np.sum((prey_pos[-1] - predator_pos[-1])**2))
    
    # Trim arrays to valid simulation time
    valid_idx = max(1, np.where(distances < min_distance)[0][0]) if np.any(distances < min_distance) else n_steps
    
    return {
        'time': time[:valid_idx],
        'predator_pos': predator_pos[:valid_idx],
        'predator_vel': predator_vel[:valid_idx],
        'prey_pos': prey_pos[:valid_idx],
        'prey_vel': prey_vel[:valid_idx],
        'distances': distances[:valid_idx],
        'los_angles': los_angles[:valid_idx],
        'visual_angles': visual_angles[:valid_idx],
        'min_distance': np.min(distances),
        'capture': np.min(distances) < min_distance
    }

def simulate_multi_prey_pursuit(predator_speed, escape_angles, initial_prey_positions,
                              prey_speed=8, threshold_angle=14.0, predator_width=5,
                              latency=0.42, navigation_gain=3.0, time_step=0.005,
                              total_time=4.0, min_distance=1.0):
    """
    Simulate predator pursuing multiple prey using proportional navigation
    
    Args:
        predator_speed (float): Initial speed of predator in cm/s
        escape_angles (list): List of escape angles for each prey in radians
        initial_prey_positions (list): List of initial positions for each prey
        prey_speed (float): Speed of prey escape in cm/s
        threshold_angle (float): Visual angle threshold in degrees
        predator_width (float): Width of predator in cm
        latency (float): Prey response latency in seconds
        navigation_gain (float): Proportional navigation gain
        time_step (float): Time step for simulation in seconds
        total_time (float): Total simulation time in seconds
        min_distance (float): Minimum distance for simulation to continue
    """
    time = np.arange(0, total_time, time_step)
    n_steps = len(time)
    n_prey = len(escape_angles)
    
    # Initialize arrays for storing simulation data
    predator_pos = np.zeros((n_steps, 2))
    predator_vel = np.zeros((n_steps, 2))
    prey_positions = np.zeros((n_steps, n_prey, 2))
    prey_velocities = np.zeros((n_steps, n_prey, 2))
    distances = np.zeros((n_steps, n_prey))
    los_angles = np.zeros((n_steps, n_prey))
    visual_angles = np.zeros((n_steps, n_prey))
    
    # Set initial conditions
    threshold_distance = calculate_threshold_distance(threshold_angle, predator_width)
    predator_pos[0] = [-threshold_distance, 0]
    predator_vel[0] = [predator_speed, 0]
    
    # Initialize prey positions
    for i in range(n_prey):
        prey_positions[0, i] = initial_prey_positions[i]
    
    # Initialize tracking variables
    predator_heading = 0
    response_idx = int(latency / time_step)
    escape_triggered = [False] * n_prey
    target_prey_idx = None
    
    # Simulation loop
    for i in range(1, n_steps):
        # Calculate current distances and angles for all prey
        for j in range(n_prey):
            distances[i-1, j] = np.sqrt(np.sum((prey_positions[i-1, j] - predator_pos[i-1])**2))
            los_angles[i-1, j] = calculate_los_angle(predator_pos[i-1], prey_positions[i-1, j])
            visual_angles[i-1, j] = calculate_visual_angle(distances[i-1, j], predator_width)
        
        # Choose target prey if haven't already
        if target_prey_idx is None:
            target_prey_idx = choose_target_prey(predator_pos[i-1], 
                                               prey_positions[i-1], 
                                               prey_velocities[i-1])
        
        # Break if predator catches target prey
        if distances[i-1, target_prey_idx] < min_distance:
            break
            
        # Update prey positions and velocities
        for j in range(n_prey):
            # Check if prey should start escaping
            if distances[i-1, j] <= threshold_distance and not escape_triggered[j]:
                escape_triggered[j] = True
                
            # Update prey velocity and position
            if escape_triggered[j] and i >= response_idx:
                prey_velocities[i, j] = [prey_speed * np.cos(escape_angles[j]),
                                       prey_speed * np.sin(escape_angles[j])]
            prey_positions[i, j] = prey_positions[i-1, j] + prey_velocities[i, j] * time_step
        
        # Update predator using proportional navigation towards target prey
        if i > 0:
            # Calculate line-of-sight rate for target prey
            los_rate = (los_angles[i-1, target_prey_idx] - los_angles[i-2, target_prey_idx]) / time_step
            
            # Update predator heading using PN law
            heading_rate = navigation_gain * los_rate
            predator_heading += heading_rate * time_step
            
            # Update predator velocity and position
            predator_vel[i] = [predator_speed * np.cos(predator_heading),
                             predator_speed * np.sin(predator_heading)]
            predator_pos[i] = predator_pos[i-1] + predator_vel[i] * time_step
    
    # Calculate final distances
    for j in range(n_prey):
        distances[-1, j] = np.sqrt(np.sum((prey_positions[-1, j] - predator_pos[-1])**2))
    
    # Trim arrays to valid simulation time
    valid_idx = max(1, np.where(distances[:, target_prey_idx] < min_distance)[0][0]) if np.any(distances[:, target_prey_idx] < min_distance) else n_steps
    
    return {
        'time': time[:valid_idx],
        'predator_pos': predator_pos[:valid_idx],
        'predator_vel': predator_vel[:valid_idx],
        'prey_positions': prey_positions[:valid_idx],
        'prey_velocities': prey_velocities[:valid_idx],
        'distances': distances[:valid_idx],
        'los_angles': los_angles[:valid_idx],
        'visual_angles': visual_angles[:valid_idx],
        'target_prey_idx': target_prey_idx,
        'min_distances': np.min(distances, axis=0),
        'capture': np.min(distances[:, target_prey_idx]) < min_distance
    }

def plot_pursuit(sim_results):
    """Plot the pursuit trajectories and analysis for single prey"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Plot trajectories
    predator_pos = sim_results['predator_pos']
    prey_pos = sim_results['prey_pos']
    
    ax1.plot(predator_pos[:, 0], predator_pos[:, 1], 'r-', label='Predator')
    ax1.plot(prey_pos[:, 0], prey_pos[:, 1], 'b-', label='Prey')
    
    # Plot position markers at regular intervals
    interval = len(sim_results['time']) // 10
    ax1.plot(predator_pos[::interval, 0], predator_pos[::interval, 1], 'ro', alpha=0.5)
    ax1.plot(prey_pos[::interval, 0], prey_pos[::interval, 1], 'bo', alpha=0.5)
    
    ax1.set_xlabel('X Position (cm)')
    ax1.set_ylabel('Y Position (cm)')
    ax1.set_title('Pursuit Trajectories')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot distance over time
    ax2.plot(sim_results['time'], sim_results['distances'], 'k-')
    ax2.axhline(y=2.0, color='r', linestyle='--', label='Capture threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (cm)')
    ax2.set_title('Predator-Prey Distance')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(sim_results['time'][0:-1], sim_results['visual_angles'][0:-1], '-', label='Visual Angle')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Visual Angle')
    ax3.set_title('Visual Angle Over Time')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    return fig


def plot_multi_prey_pursuit(sim_results):
    """Plot the pursuit trajectories and analysis for multiple prey"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(12, 4))
    
    # Plot trajectories
    predator_pos = sim_results['predator_pos']
    prey_positions = sim_results['prey_positions']
    target_idx = sim_results['target_prey_idx']
    
    # Plot predator trajectory
    ax1.plot(predator_pos[:, 0], predator_pos[:, 1], 'r-', label='Predator')
    
    # Plot prey trajectories
    colors = ['b', 'g', 'm', 'c', 'y']  # Colors for up to 5 prey
    labels = [f'Prey {i+1}' for i in range(prey_positions.shape[1])]
    
    for i in range(prey_positions.shape[1]):
        style = '-' if i == target_idx else '--'
        ax1.plot(prey_positions[:, i, 0], prey_positions[:, i, 1], 
                color=colors[i % len(colors)], linestyle=style, 
                label=f'{labels[i]} {"(Target)" if i == target_idx else ""}')
    
    # Plot position markers at regular intervals
    interval = len(sim_results['time']) // 10
    ax1.plot(predator_pos[::interval, 0], predator_pos[::interval, 1], 'ro', alpha=0.5)
    for i in range(prey_positions.shape[1]):
        ax1.plot(prey_positions[::interval, i, 0], prey_positions[::interval, i, 1], 
                f'{colors[i % len(colors)]}o', alpha=0.5)
    
    ax1.set_xlabel('X Position (cm)')
    ax1.set_ylabel('Y Position (cm)')
    ax1.set_title('Pursuit Trajectories')
    ax1.legend( loc='upper left')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot distances over time
    for i in range(prey_positions.shape[1]):
        style = '-' if i == target_idx else '--'
        ax2.plot(sim_results['time'], sim_results['distances'][:, i], 
                color=colors[i % len(colors)], linestyle=style, 
                label=f'{labels[i]} Distance')
    
    ax2.axhline(y=2.0, color='r', linestyle='--', label='Capture threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (cm)')
    ax2.set_title('Predator-Prey Distances')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Plot visual angles over time
    for i in range(prey_positions.shape[1]):
        style = '-' if i == target_idx else '--'
        ax3.plot(sim_results['time'][:-1], sim_results['visual_angles'][:-1, i],
                color=colors[i % len(colors)], linestyle=style, 
                label=f'{labels[i]} Visual Angle')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Visual Angle (degrees)')
    ax3.set_title('Visual Angles')
    ax3.grid(True)
    ax3.legend( loc='upper left')
    
    plt.tight_layout()
    return fig

def analyze_pursuit_success(predator_speeds, escape_angles, navigation_gains):
    """
    Analyze pursuit success for different parameters
    
    Args:
        predator_speeds (list): List of predator speeds to test
        escape_angles (list): List of prey escape angles to test
        navigation_gains (list): List of navigation gains to test
        
    Returns:
        list: List of dictionaries containing results for each parameter combination
    """
    results = []
    
    for speed in predator_speeds:
        for angle in escape_angles:
            for gain in navigation_gains:
                sim = simulate_pursuit(
                    predator_speed=speed,
                    escape_angle=angle,
                    navigation_gain=gain
                )
                
                results.append({
                    'speed': speed,
                    'angle': angle * 180 / np.pi,  # Convert to degrees
                    'gain': gain,
                    'min_distance': sim['min_distance'],
                    'capture': sim['capture']
                })
    
    return results

def plot_parameter_analysis(analysis_results):
    """
    Plot analysis of pursuit success across different parameters
    
    Args:
        analysis_results (list): Results from analyze_pursuit_success
    """
    # Convert results to numpy arrays for easier processing
    speeds = np.unique([r['speed'] for r in analysis_results])
    angles = np.unique([r['angle'] for r in analysis_results])
    gains = np.unique([r['gain'] for r in analysis_results])
    
    # Create success rate matrices
    speed_angle_success = np.zeros((len(speeds), len(angles)))
    speed_gain_success = np.zeros((len(speeds), len(gains)))
    angle_gain_success = np.zeros((len(angles), len(gains)))
    
    # Fill matrices
    for r in analysis_results:
        i_speed = np.where(speeds == r['speed'])[0][0]
        i_angle = np.where(angles == r['angle'])[0][0]
        i_gain = np.where(gains == r['gain'])[0][0]
        
        success = 1 if r['capture'] else 0
        speed_angle_success[i_speed, i_angle] += success
        speed_gain_success[i_speed, i_gain] += success
        angle_gain_success[i_angle, i_gain] += success
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Speed vs Angle
    im1 = ax1.imshow(speed_angle_success.T, origin='lower', aspect='auto',
                     extent=[speeds[0], speeds[-1], angles[0], angles[-1]])
    ax1.set_xlabel('Predator Speed (cm/s)')
    ax1.set_ylabel('Escape Angle (degrees)')
    ax1.set_title('Capture Success: Speed vs Angle')
    plt.colorbar(im1, ax=ax1)
    
    # Speed vs Gain
    im2 = ax2.imshow(speed_gain_success.T, origin='lower', aspect='auto',
                     extent=[speeds[0], speeds[-1], gains[0], gains[-1]])
    ax2.set_xlabel('Predator Speed (cm/s)')
    ax2.set_ylabel('Navigation Gain')
    ax2.set_title('Capture Success: Speed vs Gain')
    plt.colorbar(im2, ax=ax2)
    
    # Angle vs Gain
    im3 = ax3.imshow(angle_gain_success.T, origin='lower', aspect='auto',
                     extent=[angles[0], angles[-1], gains[0], gains[-1]])
    ax3.set_xlabel('Escape Angle (degrees)')
    ax3.set_ylabel('Navigation Gain')
    ax3.set_title('Capture Success: Angle vs Gain')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Single prey example
    # results_single = simulate_pursuit(
    #     predator_speed=10.0,
    #     escape_angle=np.pi/4,
    #     navigation_gain=3.0
    # )
    
    # fig_single = plot_pursuit(results_single)
    # plt.figure(fig_single.number)
    # plt.show()
    
    # Multiple prey example
    escape_angles = [np.pi/4, -np.pi/4]  # One prey escapes up, one down
    initial_positions = [[0, 0], [0, 5]]  # Different starting positions
    
    results_multi = simulate_multi_prey_pursuit(
        predator_speed=12.0,
        escape_angles=escape_angles,
        initial_prey_positions=initial_positions,
        navigation_gain=3.0
    )
    
    fig_multi = plot_multi_prey_pursuit(results_multi)
    plt.figure(fig_multi.number)
    plt.show()
    
    # # Parameter analysis example
    # analysis_results = analyze_pursuit_success(
    #     predator_speeds=[8, 10, 12, 14],
    #     escape_angles=[np.pi/6, np.pi/4, np.pi/3],
    #     navigation_gains=[2, 3, 4]
    # )
    
    # fig_analysis = plot_parameter_analysis(analysis_results)
    # plt.figure(fig_analysis.number)
    # plt.show()