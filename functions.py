import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
import random

def angle_diff(ang1: float, ang2: float) -> float:
    """Calculate the smallest difference between two angles."""
    dA = ang1 - ang2
    while dA < -np.pi:
        dA += 2 * np.pi
    while dA > np.pi:
        dA -= 2 * np.pi
    return dA

def get_dBA(prey_x: float, prey_y: float, prey_dx: float, prey_dy: float, 
            fish_x: float, fish_y: float, fish_dx: float, fish_dy: float) -> Dict[str, float]:
    """Calculate bearing angle and its derivative."""
    x_diff = fish_x - prey_x
    y_diff = prey_y - fish_y
    dx_diff = prey_dx - fish_dx
    dy_diff = prey_dy - fish_dy
    dBA = (x_diff * dy_diff - y_diff * dx_diff)/(x_diff**2 + y_diff**2)
    BA = -np.arctan2(y_diff, x_diff)
    return {"BA": -BA, "DBA": -dBA}

def get_dBA_exp(prey_x: float, prey_y: float, prey_dx: float, prey_dy: float,
                fish_x: float, fish_y: float, fish_dx: float, fish_dy: float) -> Dict[str, float]:
    """Calculate experimental bearing angle and its derivative."""
    x_diff = fish_x - prey_x
    y_diff = fish_y - prey_y
    dx_diff = fish_dx - prey_dx
    dy_diff = fish_dy - prey_dy
    dBA = (y_diff * dx_diff - x_diff * dy_diff)/(x_diff**2 + y_diff**2)
    BA = -np.arctan2(y_diff, x_diff)
    return {"BA": BA, "DBA": dBA}

def get_pred_BA(r_t: np.ndarray, v_t: np.ndarray, r_p: np.ndarray, v_p: np.ndarray, 
                t: int, lag_steps: int, dt: float, f_time: float) -> float:
    """Calculate predicted bearing angle."""
    pred_prey_x = r_t[t - lag_steps, 0] + v_t[t - lag_steps, 0] * f_time
    pred_prey_y = r_t[t - lag_steps, 1] + v_t[t - lag_steps, 1] * f_time
    pred_fish_x = r_p[t - 1, 0]
    pred_fish_y = r_p[t - 1, 1]
    
    x_diff = pred_fish_x - pred_prey_x
    y_diff = pred_fish_y - pred_prey_y
    
    pred_BA = -np.arctan2(y_diff, x_diff)
    return pred_BA

def interpolate_timeseries(in_data: Union[np.ndarray, pd.DataFrame], factor: int) -> np.ndarray:
    """Interpolate time series data by a given factor."""
    if isinstance(in_data, pd.DataFrame):
        in_data = in_data.values
        
    n_rows = in_data.shape[0]
    n_cols = in_data.shape[1] if len(in_data.shape) > 1 else 1
    
    x = np.arange(n_rows)
    x_new = np.linspace(0, n_rows-1, 1 + factor*(n_rows-1))
    
    if n_cols == 1:
        f = interp1d(x, in_data, kind='linear')
        return f(x_new)
    
    out_data = np.zeros((len(x_new), n_cols))
    for i in range(n_cols):
        f = interp1d(x, in_data[:, i], kind='linear')
        out_data[:, i] = f(x_new)
    
    return out_data

def run_sim(model: str, lag: float, dt: float, s_p: float, s_t: float, 
            maneuver_time: float, maneuver_angle: float, k: float, 
            tMax: float, f_time: float) -> Tuple[np.ndarray, ...]:
    """
    Run predator-prey pursuit simulation.
    
    Args:
        model: Type of prediction model ("PN", "pPN", or "pPP")
        lag: Sensory-motor delay
        dt: Time step
        s_p: Predator speed
        s_t: Target speed
        maneuver_time: Time of target maneuver
        maneuver_angle: Angle of target maneuver
        k: Control gain
        tMax: Maximum simulation time
        f_time: Forecast time
    """
    lag_steps = int(lag / dt)
    steps = int(tMax / dt)
    
    # Initialize arrays
    r_t = np.zeros((steps, 2))  # Target position
    r_p = np.zeros((steps, 2))  # Predator position
    v_t = s_t*np.ones((steps, 2))  # Target velocity
    v_p = np.zeros((steps, 2))  # Predator velocity
    
    # Set initial conditions
    r_p[0:lag_steps+2, 0] = 60
    #v_t[:, 1] = 0
    v_p[0:lag_steps+2, 0] = -s_p
    h = np.zeros(steps)  # Heading angle
    
    sensory_update_rate = 60  # Hz
    sense_frame = random.randint(0, int(1/(sensory_update_rate*dt)))
    update_rate_frames = round(1/(sensory_update_rate*dt))
    
    t = lag_steps + 1
    r = np.sqrt(np.sum((r_t[t] - r_p[t])**2))
    min_r = r
    dh = 0
    man_frame = None
    
    while True:
        if t % update_rate_frames == sense_frame:
            if model == "PN":
                LOS = get_dBA_exp(r_t[t-lag_steps, 0], r_t[t-lag_steps, 1],
                                v_t[t-lag_steps, 0], v_t[t-lag_steps, 1],
                                r_p[t-lag_steps, 0], r_p[t-lag_steps, 1],
                                v_p[t-lag_steps, 0], v_p[t-lag_steps, 1])
                dh = k * LOS["DBA"]
                
            elif model == "pPN":
                # Predictive proportional navigation
                LOS = get_dBA_exp(r_t[t-lag_steps, 0], r_t[t-lag_steps, 1],
                                v_t[t-lag_steps, 0], v_t[t-lag_steps, 1],
                                r_p[t-lag_steps, 0], r_p[t-lag_steps, 1],
                                v_p[t-lag_steps, 0], v_p[t-lag_steps, 1])
                LOS_min1 = get_dBA_exp(r_t[t-lag_steps-1, 0], r_t[t-lag_steps-1, 1],
                                     v_t[t-lag_steps-1, 0], v_t[t-lag_steps-1, 1],
                                     r_p[t-lag_steps-1, 0], r_p[t-lag_steps-1, 1],
                                     v_p[t-lag_steps-1, 0], v_p[t-lag_steps-1, 1])
                dBA2 = (LOS["DBA"] - LOS_min1["DBA"]) / dt
                pred_dBA = LOS["DBA"] + dBA2 * f_time
                dh = k * pred_dBA
                
            elif model == "pPP":
                # Predictive pursuit
                pred_BA = get_pred_BA(r_t, v_t, r_p, v_p, t, lag_steps, dt, f_time)
                dh = k * angle_diff(pred_BA, h[t-1])
        
        h[t] = h[t-1] + dh * dt
        v_p[t] = s_p * np.array([-np.cos(h[t]), np.sin(h[t])])
        r_p[t] = r_p[t-1] + v_p[t] * dt
        
        if (r / s_p) < maneuver_time and man_frame is None:
            man_frame = t
            print(r, r/s_p, "Maneuver!")
            v_t[t] = s_t * np.array([np.cos(maneuver_angle), np.sin(maneuver_angle)])
        else:
            v_t[t] = v_t[t-1]
            print(r, r/s_p)
        r_t[t] = r_t[t-1] + v_t[t] * dt
        
        r = np.sqrt(np.sum((r_t[t] - r_p[t])**2))
        min_r = min(r, min_r)
        
        t += 1
        if t >= steps-1 or r < 1:
            break
    
    return r_t[:t], v_t[:t], r_p[:t], v_p[:t], min_r, h[:t], t

# Optional: Summary statistics function (similar to R's summarySE)
def summary_statistics(data: pd.DataFrame, measure_var: str, 
                      group_vars: Optional[List[str]] = None,
                      conf_interval: float = 0.95) -> pd.DataFrame:
    """
    Calculate summary statistics including mean, SD, SE, and confidence intervals.
    
    Args:
        data: Input DataFrame
        measure_var: Variable to summarize
        group_vars: Variables to group by
        conf_interval: Confidence interval level
    """
    if group_vars is None:
        group_vars = []
    
    def summarize(group):
        n = len(group)
        mean = group.mean()
        sd = group.std(ddof=1)  # Sample standard deviation
        se = sd / np.sqrt(n)
        ci = se * stats.t.ppf((1 + conf_interval) / 2, n-1)
        return pd.Series({'N': n, 'mean': mean, 'sd': sd, 'se': se, 'ci': ci})
    
    if not group_vars:
        summary = summarize(data[measure_var])
        return pd.DataFrame(summary).T
    
    grouped = data.groupby(group_vars)[measure_var].apply(lambda x: summarize(x)).unstack()
    return grouped.reset_index()