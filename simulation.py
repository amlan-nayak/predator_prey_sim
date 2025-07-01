#!/usr/bin/env python3
"""
sim.py
──────────────────────────────────────────────────────────────────────────────
Self-contained functional simulation of a 2-D predator–prey chase that measures
*miss distance* (minimum separation) as in Martin et al. (Oikos).

Predator guidance algorithms included
-------------------------------------
1. pure_pursuit(pos, target, speed, vel, inertia=0.0)
2. proportional_navigation(pos, target, speed, vel, dt, gain, prev_los)
3. predictive_pursuit(pos, target, tgt_vel, vel, speed, delay_fr, inertia=0.25)

Prey behaviour
--------------
* Burst-and-coast locomotion.
* Escape trigger: looming-rate threshold λ̇.
* Turn directly away before bursting.

Performance metric
------------------
*Minimum predator–prey distance* reached in a trial.
Simulation ends once the predator is moving away again by a factor ε of that
minimum (mirrors the Oikos termination rule).

Outputs
-------
* CSV:  median_miss_results.csv
* Plot: Escape threshold vs median miss distance
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from multiprocessing import Pool

# ─────────────────── simulation constants ────────────────────
DT                = 0.02          # s   – integrator time-step
MAX_TIME          = 10.0          # s   – hard cap per trial
ARENA_SIZE        = 500.0         # px  – toroidal arena (wrap-around)
CAPTURE_EPSILON   = 10.0          # ε   – end trial when dist > ε·min_dist

# Predator parameters
PRED_SPEED        = 4.0           # px·step⁻¹
PN_GAIN           = 1.0           # N  – proportional-navigation gain
SENSOR_DELAY_FR   = 2             # frames for predictive pursuit (≈ DT*2)

# Prey parameters
PREY_COAST_SPEED  = 3.0           # px·step⁻¹
PREY_BURST_SPEED  = 5.0           # px·step⁻¹
BURST_DURATION    = 0.5          # s   – burst phase length
# ──────────────────────────────────────────────────────────────

# ===== helper maths ========================================================
def _norm(v: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
    """Euclidean norm with safe zero-division handling."""
    return np.linalg.norm(v, axis=axis, keepdims=keepdims)


def _unit(v: np.ndarray) -> np.ndarray:
    """Return v normalised; if ‖v‖≈0 return zeros."""
    n = _norm(v, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n
# ===========================================================================

# ===== predator guidance algorithms =======================================
def pure_pursuit(pos: np.ndarray, target: np.ndarray,
                 speed: float, vel: np.ndarray,
                 inertia: float = 0.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Point heading directly at instantaneous target position.
    Optional first-order lag via `inertia` (0 = none).
    """
    direction = target - pos
    if _norm(direction) < 1e-12:
        return pos, vel
    desired_vel = _unit(direction) * speed
    vel = inertia*vel + (1.0 - inertia)*desired_vel
    pos += vel
    pos[:] = np.mod(pos, ARENA_SIZE)
    return pos, vel


def proportional_navigation(pos: np.ndarray, target: np.ndarray,
                            speed: float, vel: np.ndarray,
                            dt: float, gain: float,
                            prev_los: float | None
                            ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Guidance law: turn rate ω = N · λ̇,  where λ̇ is LOS rate.
    `prev_los` carries LOS angle from previous frame; pass None on first call.
    """
    rel = target - pos
    if _norm(rel) < 1e-12:
        return pos, vel, prev_los
    los = np.arctan2(rel[1], rel[0])

    # LOS rate
    if prev_los is None:
        lam_dot = 0.0
    else:
        dθ = los - prev_los
        lam_dot = np.arctan2(np.sin(dθ), np.cos(dθ)) / dt

    # Current heading
    if _norm(vel) < 1e-6:
        heading = los
    else:
        heading = np.arctan2(vel[1], vel[0])

    # Apply turn
    new_heading = heading + gain * lam_dot * dt
    vel = speed * np.array([np.cos(new_heading), np.sin(new_heading)])
    pos += vel
    pos[:] = np.mod(pos, ARENA_SIZE)
    return pos, vel, los


def predictive_pursuit(pos: np.ndarray, target: np.ndarray,
                       tgt_vel: np.ndarray, vel: np.ndarray,
                       speed: float, delay_fr: int,
                       inertia: float = 0.25
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple target-motion-prediction pursuit (a.k.a. constant-heading pursuit):
    • Predict where both predator & prey will be after `delay_fr` frames.
    • Steer toward that predicted position.
    • `inertia` (0-1) gives first-order heading lag.
    """
    future_target = target + tgt_vel * delay_fr
    future_self   = pos    + vel      * delay_fr
    desired_vel   = _unit(future_target - future_self) * speed
    vel = inertia*vel + (1.0 - inertia)*desired_vel
    pos += vel
    pos[:] = np.mod(pos, ARENA_SIZE)
    return pos, vel
# ===========================================================================

# ===== prey behaviour ======================================================
def update_prey(prey_pos: np.ndarray, prey_vel: np.ndarray,
                predator_pos: np.ndarray,
                looming_thr: float,
                state: Dict
                ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Burst-and-coast prey with looming-rate escape trigger.

    `state` keys
    ------------
    phase       : 'coast' | 'burst'
    burst_left  : frames remaining in current burst
    prev_dist   : distance to predator at t-dt (for looming derivative)
    """
    dist = _norm(predator_pos - prey_pos)
    if state["prev_dist"] is None:
        state["prev_dist"] = dist
    lam_dot = (state["prev_dist"] - dist) / (dist*DT + 1e-9)  # looming rate
    state["prev_dist"] = dist

    # Trigger burst
    if state["phase"] == "coast" and lam_dot > looming_thr:
        state["phase"]      = "burst"
        state["burst_left"] = int(BURST_DURATION / DT)
        # Instantaneous turn directly away before sprint
        prey_vel[:] = _unit(prey_pos - predator_pos) * PREY_BURST_SPEED

    # Propagate motion
    if state["phase"] == "burst":
        prey_pos += prey_vel
        state["burst_left"] -= 1
        if state["burst_left"] <= 0:        # back to coast
            state["phase"] = "coast"
            prey_vel[:]    = _unit(prey_vel) * PREY_COAST_SPEED
    else:
        prey_pos += prey_vel

    prey_pos[:] = np.mod(prey_pos, ARENA_SIZE)
    return prey_pos, prey_vel, state
# ===========================================================================

# ===== one trial ===========================================================
def run_trial(looming_thr: float,
              guidance_mode: str = "predictive"
              ) -> float:
    """
    Execute a single simulation and return the *minimum* predator–prey
    distance (miss distance).
    """
    # Initial positions
    prey_pos = np.array([ARENA_SIZE*0.5, ARENA_SIZE*0.5])
    predator_pos = np.array([ARENA_SIZE*0.7,
                             np.random.uniform(0, ARENA_SIZE)])

    # Initial velocities
    prey_vel = _unit(np.random.randn(2)) * PREY_COAST_SPEED
    pred_vel = _unit(prey_pos - predator_pos) * PRED_SPEED

    # Auxiliary state
    prey_state = {"phase": "coast", "burst_left": 0, "prev_dist": None}
    prev_los   : float | None = None  # for PN

    min_dist = _norm(predator_pos - prey_pos)  # record closest approach
    t = 0.0
    while t < MAX_TIME:
        # ---- prey step ----------------------------------------------------
        prey_pos, prey_vel, prey_state = update_prey(prey_pos, prey_vel,
                                                     predator_pos,
                                                     looming_thr,
                                                     prey_state)
        # ---- predator step ------------------------------------------------
        if guidance_mode == "pure":
            predator_pos, pred_vel = pure_pursuit(predator_pos, prey_pos,
                                                  PRED_SPEED, pred_vel)
        elif guidance_mode == "pn":
            predator_pos, pred_vel, prev_los = proportional_navigation(
                predator_pos, prey_pos, PRED_SPEED, pred_vel,
                DT, PN_GAIN, prev_los)
        else:  # 'predictive' (default)
            predator_pos, pred_vel = predictive_pursuit(
                predator_pos, prey_pos, prey_vel, pred_vel,
                PRED_SPEED, SENSOR_DELAY_FR)

        # ---- metric update & termination check ---------------------------
        dist = _norm(predator_pos - prey_pos)
        min_dist = min(min_dist, dist)
        if dist > CAPTURE_EPSILON * min_dist:
            break
        t += DT

    return float(min_dist)
# ===========================================================================

# ===== parallel sweep helper ===============================================
def _process_threshold(args: Tuple[float, int, str]) -> Tuple[float, float, float]:
    thr, trials_per, guidance_mode = args
    dists = [run_trial(thr, guidance_mode) for _ in range(trials_per)]
    return thr, float(np.median(dists)), float(np.mean(dists))


def sweep_thresholds_parallel(thresholds: np.ndarray,
                              trials_per: int = 300,
                              guidance_mode: str = "predictive",
                              num_workers: int = 6
                              ) -> pd.DataFrame:
    """
    Parallelized parameter sweep using multiprocessing Pool.
    """
    with Pool(processes=num_workers) as pool:
        args = [(thr, trials_per, guidance_mode) for thr in thresholds]
        results = pool.map(_process_threshold, args)
    thr_vals, medians, means = zip(*results)
    return pd.DataFrame({
        "looming_threshold": thr_vals,
        "median_miss_distance": medians,
        "mean_miss_distance": means
    })
# ===========================================================================

# ===== main / demo =========================================================
if __name__ == "__main__":
    np.random.seed(2025)                           # reproducibility

    # Parameter sweep grid (s⁻¹)
    thr_values = np.linspace(0.5, 5.0, 100)

    # Choose guidance mode: 'predictive' | 'pn' | 'pure'
    mode = "pure"

    print(f"Running simulations with {mode} pursuit on 6 cores …")
    results_df = sweep_thresholds_parallel(
        thr_values,
        trials_per=100,
        guidance_mode=mode,
        num_workers=6
    )

    # --- summary printout --------------------------------------------------
    print("\nMedian miss distance (px) per threshold")
    print(results_df[["looming_threshold", "median_miss_distance"]])

    # --- CSV output --------------------------------------------------------
    csv_path = f"{mode}_median_miss_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved ➜ {csv_path}")

    # --- plot --------------------------------------------------------------
    plt.figure()
    plt.plot(results_df["looming_threshold"],
             results_df["median_miss_distance"],
             marker="o")
    plt.xlabel("Looming-rate threshold λ̇ (s⁻¹)")
    plt.ylabel("Median miss distance (px)")
    plt.title(f"Escape threshold vs median miss distance ({mode} pursuit)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

