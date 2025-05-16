import numpy as np
import random
from typing import Dict


def generate_hover(duration: float = None, samples: int = None) -> Dict:
    """Generate hover-at-fixed-point trajectory with random parameters"""
    # Randomize parameters if not provided
    if duration is None:
        duration = random.uniform(5.0, 15.0)  # Random duration between 5-15 seconds
    if samples is None:
        samples = random.randint(500, 1500)  # Random samples between 500-1500

    # Random height between 1.0-3.0 meters
    height = random.uniform(1.0, 3.0)
    state = [0.0, 0.0, 0.0, 0.0, height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {
        "type": "hover",
        "duration": duration,
        "samples": samples,
        "states": [state.copy() for _ in range(samples)]
    }

def generate_line(duration: float = None, samples: int = None) -> Dict:
    """Straight line at constant velocity with random parameters"""
    # Randomize parameters if not provided
    if duration is None:
        duration = random.uniform(5.0, 15.0)
    if samples is None:
        samples = random.randint(500, 1500)

    t = np.linspace(0, duration, samples)
    velocity = random.uniform(0.2, 1.0)  # Random velocity between 0.2-1.0 m/s
    x = velocity * t
    x_dot = np.full_like(t, velocity)

    states = []
    for i in range(samples):
        states.append([
            float(x[i]), float(x_dot[i]),
            0.0, 0.0,
            random.uniform(1.0, 3.0), 0.0,  # Random height
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0
        ])
    return {
        "type": "line",
        "duration": duration,
        "samples": samples,
        "states": states
    }

def generate_circle(radius: float = None, duration: float = None, samples: int = None) -> Dict:
    """Circular trajectory (1 full loop) with random parameters"""
    # Randomize parameters if not provided
    if radius is None:
        radius = random.uniform(1.0, 4.0)
    if duration is None:
        duration = random.uniform(5.0, 20.0)
    if samples is None:
        samples = random.randint(500, 1500)

    t = np.linspace(0, duration, samples)
    omega = 2 * np.pi / duration

    x = radius * np.sin(omega * t)
    y = radius * (1 - np.cos(omega * t))  # Centered at (0, radius)
    x_dot = radius * omega * np.cos(omega * t)
    y_dot = radius * omega * np.sin(omega * t)

    states = []
    for i in range(samples):
        states.append([
            float(x[i]), float(x_dot[i]),
            float(y[i]), float(y_dot[i]),
            random.uniform(1.0, 3.0), 0.0,  # Random height
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0
        ])
    return {
        "type": "circle",
        "duration": duration,
        "samples": samples,
        "states": states,
        "radius": radius  # Include radius in output for reference
    }

def generate_s_curve(duration: float = None, samples: int = None) -> Dict:
    """Smooth S-curve with polynomial planning and random parameters"""
    # Randomize parameters if not provided
    if duration is None:
        duration = random.uniform(5.0, 15.0)
    if samples is None:
        samples = random.randint(500, 1500)

    t = np.linspace(0, duration, samples)
    # 5th-order polynomial (smooth start/stop)
    tau = t / duration
    max_distance = random.uniform(2.0, 5.0)  # Random max distance
    x = max_distance * (6*tau**5 - 15*tau**4 + 10*tau**3)
    x_dot = max_distance * (30*tau**4 - 60*tau**3 + 30*tau**2) / duration

    states = []
    for i in range(samples):
        states.append([
            float(x[i]), float(x_dot[i]),
            0.0, 0.0,
            random.uniform(1.0, 3.0), 0.0,  # Random height
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0
        ])
    return {
        "type": "s_curve",
        "duration": duration,
        "samples": samples,
        "states": states,
        "max_distance": max_distance  # Include max distance in output
    }

def generate_step(duration: float = None, samples: int = None) -> Dict:
    """Discrete step change at midpoint with random parameters"""
    # Randomize parameters if not provided
    if duration is None:
        duration = random.uniform(5.0, 15.0)
    if samples is None:
        samples = random.randint(500, 1500)

    t = np.linspace(0, duration, samples)
    step_size = random.uniform(1.0, 3.0)  # Random step size
    x = np.where(t < duration/2, 0.0, step_size)

    states = []
    for i in range(samples):
        states.append([
            float(x[i]), 0.0,
            0.0, 0.0,
            random.uniform(1.0, 3.0), 0.0,  # Random height
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0
        ])
    return {
        "type": "step",
        "duration": duration,
        "samples": samples,
        "states": states,
        "step_size": step_size  # Include step size in output
    }
