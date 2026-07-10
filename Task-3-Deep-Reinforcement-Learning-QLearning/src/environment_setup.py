"""
environment_setup.py
---------------------
Creates and configures the Gymnasium FrozenLake-v1 environment, and
initializes the Q-table used by the Q-learning agent.

FrozenLake-v1 is a grid-world environment:
    S F F F
    F H F H
    F F F H
    H F F G

  - S = Start
  - F = Frozen (safe) tile
  - H = Hole (falling in ends the episode with reward 0)
  - G = Goal (reaching it ends the episode with reward 1)

The agent has 4 discrete actions (LEFT, DOWN, RIGHT, UP) and 16
discrete states (one per grid cell, for the default 4x4 map). By
default `is_slippery=True`, meaning actions succeed as intended only
some of the time (stochastic transitions) — this is what makes the
environment a genuine reinforcement learning problem rather than a
simple deterministic maze.
"""

import json
import os

import gymnasium as gym
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "environment_config.json")


def load_environment_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def create_environment(map_name: str = "4x4", is_slippery: bool = True, render_mode=None):
    """Create and return a fresh FrozenLake-v1 environment."""
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )
    return env


def initialize_q_table(env) -> np.ndarray:
    """Initialize a Q-table of zeros with shape (num_states, num_actions)."""
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return np.zeros((num_states, num_actions))


if __name__ == "__main__":
    config = load_environment_config()
    print("Environment config:")
    print(json.dumps(config, indent=2))

    env = create_environment(
        map_name=config["map_name"], is_slippery=config["is_slippery"]
    )
    q_table = initialize_q_table(env)

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Q-table shape: {q_table.shape}")
    env.close()
