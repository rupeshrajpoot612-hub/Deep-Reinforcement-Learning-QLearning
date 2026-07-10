"""
q_learning_agent.py
--------------------
Trains a tabular Q-learning agent on the FrozenLake-v1 environment.

Q-learning update rule:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

where:
    alpha  = learning rate
    gamma  = discount factor
    r      = reward received
    s, a   = current state, action
    s'     = next state

The agent follows an epsilon-greedy policy during training: with
probability epsilon it explores (random action), otherwise it
exploits (picks the best known action from the Q-table). Epsilon is
decayed after every episode so the agent explores heavily at first and
increasingly exploits its learned knowledge over time.

Workflow implemented here:

    Agent -> Environment -> Reward -> Update Q-Table -> Learn Policy

Running this script trains the agent, saves the learned Q-table,
plots training curves (rewards + success rate), and saves a training
report to outputs/.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment_setup import create_environment, initialize_q_table, load_environment_config

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_EPISODES = 15000
MAX_STEPS_PER_EPISODE = 100

LEARNING_RATE = 0.1        # alpha
DISCOUNT_FACTOR = 0.99     # gamma

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_RATE = 0.0005

RANDOM_SEED = 42


def choose_action(q_table: np.ndarray, state: int, epsilon: float, env) -> int:
    """Epsilon-greedy action selection."""
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # explore
    return int(np.argmax(q_table[state, :]))  # exploit


def train_agent(
    env,
    q_table: np.ndarray,
    num_episodes: int = NUM_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    alpha: float = LEARNING_RATE,
    gamma: float = DISCOUNT_FACTOR,
    epsilon_start: float = EPSILON_START,
    epsilon_min: float = EPSILON_MIN,
    epsilon_decay_rate: float = EPSILON_DECAY_RATE,
    seed: int = RANDOM_SEED,
):
    """
    Trains the Q-table via the Q-learning algorithm.

    Returns
    -------
    q_table : the learned Q-table
    rewards_per_episode : list of total reward obtained each episode (0 or 1 for FrozenLake)
    epsilon_history : epsilon value at each episode (for diagnostics)
    """
    np.random.seed(seed)
    rewards_per_episode = []
    epsilon_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0

        for _ in range(max_steps):
            action = choose_action(q_table, state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update rule
            best_next_action_value = np.max(q_table[next_state, :])
            td_target = reward + gamma * best_next_action_value * (not terminated)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon (more exploitation as training progresses)
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(
            -epsilon_decay_rate * episode
        )

        rewards_per_episode.append(total_reward)
        epsilon_history.append(epsilon)

    return q_table, rewards_per_episode, epsilon_history


def compute_success_rate(rewards_per_episode, window: int = 100):
    """Rolling success rate (fraction of episodes reaching the goal) over a window."""
    rewards = np.array(rewards_per_episode)
    success_rate = np.convolve(rewards, np.ones(window) / window, mode="valid")
    return success_rate


def plot_rewards(rewards_per_episode, save_path, window: int = 100):
    rewards = np.array(rewards_per_episode)
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(9, 5))
    plt.plot(rolling_avg, color="#4C72B0")
    plt.xlabel(f"Episode (rolling window = {window})")
    plt.ylabel("Average Reward")
    plt.title("Training Reward Over Time — Q-Learning on FrozenLake-v1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_success_rate(rewards_per_episode, save_path, window: int = 100):
    success_rate = compute_success_rate(rewards_per_episode, window) * 100

    plt.figure(figsize=(9, 5))
    plt.plot(success_rate, color="#55A868")
    plt.xlabel(f"Episode (rolling window = {window})")
    plt.ylabel("Success Rate (%)")
    plt.title("Agent Success Rate Over Training — FrozenLake-v1")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_q_table_heatmap(q_table: np.ndarray, save_path, action_labels=None):
    if action_labels is None:
        action_labels = ["LEFT", "DOWN", "RIGHT", "UP"]

    plt.figure(figsize=(6, 7))
    im = plt.imshow(q_table, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Q-value")
    plt.xticks(range(len(action_labels)), action_labels)
    plt.yticks(range(q_table.shape[0]), [f"S{s}" for s in range(q_table.shape[0])])
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.title("Learned Q-Table Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    config = load_environment_config()
    env = create_environment(map_name=config["map_name"], is_slippery=config["is_slippery"])
    q_table = initialize_q_table(env)

    print(f"Training Q-learning agent on FrozenLake-v1 "
          f"(map={config['map_name']}, slippery={config['is_slippery']})")
    print(f"Episodes: {NUM_EPISODES}, alpha={LEARNING_RATE}, gamma={DISCOUNT_FACTOR}\n")

    q_table, rewards_per_episode, epsilon_history = train_agent(env, q_table)

    total_reward = sum(rewards_per_episode)
    overall_success_rate = total_reward / len(rewards_per_episode)
    last_1000_success_rate = sum(rewards_per_episode[-1000:]) / 1000

    print(f"Total reward across all episodes: {total_reward:.0f} / {NUM_EPISODES}")
    print(f"Overall success rate: {overall_success_rate * 100:.2f}%")
    print(f"Success rate (last 1000 episodes): {last_1000_success_rate * 100:.2f}%")

    # Plots
    plot_rewards(rewards_per_episode, os.path.join(SCREENSHOTS_DIR, "rewards_per_episode.png"))
    plot_success_rate(rewards_per_episode, os.path.join(SCREENSHOTS_DIR, "success_rate.png"))
    plot_q_table_heatmap(q_table, os.path.join(SCREENSHOTS_DIR, "q_table_heatmap.png"))

    # Save the learned Q-table + training metadata
    artifact = {
        "q_table": q_table,
        "hyperparameters": {
            "num_episodes": NUM_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "alpha": LEARNING_RATE,
            "gamma": DISCOUNT_FACTOR,
            "epsilon_start": EPSILON_START,
            "epsilon_min": EPSILON_MIN,
            "epsilon_decay_rate": EPSILON_DECAY_RATE,
        },
        "environment_config": config,
        "total_reward": total_reward,
        "overall_success_rate": overall_success_rate,
        "last_1000_success_rate": last_1000_success_rate,
    }
    q_table_path = os.path.join(OUTPUTS_DIR, "q_table.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved learned Q-table to {q_table_path}")

    env.close()
    return artifact, rewards_per_episode


if __name__ == "__main__":
    main()
