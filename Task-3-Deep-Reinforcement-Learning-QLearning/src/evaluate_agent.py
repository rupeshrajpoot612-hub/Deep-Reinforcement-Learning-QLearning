"""
evaluate_agent.py
------------------
Loads a trained Q-table (outputs/q_table.pkl) and evaluates the greedy
policy derived from it — i.e. the agent always picks
argmax(Q[state, :]) with no exploration — over a number of test
episodes on FrozenLake-v1.

Also prints the learned policy as a human-readable arrow grid, and
writes a final text report to outputs/evaluation_report.txt.

Usage:
    python evaluate_agent.py
"""

import os
import pickle

import numpy as np

from environment_setup import create_environment, load_environment_config

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
Q_TABLE_PATH = os.path.join(OUTPUTS_DIR, "q_table.pkl")

ACTION_ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}  # LEFT, DOWN, RIGHT, UP


def load_q_table(path: str = Q_TABLE_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_policy(env, q_table: np.ndarray, num_episodes: int = 1000, max_steps: int = 100,
                     seed: int = 123):
    """Run the greedy (no-exploration) policy and measure its success rate."""
    successes = 0
    total_rewards = 0
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        steps = 0

        for _ in range(max_steps):
            action = int(np.argmax(q_table[state, :]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            total_rewards += reward

            if done:
                if reward > 0:
                    successes += 1
                break

        episode_lengths.append(steps)

    success_rate = successes / num_episodes
    avg_episode_length = float(np.mean(episode_lengths))

    return {
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "total_reward": total_rewards,
        "avg_episode_length": avg_episode_length,
    }


def render_policy_grid(q_table: np.ndarray, map_name: str = "4x4") -> str:
    """Render the greedy policy as a grid of arrows (assumes a square map)."""
    grid_size = int(np.sqrt(q_table.shape[0]))
    best_actions = np.argmax(q_table, axis=1)

    lines = []
    for row in range(grid_size):
        row_symbols = []
        for col in range(grid_size):
            state = row * grid_size + col
            row_symbols.append(ACTION_ARROWS[best_actions[state]])
        lines.append(" ".join(row_symbols))
    return "\n".join(lines)


def main():
    artifact = load_q_table()
    q_table = artifact["q_table"]
    config = artifact["environment_config"]

    print("Loaded Q-table with shape:", q_table.shape)
    print(f"Training overall success rate: {artifact['overall_success_rate'] * 100:.2f}%")
    print(f"Training success rate (last 1000 episodes): "
          f"{artifact['last_1000_success_rate'] * 100:.2f}%\n")

    env = create_environment(map_name=config["map_name"], is_slippery=config["is_slippery"])

    print("Evaluating greedy policy (no exploration) over 1000 test episodes...")
    eval_results = evaluate_policy(env, q_table, num_episodes=1000)

    print(f"\nEvaluation success rate: {eval_results['success_rate'] * 100:.2f}%")
    print(f"Total reward over {eval_results['num_episodes']} episodes: "
          f"{eval_results['total_reward']:.0f}")
    print(f"Average episode length: {eval_results['avg_episode_length']:.2f} steps")

    policy_grid = render_policy_grid(q_table, config["map_name"])
    print("\nLearned policy (arrows show the best action per state):")
    print(policy_grid)

    env.close()

    # Write evaluation report
    report_lines = [
        "Q-Learning Agent — Evaluation Report",
        "=" * 40,
        "",
        f"Environment: FrozenLake-v1 (map={config['map_name']}, "
        f"slippery={config['is_slippery']})",
        "",
        "Training summary:",
        f"  Total reward (training): {artifact['total_reward']:.0f}",
        f"  Overall success rate (training): {artifact['overall_success_rate'] * 100:.2f}%",
        f"  Success rate, last 1000 episodes (training): "
        f"{artifact['last_1000_success_rate'] * 100:.2f}%",
        "",
        "Evaluation summary (greedy policy, no exploration):",
        f"  Episodes evaluated: {eval_results['num_episodes']}",
        f"  Successes: {eval_results['successes']}",
        f"  Success rate: {eval_results['success_rate'] * 100:.2f}%",
        f"  Total reward: {eval_results['total_reward']:.0f}",
        f"  Average episode length: {eval_results['avg_episode_length']:.2f} steps",
        "",
        "Learned policy (arrows = best action per state):",
        policy_grid,
    ]
    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nEvaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
