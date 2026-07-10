# Task 3: Deep Reinforcement Learning (Q-Learning)

## Intern Details

- Intern ID: **[CITS2473]**
- Full Name: **[Rupesh Singh]**
- Duration: **[8 Weeks]**

An agent trained with **tabular Q-learning** to learn the optimal
policy for navigating the **FrozenLake-v1** environment from
[Gymnasium](https://gymnasium.farama.org/).

## 🎯 Objective

Train an agent to learn optimal actions through trial-and-error
interaction with an environment, using the Q-learning algorithm.

## 📁 Project Structure

```
Task-3-Deep-Reinforcement-Learning-QLearning/
│
├── data/
│   └── environment_config.json     # FrozenLake-v1 configuration (map, slipperiness, etc.)
│                                    # No external dataset is needed — Gymnasium generates
│                                    # the environment itself; this file documents the exact
│                                    # setup used so results are reproducible.
│
├── notebooks/
│   └── qlearning_analysis.ipynb    # Full training + evaluation walkthrough (pre-run)
│
├── src/
│   ├── environment_setup.py        # Creates the FrozenLake-v1 env + initializes Q-table
│   ├── q_learning_agent.py         # Epsilon-greedy Q-learning training loop + plots
│   └── evaluate_agent.py           # Evaluates the learned (greedy) policy
│
├── screenshots/
│   ├── rewards_per_episode.png     # Rolling average reward during training
│   ├── success_rate.png            # Rolling success rate during training
│   └── q_table_heatmap.png         # Heatmap of the learned Q-table
│
├── outputs/
│   ├── q_table.pkl                 # Learned Q-table + hyperparameters + training stats
│   └── evaluation_report.txt       # Evaluation results + learned policy grid
│
├── documentation/
│   └── project_report.pdf          # Written project report
│
├── requirements.txt
└── README.md
```

## 🌎 Environment: FrozenLake-v1

FrozenLake-v1 is a 4x4 grid-world:

```
S F F F
F H F H
F F F H
H F F G
```

- **S** = Start tile
- **F** = Frozen (safe) tile
- **H** = Hole — falling in ends the episode with reward 0
- **G** = Goal — reaching it ends the episode with reward 1

The agent has 4 discrete actions (`LEFT`, `DOWN`, `RIGHT`, `UP`) and 16
discrete states. By default `is_slippery=True`, meaning the agent's
chosen action only succeeds part of the time — the ice is slippery, so
the agent sometimes slides in a different direction than intended.
This stochasticity is what makes the environment a genuine
reinforcement learning problem.

## 🔄 Workflow

```
Agent
  ↓
Environment
  ↓
Reward
  ↓
Update Q-Table
  ↓
Learn Policy
```

## 🧠 Algorithm: Q-Learning

The agent maintains a Q-table `Q[state, action]` estimating the
expected future reward of taking each action in each state. After
every step it updates the table using the Bellman equation:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

During training the agent follows an **epsilon-greedy** policy:
starting fully random (`epsilon = 1.0`) and decaying towards mostly
greedy (`epsilon_min = 0.01`) as training progresses, so it explores
early and increasingly exploits what it has learned.

| Hyperparameter | Value |
|---|---|
| Episodes | 15,000 |
| Max steps per episode | 100 |
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.99 |
| Epsilon start / min | 1.0 / 0.01 |
| Epsilon decay rate | 0.0005 |

## ⚙️ Setup & Installation

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ How to Run

### Option A — Run the scripts directly
```bash
cd src
python environment_setup.py    # sanity-check the environment + Q-table shape
python q_learning_agent.py     # trains the agent, saves plots + q_table.pkl
python evaluate_agent.py       # evaluates the learned greedy policy
```

### Option B — Explore the notebook
```bash
jupyter notebook notebooks/qlearning_analysis.ipynb
```
The notebook is already pre-executed with saved outputs.

## 📈 Results

```
Training — Total reward: 7052 / 15000 episodes
Training — Overall success rate: 47.01%
Training — Success rate (last 1000 episodes): 66.50%

Evaluation (greedy policy, 1000 test episodes):
  Success rate: 74.10%
  Average episode length: 44.33 steps
```

The success rate climbs steadily as epsilon decays and the agent
relies more on its learned Q-values — see
`screenshots/success_rate.png`. The final greedy policy (no
exploration) reaches the goal far more often than random chance,
confirming the agent learned a meaningful strategy despite the
slippery, stochastic environment.

- **Rewards over training:** `screenshots/rewards_per_episode.png`
- **Success rate over training:** `screenshots/success_rate.png`
- **Learned Q-table heatmap:** `screenshots/q_table_heatmap.png`
- **Learned policy + full evaluation stats:** `outputs/evaluation_report.txt`

See `documentation/project_report.pdf` for the full write-up.

## 🛠️ Tech Stack

- Python 3
- Gymnasium (FrozenLake-v1 environment)
- NumPy (Q-table + numerical operations)
- Matplotlib (visualization)
- Jupyter Notebook

## 📌 Notes for Submission

- This folder is self-contained and can be zipped/pushed to GitHub as-is.
- All outputs (`q_table.pkl`, plots, reports) are already generated —
  re-running `q_learning_agent.py` will regenerate them deterministically
  (fixed random seed = 42 for training, 123 for evaluation).
- No external dataset is required; `data/environment_config.json`
  simply documents the exact environment configuration used, since the
  environment itself is procedurally generated by Gymnasium.
