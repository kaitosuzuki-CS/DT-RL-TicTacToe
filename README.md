# DT-RL Tic-Tac-Toe

## Project Overview

This project implements a Tic-Tac-Toe AI using two distinct approaches: **Decision Trees (DT)** and **Reinforcement Learning (RL)**. The goal is to compare the performance and training dynamics of an RL agent trained against a perfect opponent (Decision Tree) versus an RL agent trained against itself (Self-Play).

The project explores two primary training modes:

1.  **DT_RL**: A Q-Learning agent trains against a minimax-based Decision Tree agent (a perfect player).
2.  **RL_RL**: A Q-Learning agent trains against another Q-Learning agent (self-play).

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Training](#training)
  - [Playing (Inference)](#playing-inference)
  - [Testing (Benchmarking)](#testing-benchmarking)
- [Project Files](#project-files)

## Project Structure

```
DT-RL-TicTacToe/
├── ai/
│   ├── decision_tree.py   # Decision Tree (Minimax) implementation
│   ├── q_learning.py      # Q-Learning algorithm and training loops
│   └── utils.py           # Game logic (board state, win checking, etc.)
├── checkpoints/           # Directory for saving trained model (.pkl) files
├── infer.py               # Script to play against the trained AI
├── train.py               # Entry point for training the models
├── requirements.txt       # Python dependencies
└── .gitignore             # Git ignore rules
```

## Tech Stack

- **Python 3**: Core programming language.
- **NumPy**: Used for efficient numerical operations.
- **tqdm**: Provides progress bars for training loops.
- **Pickle**: Used for serializing and saving/loading trained models.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Hachiman-potassiumdesu/DT-RL-TicTacToe.git
    cd DT-RL-TicTacToe
    ```

2.  **Set up a virtual environment (optional but recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the AI models, use the `train.py` script. You must specify the training mode.

**Train RL Agent vs. Decision Tree:**

```bash
python train.py --mode DT_RL
```

**Train RL Agent vs. RL Agent (Self-Play):**

```bash
python train.py --mode RL_RL
```

**Optional Arguments:**

- `--num-episodes`: Set the number of training episodes (default: 1,200,000).
  ```bash
  python train.py --mode RL_RL --num-episodes 500000
  ```

### Playing (Inference)

To play Tic-Tac-Toe against a trained AI, use the `infer.py` script.

**Play against the model trained with DT_RL:**

```bash
python infer.py --type DT_RL
```

**Play against the model trained with RL_RL:**

```bash
python infer.py --type RL_RL
```

**Play against the perfect Decision Tree directly:**

```bash
python infer.py --type DT
```

**Optional Arguments:**

- `--first`: Specify if you want to play first (`true`) or second (`false`). Default is `true`.
- `--num-games`: Number of games to play in a row. Default is `1`.

**Example:**

```bash
python infer.py --type DT_RL --first false --num-games 3
```

### Testing (Benchmarking)

To evaluate the trained AI models against the perfect Decision Tree opponent, use the `test_model.py` script.

**Test RL Agent (trained vs DT) against Decision Tree:**

```bash
python test_model.py --type DT_RL
```

**Test RL Agent (trained vs Self) against Decision Tree:**

```bash
python test_model.py --type RL_RL
```

**Optional Arguments:**

- `--first`: Specify if the RL agent plays first (`true`) or second (`false`). Default is `true`.
- `--num-games`: Number of games to simulate. Default is `1`.

**Example:**

```bash
python test_model.py --type RL_RL --first true --num-games 100
```

This will simulate 100 games where the RL agent plays first against the Decision Tree.

## Project Files

- **`train.py`**: The main driver for training. It parses command-line arguments to select the training mode (`DT_RL` or `RL_RL`) and initiates the training process.
- **`infer.py`**: The interface for users to play against the AI. It loads the saved models from the `checkpoints/` directory and handles the game loop, input validation, and board rendering.
- **`test_model.py`**: A benchmarking script to test the performance of trained RL models against the perfect Decision Tree opponent. It tracks wins, losses, and draws over a specified number of games.
- **`ai/decision_tree.py`**: Contains the `DecisionTree` class. This implements a minimax-style algorithm to determine the optimal move for any given board state. It can be used as a perfect opponent for training or gameplay.
- **`ai/q_learning.py`**: Contains the Q-Learning logic.
  - `train_RL_RL`: Logic for self-play training.
  - `train_with_DT`: Logic for training the RL agent against the Decision Tree.
  - `choose_action`: Epsilon-greedy strategy for action selection.
- **`ai/utils.py`**: Helper functions for the game mechanics, such as `check_winner`, `check_draw`, `add_move`, and board representation conversions.
- **`requirements.txt`**: Lists the library dependencies required to run the project.
