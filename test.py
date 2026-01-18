import argparse
import os
import pickle
from pathlib import Path

from ai.decision_tree import DecisionTree
from ai.q_learning import choose_action, play
from ai.utils import add_move, check_draw, check_winner

parent_dir = Path(__file__).resolve().parent
checkpoints_dir = parent_dir / "checkpoints"

ckpt_path = {
    "DT": "decision_tree.pkl",
    "DT_RL": "q_table_DT_RL.pkl",
    "RL_RL": "q_table_RL_RL.pkl",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for testing Tic-Tac-Toe AI")
    parser.add_argument(
        "--type",
        type=str,
        choices=["DT_RL", "RL_RL"],
        required=True,
        help="Opponent Training Type: Decision Tree vs Q-Learning (DT_RL), Q-Learning vs Q-Learning (RL_RL), Decision Tree (DT)",
    )
    parser.add_argument(
        "--first",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether the player plays first or second",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play",
    )

    args = parser.parse_args()

    with open(os.path.join(checkpoints_dir, ckpt_path[args.type]), "rb") as f:
        Q = pickle.load(f)

    with open(os.path.join(checkpoints_dir, ckpt_path["DT"]), "rb") as f:
        dt = pickle.load(f)

    play(Q, dt, player=1 if args.first == "true" else 0, num_games=args.num_games)
