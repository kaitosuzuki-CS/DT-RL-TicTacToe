import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from ai.decision_tree import DecisionTree
from ai.q_learning import choose_action
from ai.utils import add_move, check_draw, check_winner

parent_dir = Path(__file__).resolve().parent
checkpoints_dir = parent_dir / "checkpoints"

ckpt_path = {
    "DT": "decision_tree.pkl",
    "DT_RL": "q_table_DT_RL.pkl",
    "RL_RL": "q_table_RL_RL.pkl",
}


def play(rl, dt, player=0, num_games=1):
    wins = defaultdict(int)

    for _ in tqdm(range(num_games)):
        board = [["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]]
        curr_dt = dt
        current_player = 0
        while True:
            if current_player == player:
                move = curr_dt.get_best_move()
            else:
                move = choose_action(rl, board, current_player, 0)

            new_board = add_move(board, move, current_player)

            if not new_board:
                return

            board = new_board

            if check_winner(board, current_player):
                wins[current_player] += 1
                break
            elif check_draw(board):
                wins["draw"] += 1
                break

            current_player = 1 - current_player
            curr_dt = curr_dt.moves[move][0]

    if player == 0:
        print(f'DT wins: {wins[0]}, RL wins: {wins[1]}, Draws: {wins["draw"]}')
    else:
        print(f'RL wins: {wins[0]}, DT wins: {wins[1]}, Draws: {wins["draw"]}')


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
