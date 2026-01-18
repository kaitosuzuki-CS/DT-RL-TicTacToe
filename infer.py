import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

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


def play(player=0, num_games=1, rl=None, dt=None):
    wins = defaultdict(int)
    for _ in range(num_games):
        print("----------")
        board = [["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]]
        curr_dt = dt
        current_player = 0

        while True:
            if current_player == player:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter column (0-2): "))
                except ValueError:
                    print("Invalid input, please enter numbers between 0 and 2.")
                    continue
                move = (row, col)

                print("Player's move:", move)
            else:
                if rl:
                    move = choose_action(rl, board, current_player, 0)
                    print(f"AI's move: {move}")
                elif curr_dt:
                    move = curr_dt.get_best_move()
                    print(f"AI's move: {move}")

            new_board = add_move(board, move, current_player)

            if not new_board:
                print("Invalid move, try again.")
                continue

            board = new_board

            print("Current board:")
            for row in board:
                print(" ".join(row))

            if check_winner(board, current_player):
                print(f"Player {current_player} wins!")
                wins[current_player] += 1
                break
            elif check_draw(board):
                print("It's a draw!")
                wins["draw"] += 1
                break

            current_player = 1 - current_player
            if curr_dt:
                curr_dt = curr_dt.moves[move][0]

    print(f'Player 0 wins: {wins[0]}, Player 1 wins: {wins[1]}, Draws: {wins["draw"]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for playing against Tic-Tac-Toe AI"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["DT_RL", "RL_RL", "DT"],
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

    if args.type in ("DT_RL", "RL_RL"):
        ckpt_file = os.path.join(checkpoints_dir, ckpt_path[args.type])

        with open(ckpt_file, "rb") as f:
            Q = pickle.load(f)

        play(player=0 if args.first == "true" else 1, num_games=args.num_games, rl=Q)
    elif args.type == "DT":
        ckpt_file = os.path.join(checkpoints_dir, ckpt_path[args.type])

        with open(ckpt_file, "rb") as f:
            ai = pickle.load(f)

        play(player=0 if args.first == "true" else 1, num_games=args.num_games, dt=ai)
