import argparse

from ai.decision_tree import DecisionTree
from ai.q_learning import train_RL_RL, train_with_DT, train_with_self_play

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training Tic-Tac-Toe AI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["DT_RL", "RL_RL"],
        required=True,
        help="Training mode: DT_RL (Decision Tree vs Q-Learning), RL_RL (Q-Learning vs Q-Learning)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1200000,
        help="Number of episodes for training",
    )

    args = parser.parse_args()

    if args.mode == "DT_RL":
        ai = DecisionTree()
        ai.train()

        train_with_DT(ai, args.num_episodes)
    elif args.mode == "RL_RL":
        train_with_self_play(args.num_episodes)
    else:
        print("Invalid mode selected.")
