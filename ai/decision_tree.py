import os
import pickle
import random
from pathlib import Path

from ai.utils import add_move, check_draw, check_winner

parent_dir = Path(__file__).resolve().parent.parent
checkpoints_dir = parent_dir / "checkpoints"
checkpoints_dir.mkdir(exist_ok=True)


class DecisionTree:
    def __init__(
        self, board=(("_", "_", "_"), ("_", "_", "_"), ("_", "_", "_")), player=0
    ):
        self.moves = {}
        self.board = board
        self.player = player

    def __str__(self):
        for row in self.board:
            print(" ".join(row))
        return f"Player: {self.player}, Move: {self.moves}"

    def get_best_score(self):
        if self.player == 0:
            return self.moves[max(self.moves, key=lambda x: self.moves[x][1])][1]
        else:
            return self.moves[min(self.moves, key=lambda x: self.moves[x][1])][1]

    def get_best_move(self):
        best_score = self.get_best_score()
        options = [m for m in self.moves if self.moves[m][1] == best_score]
        return random.choice(options) if options else None

    def _train(self):
        prev_player = 1 - self.player
        if check_winner(self.board, prev_player):
            return 1 if prev_player == 0 else -1
        if check_draw(self.board):
            return 0

        for row in range(3):
            for col in range(3):
                move = (row, col)

                new_board = add_move(self.board, move, self.player)

                if new_board:
                    child_tree = DecisionTree(new_board, 1 - self.player)

                    self.moves[move] = (child_tree, child_tree._train())

        return self.get_best_score()

    def train(self):
        self._train()

        with open(os.path.join(checkpoints_dir, "decision_tree.pkl"), "wb") as f:
            pickle.dump(self, f)
