import pickle
import random
from utils import add_move, check_winner, check_draw
class DecisionTree():
    def __init__(self, board=(('_', '_', '_'), ('_', '_', '_'), ('_', '_', '_')), player=0):
        self.moves = {}
        self.board = board
        self.player = player

    def __str__(self):
        for row in self.board:
            print(' '.join(row))
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

    def train(self):
        prev_player = 1 - self.player
        if check_winner(self.board, prev_player):
            return 1 if prev_player == 0 else -2
        if check_draw(self.board):
            return -1
        
        for row in range(3):
            for col in range(3):
                move = (row, col)

                new_board = add_move(self.board, move, self.player)
                
                if new_board:
                    child_tree = DecisionTree(new_board, 1 - self.player)

                    self.moves[move] = (child_tree, child_tree.train())
        
        return self.get_best_score()

def play():
    board = (('_', '_', '_'), ('_', '_', '_'), ('_', '_', '_'))
    player = 0

    with open('decision_tree.pkl', 'rb') as f:
        ai = pickle.load(f)
    
    while True:
        if player == 0:
            move = ai.get_best_move()
            print(f"AI's move: {move}")
        elif player == 1:
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
            except ValueError:
                print("Invalid input, please enter numbers between 0 and 2.")
                continue
            move = (row, col)

            print("Player's move:", move)
        
        new_board = add_move(board, move, player)
        
        if not new_board:
            print("Invalid move, try again.")
            continue

        board = new_board
            
        print("Current board:")
        for row in board:
            print(' '.join(row))

        if check_winner(board, player):
            print(f"Player {player} wins!")
            break
        elif check_draw(board):
            print("It's a draw!")
            break
            
        player = 1 - player
        ai = ai.moves[move][0]