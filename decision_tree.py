import pickle
import random

def check_row(board, player):
    for row in board:
        if row.count(player) == 3:
            return True
    return False

def check_column(board, player):
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    return False

def check_diagonal(board, player):
    if all(board[i][i] == player for i in range(3)):
        return True
    if all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def check_draw(board):
    for row in board:
        if '_' in row:
            return False
    return True

def check_winner(board, player):
    player = 'X' if player == 0 else 'O'
    return check_row(board, player) or \
           check_column(board, player) or \
           check_diagonal(board, player)

def add_move(board, move, player):
    row, col = move

    try:
        if row < 0 or row >= 3 or col < 0 or col >= 3:
            return False, board
        
        if board[row][col] == '_':
            new_board = [row[:] for row in board]
            new_board[row][col] = 'X' if player == 0 else 'O'
            return True, new_board
        
        return False, board
    except:
        return False, board
    
class DecisionTree():
    def __init__(self, board=[['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']], player=0):
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
        print(options)
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

                placement, new_board = add_move(self.board, move, self.player)
                
                if placement:
                    child_tree = DecisionTree(new_board, 1 - self.player)

                    self.moves[move] = (child_tree, child_tree.train())
        
        return self.get_best_score()

def play():
    board = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
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
        
        valid_move, board = add_move(board, move, player)
        
        if not valid_move:
            print("Invalid move, try again.")
            continue
            
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

# if __name__ == "__main__":
#     ai = DecisionTree()
#     ai.train()

#     with open('decision_tree.pkl', 'wb') as f:
#         pickle.dump(ai, f)

if __name__ == "__main__":
    play()