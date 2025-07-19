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

def convert_list_to_tuple(board):
    return tuple(tuple(row) for row in board)

def convert_tuple_to_list(board):
    return [list(row) for row in board]

def add_move(board, move, player):
    row, col = move

    try:
        if row < 0 or row >= 3 or col < 0 or col >= 3:
            return None
        
        if board[row][col] == '_':
            new_board = convert_tuple_to_list(board)
            new_board[row][col] = 'X' if player == 0 else 'O'

            new_board = convert_list_to_tuple(new_board)
            return new_board
        
        return None
    except:
        return None
    