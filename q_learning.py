import numpy as np
import random
import pickle
from utils import check_winner, check_draw, add_move, convert_list_to_tuple, convert_tuple_to_list

Q = {}

def get_afterstates(board, player):
    moves = []
    next_states = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == '_':
                moves.append((row, col))

                new_board = convert_tuple_to_list(board)
                new_board[row][col] = 'X' if player == 0 else 'O'

                new_board = convert_list_to_tuple(new_board)
                next_states.append(new_board)
    
    return moves, next_states

def initialize_q(board=[['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']], player=0):
    action, next_states = get_afterstates(board, player)
    
    for next_state in next_states:
        if not next_state in Q:
            Q[next_state] = 0.0

            if check_winner(next_state, player) or check_draw(next_state):
                continue
            
            initialize_q(next_state, 1 - player)

def get_best_score(Q, states):
    best_score = max([Q[state] for state in states])
    return best_score

def get_max_action(Q, states, actions):
    best_score = get_best_score(Q, states)
    best_actions = [state for state in states if Q[state] == best_score]

    return actions[states.index(random.choice(best_actions))]
    

def choose_action(Q, state, player, epsilon=0.1):
    actions, next_states = get_afterstates(state, player)

    if np.random.rand() < epsilon:
        return random.choice(actions)
    
    return get_max_action(Q, next_states, actions)

def train(dt):
    initialize_q()
    num_episodes = 1000000
    alpha = 0.001
    gamma = 0.99
    epsilon = 1.0

    total_reward = 0
    total_loss = 0

    for _ in range(num_episodes):
        ai = dt
        state = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
        player = 0
        reward = 0

        while True:
            action = choose_action(Q, state, player, epsilon)
            afterstate = add_move(state, action, player)

            # print(ai.moves)

            if check_winner(afterstate, player):
                reward = 1
                break
            elif check_draw(afterstate):
                reward = 1
                break

            ai = ai.moves[action][0]

            DT_move = ai.get_best_move()
            # DT_move = choose_action(afterstate, 1 - player, 1)
            new_state = add_move(afterstate, DT_move, 1 - player)

            if check_winner(new_state, 1 - player):
                total_loss += 1
                reward = -1
                break
            elif check_draw(new_state):
                reward = 1
                break

            ___, new_afterstates = get_afterstates(new_state, player)

            Q[afterstate] = Q.get(afterstate, 0) + alpha * (reward + gamma * get_best_score(new_afterstates) - Q.get(afterstate, 0))

            state = new_state
            ai = ai.moves[DT_move][0]
        
        Q[afterstate] = Q.get(afterstate, 0) + alpha * (reward - Q.get(afterstate, 0))

        total_reward += reward

        if _ % 1000 == 0:
            epsilon = epsilon * 0.99

        if _ % 10000 == 0:
            print(f"Episode {_}: Epsilon: {epsilon}, Alpha: {alpha}, Reward: {total_reward / 10000}, Loss: {total_loss}")
            total_reward = 0 
            total_loss = 0

    with open('checkpoints/q_table_DT.pkl', 'wb') as f:
        pickle.dump(Q, f)

def play(Q, num_games=10, dt=None):
    state = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
    player_0_wins = 0
    player_1_wins = 0
    draws = 0

    for _ in range(num_games):
        ai = dt
        while True:
            action = choose_action(Q, state, 0, 0)
            afterstate = add_move(state, action, 0)

            if check_winner(afterstate, 0):
                player_0_wins += 1
                break
            elif check_draw(afterstate):
                draws += 1
                break

            if ai:
                ai = ai.moves[action][0]
                DT_move = ai.get_best_move()
            else:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))

                DT_move = (row, col)

            new_state = add_move(afterstate, DT_move, 1)

            if check_winner(new_state, 1):
                player_1_wins += 1
                break
            elif check_draw(new_state):
                draws += 1

            state = new_state
            if ai:
                ai = ai.moves[DT_move][0]

    print(f'Player 0 wins: {player_0_wins}, Player 1 wins: {player_1_wins}, Draws: {draws}')