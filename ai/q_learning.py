import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ai.decision_tree import DecisionTree
from ai.utils import (
    add_move,
    check_draw,
    check_winner,
    convert_list_to_tuple,
    convert_tuple_to_list,
)

parent_dir = Path(__file__).resolve().parent.parent
checkpoints_dir = parent_dir / "checkpoints"
checkpoints_dir.mkdir(exist_ok=True)

Q = {}


def get_afterstates(board, player):
    moves = []
    next_states = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == "_":
                moves.append((row, col))

                new_board = convert_tuple_to_list(board)
                new_board[row][col] = "X" if player == 0 else "O"

                new_board = convert_list_to_tuple(new_board)
                next_states.append(new_board)

    return moves, next_states


def initialize_q(board=[["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]], player=0):
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


def train_RL_RL(num_episodes=1200000):
    initialize_q()
    alpha = 0.001
    gamma = 0.99
    epsilon = 1.0

    total_reward = 0
    total_loss = 0

    for _ in range(1, num_episodes + 1):
        state = [["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]]
        reward = 0

        new_state = None

        while True:
            action = choose_action(Q, state, 0, epsilon)
            afterstate = add_move(state, action, 0)

            if check_winner(afterstate, 0):
                reward = 1
                break
            elif check_draw(afterstate):
                reward = 0
                break

            if new_state:
                __, new_afterstates = get_afterstates(afterstate, 1)
                Q[new_state] = Q.get(new_state, 0) + alpha * (
                    reward
                    + gamma * get_best_score(Q, new_afterstates)
                    - Q.get(new_state, 0)
                )

            DT_move = choose_action(Q, afterstate, 1, epsilon)
            new_state = add_move(afterstate, DT_move, 1)

            if check_winner(new_state, 1):
                total_loss += 1
                reward = -1
                break
            elif check_draw(new_state):
                reward = 0
                break

            ___, new_afterstates = get_afterstates(new_state, 0)

            Q[afterstate] = Q.get(afterstate, 0) + alpha * (
                reward
                + gamma * get_best_score(Q, new_afterstates)
                - Q.get(afterstate, 0)
            )

            state = new_state

        Q[afterstate] = Q.get(afterstate, 0) + alpha * (reward - Q.get(afterstate, 0))
        Q[new_state] = Q.get(new_state, 0) + alpha * (-reward - Q.get(new_state, 0))

        total_reward += reward

        if _ % 1000 == 0:
            epsilon = epsilon * 0.99 if epsilon * 0.99 >= 0.0005 else 0
            alpha = 0.001 if epsilon * 0.99 >= 0.0005 else 0.0001

        if _ % 10000 == 0:
            print(
                f"Episode {_}: Epsilon: {epsilon}, Alpha: {alpha}, Reward: {total_reward / 10000}, Loss: {total_loss}"
            )
            total_reward = 0
            total_loss = 0

    with open(os.path.join(checkpoints_dir, "q_table_RL_RL.pkl"), "wb") as f:
        pickle.dump(Q, f)


def train_RL_DT(dt, num_episodes=1200000):
    alpha = 0.001
    gamma = 0.99
    epsilon = 1.0

    total_reward = 0
    total_loss = 0

    for _ in range(1, num_episodes + 1):
        ai = dt
        state = [["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]]
        reward = 0

        while True:
            action = choose_action(Q, state, 0, epsilon)
            afterstate = add_move(state, action, 0)

            if check_winner(afterstate, 0):
                reward = 1
                break
            elif check_draw(afterstate):
                reward = 0
                break

            ai = ai.moves[action][0]

            DT_move = ai.get_best_move()
            new_state = add_move(afterstate, DT_move, 1)

            if check_winner(new_state, 1):
                total_loss += 1
                reward = -1
                break
            elif check_draw(new_state):
                reward = 0
                break

            ___, new_afterstates = get_afterstates(new_state, 0)

            Q[afterstate] = Q.get(afterstate, 0) + alpha * (
                reward
                + gamma * get_best_score(Q, new_afterstates)
                - Q.get(afterstate, 0)
            )

            state = new_state
            ai = ai.moves[DT_move][0]

        Q[afterstate] = Q.get(afterstate, 0) + alpha * (reward - Q.get(afterstate, 0))

        total_reward += reward

        if _ % 1000 == 0:
            epsilon = epsilon * 0.99 if epsilon * 0.99 >= 0.0005 else 0
            alpha = 0.001 if epsilon * 0.99 >= 0.0005 else 0.0001

        if _ % 10000 == 0:
            print(
                f"Episode {_}: Epsilon: {epsilon}, Alpha: {alpha}, Reward: {total_reward / 10000}, Loss: {total_loss}"
            )
            total_reward = 0
            total_loss = 0


def train_DT_RL(dt, num_episodes=1200000):
    alpha = 0.001
    gamma = 0.99
    epsilon = 1.0

    total_reward = 0
    total_loss = 0

    for _ in range(1, num_episodes + 1):
        ai = dt
        state = [["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]]
        reward = 0

        new_state = None

        while True:
            DT_move = ai.get_best_move()
            afterstate = add_move(state, DT_move, 0)

            if check_winner(afterstate, 0):
                reward = -1
                total_loss += 1
                break
            elif check_draw(afterstate):
                reward = 0
                break

            if new_state:
                __, new_afterstates = get_afterstates(afterstate, 1)
                Q[new_state] = Q.get(new_state, 0) + alpha * (
                    reward
                    + gamma * get_best_score(Q, new_afterstates)
                    - Q.get(new_state, 0)
                )

            action = choose_action(Q, afterstate, 1, epsilon)
            new_state = add_move(afterstate, action, 1)

            if check_winner(new_state, 1):
                reward = 1
                break
            elif check_draw(new_state):
                reward = 0
                break

            state = new_state

            ai = ai.moves[DT_move][0]
            ai = ai.moves[action][0]

        Q[new_state] = Q.get(new_state, 0) + alpha * (reward - Q.get(new_state, 0))

        total_reward += reward

        if _ % 1000 == 0:
            epsilon = epsilon * 0.99 if epsilon * 0.99 >= 0.0005 else 0
            alpha = 0.001 if epsilon * 0.99 >= 0.0005 else 0.0001

        if _ % 10000 == 0:
            print(
                f"Episode {_}: Epsilon: {epsilon}, Alpha: {alpha}, Reward: {total_reward / 10000}, Loss: {total_loss}"
            )
            total_reward = 0
            total_loss = 0


def train_with_DT(dt, num_episodes=1200000):
    initialize_q()

    train_DT_RL(dt, num_episodes)
    train_RL_DT(dt, num_episodes)

    with open(os.path.join(checkpoints_dir, "q_table_DT_RL.pkl"), "wb") as f:
        pickle.dump(Q, f)


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
    with open(os.path.join(checkpoints_dir, "q_table_DT_RL.pkl"), "rb") as f:
        q = pickle.load(f)

    with open(os.path.join(checkpoints_dir, "decision_tree.pkl"), "rb") as f:
        dt = pickle.load(f)

    play(q, dt, player=0, num_games=100)  # DT RL
    play(q, dt, player=1, num_games=100)  # RL DT
