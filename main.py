import pickle

from q_learning import train, play
from decision_tree import DecisionTree

# if __name__ == "__main__":
#     ai = DecisionTree()
#     ai.train()

#     train(ai)


if __name__ == "__main__":
    with open('checkpoints/decision_tree.pkl', 'rb') as f:
        ai = pickle.load(f)

    with open('checkpoints/q_table_DT.pkl', 'rb') as f:
        Q = pickle.load(f)

    play(Q, 100, ai)


