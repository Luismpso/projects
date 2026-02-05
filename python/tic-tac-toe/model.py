import numpy as np
import pickle

class QLearner:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.2
        self.gamma = 0.9
        self.epsilon = 0.3

    def get_state_key(self, board):
        return str(board.tolist())

    def choose_action(self, board, available_moves):
        if np.random.uniform(0, 1) < self.epsilon:
            return available_moves[np.random.choice(len(available_moves))]
        
        values = []
        for m in available_moves:
            next_board = board.copy()
            next_board[m] = -1
            values.append(self.q_table.get(self.get_state_key(next_board), 0))
        return available_moves[np.argmax(values)]