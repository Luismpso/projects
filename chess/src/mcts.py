import math
import torch
import chess
import numpy as np
from src.board_encoding import encode_board
from src.dataset import encode_move, decode_move

class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}  
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.5):
        best_score = -float('inf')
        best_child = None
        best_move_idx = -1

        for move_idx, child in self.children.items():
            # Importante: A perspetiva inverte a cada nível da árvore
            q_value = -child.value()
            
            u_value = c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child
                best_move_idx = move_idx
        
        return best_move_idx, best_child

class MCTS:
    def __init__(self, model, device, num_simulations=50):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    def search(self, board):
        # Para jogar (devolve apenas a melhor jogada)
        root = self.run_simulations(board)
        best_move_idx = -1
        max_visits = -1
        for move_idx, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move_idx = move_idx
        if best_move_idx == -1:
            return list(board.legal_moves)[0].uci(), 0.0
        return decode_move(best_move_idx), root.value()

    def search_return_root(self, board):
        # Para análise (devolve a árvore toda)
        return self.run_simulations(board)

    def run_simulations(self, board):
        root = MCTSNode()
        # Avalia a raiz (posição atual)
        self._expand(root, board)

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()

            # 1. Selection
            while len(node.children) > 0:
                move_idx, node = node.select_child()
                try:
                    move = chess.Move.from_uci(decode_move(move_idx))
                    scratch_board.push(move)
                except: break # Evita erros raros
            
            # 2. Expansion & Evaluation
            value = 0
            if not scratch_board.is_game_over():
                value = self._expand(node, scratch_board)
            else:
                # Se jogo acabou: Se levei mate é -1 (mau para quem ia jogar)
                if scratch_board.is_checkmate():
                    value = -1.0 
                else:
                    value = 0.0

            # 3. Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                value = -value 
        return root

    def _expand(self, node, board):
        board_numpy = encode_board(board)
        board_tensor = torch.tensor(board_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.model(board_tensor)
        
        # --- A GRANDE CORREÇÃO ---
        val = value.item()
        # Se for a vez das Pretas, inverte o valor!
        # A rede diz: +1 (Brancas). Se sou preto, isso é -1 para mim.
        if board.turn == chess.BLACK:
            val = -val 
        # -------------------------

        probs = torch.exp(policy).cpu().numpy()[0]
        legal_moves = list(board.legal_moves)
        policy_sum = 0
        valid_children = {}
        
        for move in legal_moves:
            idx = encode_move(move.uci())
            if idx is not None:
                prob = probs[idx]
                policy_sum += prob
                valid_children[idx] = prob
        
        for idx, prob in valid_children.items():
            if policy_sum > 0:
                norm_prob = prob / policy_sum
            else:
                norm_prob = 1.0 / len(legal_moves)
            node.children[idx] = MCTSNode(parent=node, prior=norm_prob)
            
        return val