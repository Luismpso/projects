import math
import torch
import chess
import numpy as np
from src.board_encoding import encode_board
from src.dataset import encode_move, decode_move

# --- VALORES DAS PEÇAS (A "CABULA") ---
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.2,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0 # O Rei vale infinito na prática, mas aqui é só material
}

def get_material_score(board):
    """
    Calcula quem tem mais peças no tabuleiro.
    Retorna entre -1.0 (Pretas ganham) e 1.0 (Brancas ganham).
    """
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += val
            else:
                score -= val
    
    # Normalizar para ficar entre -1 e 1 (assumindo vantagem máx de 39 pontos)
    return max(-1.0, min(1.0, score / 15.0))

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
            # Q + U formula
            q_value = -child.value() # Inverte a perspetiva (Minimax)
            
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
        root = self.run_simulations(board)
        
        best_move_idx = -1
        max_visits = -1
        
        # Escolhe a jogada mais visitada (a mais robusta)
        for move_idx, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move_idx = move_idx
        
        if best_move_idx == -1:
            return list(board.legal_moves)[0].uci(), 0.0
            
        return decode_move(best_move_idx, board), root.value()

    def search_return_root(self, board):
        return self.run_simulations(board)

    def run_simulations(self, board):
        root = MCTSNode()
        self._expand(root, board)

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()

            # 1. Selection
            while len(node.children) > 0:
                move_idx, node = node.select_child()
                move_uci = decode_move(move_idx, scratch_board)
                try:
                    scratch_board.push_uci(move_uci)
                except:
                    break 
            
            # 2. Expansion & Evaluation
            value = 0
            if not scratch_board.is_game_over():
                value = self._expand(node, scratch_board)
            else:
                if scratch_board.is_checkmate():
                    value = -1.0 # Derrota imediata para quem ia jogar
                else:
                    value = 0.0 # Empate

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
            policy, nn_value = self.model(board_tensor)
        
        # --- A GRANDE MELHORIA (HYBRID EVALUATION) ---
        # 1. Pega no valor que a Rede Neural "acha"
        nn_val = nn_value.item()
        
        # 2. Calcula o valor REAL das peças (Material)
        mat_val = get_material_score(board)
        
        # 3. Mistura os dois! (50% Inteligência Artificial + 50% Contagem de Peças)
        # Isto impede a IA de fazer sacrifícios estúpidos se não souber o que está a fazer.
        # --- AJUSTE DE PESOS ---
        # 0.2 para a IA (estratégia) e 0.8 para Material (contagem de peças)
        combined_value = (0.2 * nn_val) + (0.8 * mat_val)

        # Ajusta a perspetiva (Se é vez das Pretas, inverte o sinal)
        if board.turn == chess.BLACK:
            final_value = -combined_value
        else:
            final_value = combined_value
        # ---------------------------------------------

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
            
        return final_value