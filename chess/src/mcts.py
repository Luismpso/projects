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
        """Retorna o valor médio (Q) deste nó."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.5):
        """
        Escolhe a melhor jogada para explorar usando a fórmula PUCT (AlphaZero).
        Equilibra:
        1. Valor (Q): O quão boa a jogada parece ser.
        2. Exploração (U): Jogadas que a rede neural sugeriu mas ainda visitamos pouco.
        """
        best_score = -float('inf')
        best_child = None
        best_move_idx = -1

        for move_idx, child in self.children.items():
            q_value = -child.value()
            
            u_value = c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child
                best_move_idx = move_idx
        
        return best_move_idx, best_child

class MCTS:
    def __init__(self, model, device, num_simulations=600):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    def search(self, board):
        """
        Executa a busca MCTS e retorna a melhor jogada (string) e a avaliação.
        Usado pelo Jogo (app.py).
        """
        root = self.run_simulations(board)
        best_move_idx = -1
        max_visits = -1
        
        for move_idx, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move_idx = move_idx
        
        return decode_move(best_move_idx), root.value()

    def search_return_root(self, board):
        """
        Executa a busca e retorna o objeto ROOT.
        Usado pelo Treino RL (train_rl.py) para extrair probabilidades.
        """
        return self.run_simulations(board)

    def run_simulations(self, board):
        """Lógica central das simulações."""
        root = MCTSNode()
        
        # Expande a raiz inicial
        self._expand(root, board)

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()

            # 1. Selection (Descer na árvore)
            while len(node.children) > 0:
                move_idx, node = node.select_child()
                move_str = decode_move(move_idx)
                move = chess.Move.from_uci(move_str)
                
                # Segurança contra jogadas ilegais (raro, mas possível)
                if move in scratch_board.legal_moves:
                    scratch_board.push(move)
                else:
                    break 
            
            # 2. Expansion & Evaluation (O nó folha)
            value = 0
            if not scratch_board.is_game_over():
                # Se o jogo não acabou, expandimos o nó com a Rede Neural
                value = self._expand(node, scratch_board)
            else:
                # Se acabou, vemos quem ganhou
                if scratch_board.is_checkmate():
                    value = -1.0 # O nó atual perdeu (o anterior deu mate)
                else:
                    value = 0.0 # Empate

            # 3. Backpropagation (Subir na árvore)
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                value = -value # Inverte a perspetiva a cada nível

        return root

    def _expand(self, node, board):
        """
        Usa a Rede Neural para avaliar o tabuleiro e criar filhos para o nó.
        """
        # Preparar input para a GPU
        board_numpy = encode_board(board)
        board_tensor = torch.tensor(board_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Previsão da Rede
        with torch.no_grad():
            policy, value = self.model(board_tensor)
        
        # Converter log_softmax para probabilidade real (0 a 1)
        probs = torch.exp(policy).cpu().numpy()[0]
        
        # Filtrar apenas jogadas legais
        legal_moves = list(board.legal_moves)
        policy_sum = 0
        
        for move in legal_moves:
            # Converter jogada 'e2e4' para índice 0-4095
            idx = encode_move(move.uci())
            
            if idx is not None:
                prob = probs[idx]
                policy_sum += prob
                # Criar novo nó filho
                node.children[idx] = MCTSNode(parent=node, prior=prob)
        
        # Normalizar as probabilidades
        for child in node.children.values():
            if policy_sum > 0:
                child.prior /= policy_sum
            else:
                # Fallback: Se a rede der 0% a tudo, distribui uniformemente
                child.prior = 1.0 / len(legal_moves)
            
        return value.item()