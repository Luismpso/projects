import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        # Valoração simples das peças para guiar o agente
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
    def reset(self):
        self.board.reset()
        return self.get_state(), self.get_legal_actions()
        
    def get_state(self):
        """
        Retorna o estado num tensor 12 x 8 x 8.
        12 canais: 6 tipos de peças para o jogador atual, 6 para o adversário.
        O tabuleiro é sempre virado para que o agente veja as suas peças
        na base (rank 0) a avançar para o topo (rank 7).
        """
        state = np.zeros((12, 8, 8), dtype=np.float32)
        turn = self.board.turn # Jogador atual
        
        for piece_type in range(1, 7):
            for color in [chess.WHITE, chess.BLACK]:
                squares = self.board.pieces(piece_type, color)
                is_current_player = (color == turn)
                channel = (piece_type - 1) + (0 if is_current_player else 6)
                
                for sq in squares:
                    rank = chess.square_rank(sq)
                    file = chess.square_file(sq)
                    
                    # Virar o tabuleiro se formos as pretas 
                    if turn == chess.BLACK:
                        rank = 7 - rank
                        file = 7 - file
                        
                    state[channel, rank, file] = 1.0
                    
        return state

    def get_legal_actions(self):
        """
        Retorna uma lista de tuplos (action_idx, move).
        O índice de ação é calculado como (from_sq * 64 + to_sq), lidando
        também com a perspetiva de virar o tabuleiro.
        """
        actions = []
        turn = self.board.turn
        
        for move in self.board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            
            if turn == chess.BLACK:
                from_sq = from_sq ^ 63
                to_sq = to_sq ^ 63
                
            action_idx = from_sq * 64 + to_sq
            actions.append((action_idx, move))
            
        return actions

    def _get_material_score(self, color):
        """Calcula a pontuação em material de uma dada cor"""
        score = 0
        for piece_type in range(1, 6): # Ignorar o rei
            score += len(self.board.pieces(piece_type, color)) * self.piece_values[piece_type]
        return score

    def step(self, action_idx):
        """
        Avança um passo no ambiente ao executar uma jogada.
        """
        legal_actions = self.get_legal_actions()
        move = None
        
        # Encontrar a jogada que corresponde ao índice
        # Se houver várias (ex: promoções diferentes), a primeira é selecionada (geralmente Dama)
        for idx, m in legal_actions:
            if idx == action_idx:
                move = m
                break
                
        if move is None:
            # Caso o agente tente jogada ilegal (não deve acontecer devido a masking)
            return self.get_state(), -10, True, {"error": "Invalid action", "legal_actions": self.get_legal_actions()}
            
        current_turn = self.board.turn
        
        my_material_before = self._get_material_score(current_turn)
        opp_material_before = self._get_material_score(not current_turn)
        adv_before = my_material_before - opp_material_before
        
        # Fazer a jogada
        self.board.push(move)
        
        my_material_after = self._get_material_score(current_turn)
        opp_material_after = self._get_material_score(not current_turn)
        adv_after = my_material_after - opp_material_after
        
        # A recompensa é o ganho de vantagem de material
        reward = adv_after - adv_before
        
        # Adicionar recompensa por acabar o jogo
        done = self.board.is_game_over()
        if done:
            result = self.board.result()
            if result == "1-0":
                reward += 100 if current_turn == chess.WHITE else -100
            elif result == "0-1":
                reward += 100 if current_turn == chess.BLACK else -100
            # Empate dá apenas 0
                
        info = {"legal_actions": self.get_legal_actions() if not done else []}
        return self.get_state(), reward, done, info

    def render(self):
        """Mostra o tabuleiro na consola"""
        print(self.board)
        print("-" * 20)
