import torch
from torch.utils.data import Dataset
import chess
import chess.pgn
import numpy as np
from src.board_encoding import encode_board

# Funções de Codificação de Jogadas

def encode_move(move_uci):
    """
    Converte uma jogada string (ex: 'e2e4') num índice inteiro (0-4095).
    Lógica: (casa_origem * 64) + casa_destino
    """
    move = chess.Move.from_uci(move_uci)
    return move.from_square * 64 + move.to_square

def decode_move(move_idx):
    """
    Converte um índice inteiro (0-4095) de volta para uma jogada string.
    """
    if not (0 <= move_idx < 4096):
        return None # Índice inválido

    from_square, to_square = divmod(move_idx, 64)
    
    move = chess.Move(from_square, to_square)
    return move.uci()

# Classe Dataset
class ChessDataset(Dataset):
    def __init__(self, data):
        """
        data: Lista de tuplos (FEN, Jogada_UCI, Resultado)
        Ex: [("rnbqk...", "e2e4", 1.0), ...]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move_uci, result = self.data[idx]
        
        # 1. Preparar o Tabuleiro (Input)
        board = chess.Board(fen)
        board_tensor = encode_board(board) # Retorna numpy (12, 8, 8)
        
        # 2. Preparar a Jogada (Target Policy)
        # O modelo quer um índice inteiro para o CrossEntropyLoss
        move_idx = encode_move(move_uci)
        
        # 3. Preparar o Resultado (Target Value)
        # O modelo quer um float (1.0, 0.0 ou -1.0)
        result_float = float(result)
        
        # Converter tudo para tensores PyTorch
        return (
            torch.tensor(board_tensor, dtype=torch.float32), # Input
            torch.tensor(move_idx, dtype=torch.long),        # Target Move (classe)
            torch.tensor(result_float, dtype=torch.float32)  # Target Winner
        )

# Carrega jogos de um ficheiro PGN e extrai os dados para o dataset
def load_pgn_file(file_path, max_games=100):
    games_data = []
    with open(file_path) as pgn:
        count = 0
        while count < max_games:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
            if game is None: break
            
            result = game.headers.get("Result", "*")
            if result == "1-0": val = 1.0
            elif result == "0-1": val = -1.0
            elif result == "1/2-1/2": val = 0.0
            else: continue # Pula jogos sem resultado
            
            board = game.board()
            for move in game.mainline_moves():
                # Guarda: (Estado Antes, Jogada Feita, Quem Ganhou no Fim)
                games_data.append((board.fen(), move.uci(), val))
                board.push(move)
            
            count += 1
            if count % 10 == 0:
                print(f"Jogos processados: {count}")
                
    return games_data