import torch
from torch.utils.data import Dataset
import chess
import chess.pgn
import numpy as np
from src.board_encoding import encode_board

def encode_move(move_uci):
    """
    Converte string UCI para índice 0-4095.
    Se houver promoção (ex: 'a7a8q'), ignora o 'q' para calcular o índice.
    """
    try:
        # Pega apenas as coordenadas (primeiros 4 caracteres), ignora a promoção
        clean_uci = move_uci[:4] 
        move = chess.Move.from_uci(clean_uci)
        return move.from_square * 64 + move.to_square
    except:
        return None

def decode_move(move_idx, board=None):
    """
    Converte índice de volta para UCI.
    Se 'board' for fornecido, verifica se é promoção e adiciona 'q' (Auto-Queen).
    """
    if not (0 <= move_idx < 4096):
        return None

    from_sq, to_sq = divmod(move_idx, 64)
    move = chess.Move(from_sq, to_sq)
    
    # Lógica de Auto-Promoção para Rainha
    if board:
        # Se é um peão a ir para o topo/fundo
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            # Rank 7 (para brancas indo para 8) ou Rank 0 (para pretas indo para 1)
            if chess.square_rank(to_sq) in [0, 7]:
                move.promotion = chess.QUEEN
                
    return move.uci()

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move_uci, result = self.data[idx]
        board = chess.Board(fen)
        
        # Input (17 canais)
        board_tensor = encode_board(board)
        
        # Target Policy (Índice 0-4095)
        move_idx = encode_move(move_uci)
        if move_idx is None: move_idx = 0 # Fallback raro
        
        # Target Value
        result_float = float(result)
        
        return (
            torch.tensor(board_tensor, dtype=torch.float32),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(result_float, dtype=torch.float32)
        )