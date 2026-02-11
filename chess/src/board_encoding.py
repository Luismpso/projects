import chess
import numpy as np

def encode_board(board):
    """
    Versão 12 Canais (Compatível com o modelo antigo/rápido).
    Output: Matriz (12, 8, 8)
    """
    matrix = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[piece.symbol()]
            row, col = divmod(square, 8)
            matrix[channel, row, col] = 1.0

    return matrix