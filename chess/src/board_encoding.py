import chess
import numpy as np

def encode_board(board):
    """
    Input: Board object
    Output: Matriz (17, 8, 8)
    Canais 0-11: Peças (P,N,B,R,Q,K para Brancas e Pretas)
    Canal 12: Vez de Jogar (1=Brancas, 0=Pretas)
    Canais 13-16: Direitos de Roque (WK, WQ, BK, BQ)
    """
    matrix = np.zeros((17, 8, 8), dtype=np.float32)
    
    # --- 1. PEÇAS (Canais 0-11) ---
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

    # --- 2. VEZ DE JOGAR (Canal 12) ---
    if board.turn == chess.WHITE:
        matrix[12, :, :] = 1.0

    # --- 3. DIREITOS DE ROQUE (Canais 13-16) ---
    # Brancas Rei
    if board.has_kingside_castling_rights(chess.WHITE):
        matrix[13, :, :] = 1.0
    # Brancas Rainha
    if board.has_queenside_castling_rights(chess.WHITE):
        matrix[14, :, :] = 1.0
    # Pretas Rei
    if board.has_kingside_castling_rights(chess.BLACK):
        matrix[15, :, :] = 1.0
    # Pretas Rainha
    if board.has_queenside_castling_rights(chess.BLACK):
        matrix[16, :, :] = 1.0

    return matrix