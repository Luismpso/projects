import chess
import numpy as np

def encode_board(board):
    """
    Converte um tabuleiro de xadrez (objeto python-chess) numa matriz numpy 
    pronta para entrar numa Rede Neural (PyTorch).
    
    Output: Matriz de dimensão (12, 8, 8)
    """
    
    # Criamos uma matriz de zeros: 12 canais, 8 linhas, 8 colunas
    # 12 canais = 6 tipos de peças brancas + 6 tipos de peças pretas
    matrix = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Dicionário para mapear peças aos canais da matriz
    # P=Peão, N=Cavalo, B=Bispo, R=Torre, Q=Rainha, K=Rei
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Brancas
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Pretas
    }
    
    # Percorrer as 64 casas do tabuleiro
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece:
            # Pega no símbolo da peça (ex: 'P' ou 'k')
            symbol = piece.symbol()
            
            # Descobre qual é o canal correto para esta peça
            channel = piece_to_channel[symbol]
            
            # O python-chess usa 0-63 linear, precisamos converter para linha/coluna (8x8)
            # divmod devolve (linha, coluna)
            row, col = divmod(square, 8)
            
            # Colocamos um "1" na posição onde a peça está
            matrix[channel, row, col] = 1.0

    return matrix

# Bloco de teste
if __name__ == "__main__":
    # 1. Cria um tabuleiro na posição inicial
    board = chess.Board()
    print("Tabuleiro Original:")
    print(board)
    
    # 2. Converte para números
    encoded = encode_board(board)
    
    print("\nDimensão da Matriz:", encoded.shape) # Deve ser (12, 8, 8)
    
    # 3. Vamos ver o Canal 0 (Peões Brancos)
    print("\nCanal 0 (Onde estão os Peões Brancos?):")
    print(encoded[0])