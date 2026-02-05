import numpy as np
import pickle

# Configurações
AI_PIECE = -1  
HUMAN_PIECE = 1 

# Dicionário para armazenar a estratégia perfeita 
policy = {}

def check_win(board):
    win_p = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for p in win_p:
        s = board[p[0]] + board[p[1]] + board[p[2]]
        if s == 3: return 1
        if s == -3: return -1
    if 0 not in board: return 0
    return None

def get_state_key(board):
    return str(board.tolist())

def minimax(board, depth, is_maximizing):
    result = check_win(board)
    # Pontuações ajustadas pela profundidade para preferir vitórias rápidas e derrotas tardias
    if result == -1: 
        return 100 - depth 
    elif result == 1: 
        return -100 + depth
    elif result == 0: 
        return 0

    state_key = get_state_key(board)

    if is_maximizing:
        best_score = -float('inf')
        avail = [i for i, x in enumerate(board) if x == 0]
        for move in avail:
            board[move] = AI_PIECE
            score = minimax(board, depth + 1, False) 
            board[move] = 0 
            best_score = max(score, best_score)
        # Guardamos o valor do estado onde é a vez da IA jogar
        policy[state_key] = best_score
        return best_score
    
    else:
        # Turno do Humano
        best_score = float('inf')
        avail = [i for i, x in enumerate(board) if x == 0]
        for move in avail:
            board[move] = HUMAN_PIECE
            score = minimax(board, depth + 1, True) 
            board[move] = 0 
            best_score = min(score, best_score)
        # Guardamos o valor do estado onde é a vez do Humano jogar
        policy[state_key] = best_score
        return best_score

def train():
    print("A calcular a estratégia perfeita...")
    board = np.zeros(9, dtype=int)
    
    # Executa o algoritmo para todas as possibilidades
    # 1. Caso o Humano comece
    minimax(board, 0, False)
    
    # 2. Caso a IA comece
    # O minimax acima já preenche quase tudo, mas para garantir o estado inicial da IA:
    minimax(board, 0, True)

    print(f"Mapeamento concluído! {len(policy)} posições memorizadas.")
    
    with open('policy.pkl', 'wb') as f:
        pickle.dump(policy, f)
    print("Ficheiro 'policy.pkl' gerado com sucesso.")

if __name__ == "__main__":
    train()