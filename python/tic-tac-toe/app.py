from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Configurações
HUMAN = 1
AI = -1

def check_win(board):
    # Verifica todas as combinações de vitória
    win_p = [[0,1,2],[3,4,5],[6,7,8], # Linhas
             [0,3,6],[1,4,7],[2,5,8], # Colunas
             [0,4,8],[2,4,6]]         # Diagonais
    
    for p in win_p:
        s = board[p[0]] + board[p[1]] + board[p[2]]
        if s == 3: return HUMAN   # Humano ganhou
        if s == -3: return AI     # IA ganhou
        
    if 0 not in board: return 0   # Empate
    return None                   # Jogo continua

def minimax(board, depth, is_maximizing):
    result = check_win(board)
    
    # Pontuações ajustadas pela profundidade para preferir vitórias rápidas e derrotas tardias
    if result == AI:
        return 100 - depth
    elif result == HUMAN:
        return -100 + depth
    elif result == 0:
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = AI
                score = minimax(board, depth + 1, False)
                board[i] = 0 
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = HUMAN
                score = minimax(board, depth + 1, True)
                board[i] = 0 
                best_score = min(score, best_score)
        return best_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    # Recebe o tabuleiro atual
    board = np.array(request.json['board'], dtype=int)
    
    # Verifica se já acabou antes de jogar
    if check_win(board) is not None:
        return jsonify({'move': None})

    avail = [i for i, x in enumerate(board) if x == 0]
    if not avail:
        return jsonify({'move': None})

    # Estratégias iniciais para a IA:
    # Se o humano começou no canto, a IA deve pegar o centro.
    if len(avail) == 8 and board[4] == 0:
        return jsonify({'move': 4})
    
    # Se o humano começou no centro, a IA deve pegar um canto.
    if len(avail) == 8 and board[4] == 1:
        corners = [0, 2, 6, 8]
        # Escolhe o primeiro canto livre
        for c in corners:
            if board[c] == 0:
                return jsonify({'move': c})

    # Para outras situações, usa o minimax para escolher a melhor jogada
    best_score = -float('inf')
    best_move = None
    
    # Analisa todas as jogadas possíveis
    for i in avail:
        board[i] = AI 
        score = minimax(board, 0, False) 
        board[i] = 0 
        
        if score > best_score:
            best_score = score
            best_move = i
            
    return jsonify({'move': int(best_move)})

if __name__ == '__main__':
    app.run(debug=True)