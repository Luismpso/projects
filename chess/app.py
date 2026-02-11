from flask import Flask, render_template, request, jsonify
import torch
import chess
from src.model import ChessNet
from src.mcts import MCTS

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/reinforcement.pth"

# Carregar Modelo
print(f"A carregar IA no {DEVICE}...")
model = ChessNet()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Modelo carregado com sucesso!")
except FileNotFoundError:
    print("⚠️ AVISO: Modelo não encontrado.")

model.to(DEVICE)
model.eval()

# Criar Engine MCTS
ai_engine = MCTS(model, DEVICE, num_simulations=600)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def get_move():
    data = request.json
    board = chess.Board(data.get('fen'))

    if board.is_game_over():
        return jsonify({'game_over': True, 'result': board.result()})

    best_move, value = ai_engine.search(board)

    if best_move:
        win_prob = value
        if board.turn == chess.BLACK:
            win_prob = -win_prob 

        return jsonify({'move': best_move, 'win_prob': win_prob})
    
    return jsonify({'error': 'Erro na IA'})

if __name__ == '__main__':
    app.run(debug=True)