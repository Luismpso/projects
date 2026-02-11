import time
import chess
import torch
import os
import re
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from src.model import ChessNet
from src.mcts import MCTS
from src.dataset import decode_move

# --- CONFIGURAÇÕES ---
MODEL_PATH = "models/supervised.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REDUZI PARA 50 (Para ser mais rápido!)
SIMULATIONS = 50 

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model():
    print(f"🧠 A preparar IA no dispositivo: {DEVICE}...")
    model = ChessNet().to(DEVICE) # Certifica-te que o model.py está com 12 canais

    if not os.path.exists(MODEL_PATH):
        print(f"❌ O ficheiro '{MODEL_PATH}' não existe.")
        sys.exit(1)

    try:
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Cérebro carregado!")
        return model
    except Exception as e:
        print(f"\n❌ Erro ao carregar: {e}")
        print("Verifica se o model.py tem o mesmo nº de canais que o treino (12 vs 17).")
        sys.exit(1)

def get_board_state(driver):
    board = chess.Board()
    board.clear()
    
    try:
        data = driver.execute_script("""
            const pieces = [];
            document.querySelectorAll('.piece').forEach(p => pieces.push(p.className));
            const highlights = [];
            document.querySelectorAll('.highlight').forEach(h => highlights.push(h.className));
            return {pieces: pieces, highlights: highlights};
        """)

        if not data: return None, None

        for p_class in data['pieces']:
            color_match = re.search(r'([wb])([pnbrqk])', p_class)
            sq_match = re.search(r'square-(\d)(\d)', p_class)
            if color_match and sq_match:
                color = chess.WHITE if color_match.group(1) == 'w' else chess.BLACK
                role = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6}[color_match.group(2)]
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                board.set_piece_at(chess.square(f, r), chess.Piece(role, color))

        turn = chess.WHITE 
        for h_class in data['highlights']:
            sq_match = re.search(r'square-(\d)(\d)', h_class)
            if sq_match:
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                piece = board.piece_at(chess.square(f, r))
                if piece:
                    turn = chess.BLACK if piece.color == chess.WHITE else chess.WHITE
                    break
        
        board.turn = turn
        return board, turn
    except Exception:
        return None, None

def main():
    clear_screen()
    print("🚀 Espião V6 (Rápido)...")
    model = load_model()

    print("🌐 A abrir Browser...")
    options = webdriver.ChromeOptions()
    options.add_argument("--log-level=3") 
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://www.chess.com/play/computer")
    
    print("✅ Pronto.")
    
    last_key = ""

    while True:
        try:
            board, turn = get_board_state(driver)
            
            if board:
                key = f"{board.epd().split(' ')[0]} {turn}"
                
                if key != last_key:
                    last_key = key
                    clear_screen()
                    
                    who = "BRANCAS" if turn == chess.WHITE else "PRETAS"
                    print(f"🎮 VEZ: {who}")
                    print(f"⏳ A calcular ({SIMULATIONS} sims)...")
                    
                    mcts = MCTS(model, DEVICE, SIMULATIONS)
                    board.turn = turn 
                    root = mcts.search_return_root(board)
                    
                    suggestions = []
                    total_visits = sum(c.visit_count for c in root.children.values())
                    
                    if total_visits > 0:
                        for idx, child in root.children.items():
                            move = decode_move(idx, board)
                            if move:
                                try:
                                    if chess.Move.from_uci(move) in board.legal_moves:
                                        score = child.value_sum / child.visit_count
                                        conf = (child.visit_count / total_visits) * 100
                                        suggestions.append((move, conf, score))
                                except: continue
                    
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    
                    # --- MOSTRAR APENAS A MELHOR JOGADA ---
                    print("\n" + "="*30)
                    if suggestions:
                        best_move, conf, score = suggestions[0]
                        
                        # Texto de avaliação
                        status = "Neutro"
                        if score > 0.3: status = "Vantagem"
                        elif score < -0.3: status = "Cuidado"
                        
                        print(f"🔥 JOGAR:  {best_move}")
                        print(f"📊 Confiança: {conf:.1f}%")
                        print(f"📈 Estado:    {status} ({score:.2f})")
                    else:
                        print("⚠️ Sem sugestão óbvia.")
                    print("="*30)
                    # --------------------------------------

            time.sleep(0.5)
            
        except KeyboardInterrupt:
            break
        except Exception:
            pass

if __name__ == "__main__":
    main()