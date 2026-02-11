import time
import chess
import torch
import os
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from src.model import ChessNet
from src.mcts import MCTS
from src.dataset import decode_move

# --- CONFIGURAÇÕES ---
MODEL_PATH = "models/reinforcement.pth" # Ou models/supervised.pth se quiseres testar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMULATIONS = 50 # Mantém baixo para ser rápido em Python

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_board_state(driver):
    board = chess.Board()
    board.clear()
    
    try:
        # Script JS otimizado
        data = driver.execute_script("""
            const pieces = [];
            document.querySelectorAll('.piece').forEach(p => pieces.push(p.className));
            const highlights = [];
            document.querySelectorAll('.highlight').forEach(h => highlights.push(h.className));
            return {pieces: pieces, highlights: highlights};
        """)

        if not data: return None, None

        # 1. Montar Peças
        for p_class in data['pieces']:
            color_match = re.search(r'([wb])([pnbrqk])', p_class)
            sq_match = re.search(r'square-(\d)(\d)', p_class)
            if color_match and sq_match:
                color = chess.WHITE if color_match.group(1) == 'w' else chess.BLACK
                role = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6}[color_match.group(2)]
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                board.set_piece_at(chess.square(f, r), chess.Piece(role, color))

        # 2. Descobrir Vez (Pelo highlight amarelo)
        turn = chess.WHITE # Default
        for h_class in data['highlights']:
            sq_match = re.search(r'square-(\d)(\d)', h_class)
            if sq_match:
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                piece = board.piece_at(chess.square(f, r))
                if piece:
                    # Se há peça no highlight, foi a que mexeu. A vez é do outro.
                    turn = chess.BLACK if piece.color == chess.WHITE else chess.WHITE
                    break
        
        board.turn = turn
        return board, turn
    except:
        return None, None

def main():
    print("🚀 Espião V4 (Final)...")
    options = webdriver.ChromeOptions()
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    print(f"🧠 Carregando IA ({DEVICE})...")
    model = ChessNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except:
        print("❌ Erro: Modelo não encontrado.")
        return

    driver.get("https://www.chess.com/play/computer")
    print("✅ Vai ao browser e joga!")
    
    last_key = ""

    while True:
        try:
            board, turn = get_board_state(driver)
            
            if board:
                # Chave única do estado atual
                key = f"{board.epd().split(' ')[0]} {turn}"
                
                if key != last_key:
                    last_key = key
                    clear_screen()
                    who = "BRANCAS" if turn == chess.WHITE else "PRETAS"
                    print(f"📡 LIGADO | Vez: {who}")
                    print(board) 
                    print(f"\n🧠 A pensar ({SIMULATIONS} simulações)...")
                    
                    # MCTS
                    mcts = MCTS(model, DEVICE, SIMULATIONS)
                    board.turn = turn # Garantir a vez correta no objeto board
                    root = mcts.search_return_root(board)
                    
                    # Processar resultados
                    suggestions = []
                    total = sum(c.visit_count for c in root.children.values())
                    
                    if total > 0:
                        for idx, child in root.children.items():
                            move = decode_move(idx)
                            if move and chess.Move.from_uci(move) in board.legal_moves:
                                # Score do ponto de vista do jogador atual
                                score = child.value_sum / child.visit_count
                                conf = (child.visit_count / total) * 100
                                suggestions.append((move, conf, score))
                    
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    
                    print("\n🔥 SUGESTÕES:")
                    for i, (m, c, s) in enumerate(suggestions[:3]):
                        # Interpretação do Score (-1 a 1)
                        eval_txt = f"{s:.2f}"
                        if s > 0.5: eval_txt += " (Ganho)"
                        elif s < -0.5: eval_txt += " (Perdido)"
                        
                        icon = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
                        print(f"{icon} {m} -> {c:.1f}% [{eval_txt}]")
                    
                    print("-" * 30)

            time.sleep(0.5)
        except KeyboardInterrupt:
            break
        except Exception:
            pass

if __name__ == "__main__":
    main()