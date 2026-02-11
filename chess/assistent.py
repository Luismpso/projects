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

# --- CONFIGURAÇÕES ---
MODEL_PATH = "models/supervised.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMULATIONS = 400 # Igual ao app.py (ou ajusta para 50 se quiseres voar)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model():
    print(f"🧠 A preparar IA no dispositivo: {DEVICE}...")
    # Tenta criar com 12 canais (modelo antigo)
    try:
        model = ChessNet().to(DEVICE) 
        # Força o carregamento ignorando erros de pickle
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        print("✅ Modelo carregado (Modo 12 canais)!")
        return model
    except:
        print("⚠️ Erro no carregamento padrão. A tentar hacks...")
        # Se falhar, tenta reconstruir a rede (caso tenhas mudado o model.py recentemente)
        try:
            from src.model import ChessNet # Recarrega
            model = ChessNet().to(DEVICE)
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            return model
        except Exception as e:
            print(f"❌ Falha total: {e}")
            sys.exit(1)

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

        # Montar Tabuleiro
        for p_class in data['pieces']:
            color_match = re.search(r'([wb])([pnbrqk])', p_class)
            sq_match = re.search(r'square-(\d)(\d)', p_class)
            if color_match and sq_match:
                color = chess.WHITE if color_match.group(1) == 'w' else chess.BLACK
                role = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6}[color_match.group(2)]
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                board.set_piece_at(chess.square(f, r), chess.Piece(role, color))

        # Descobrir a vez
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
    except:
        return None, None

def main():
    clear_screen()
    print("🚀 Espião V7 (Modo App.py)...")
    model = load_model()

    print("🌐 A abrir Browser...")
    options = webdriver.ChromeOptions()
    options.add_argument("--log-level=3") 
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://www.chess.com/play/computer")
    print("✅ Browser aberto. NÃO MINIMIZES A JANELA!")
    
    last_key = ""

    while True:
        try:
            # 1. Leitura do Tabuleiro
            board, turn = get_board_state(driver)
            
            if board:
                key = f"{board.epd().split(' ')[0]} {turn}"
                
                # Só calcula se o tabuleiro mudou
                if key != last_key:
                    last_key = key
                    clear_screen()
                    
                    who = "BRANCAS" if turn == chess.WHITE else "PRETAS"
                    print(f"🎮 VEZ: {who}")
                    print(board)
                    print(f"\n⚡ A calcular (Modo Rápido)...")
                    
                    # 2. O Segredo da Velocidade: Usar .search() em vez de .search_return_root()
                    # Isto é exatamente o mesmo código que o app.py usa.
                    mcts = MCTS(model, DEVICE, SIMULATIONS)
                    board.turn = turn 
                    
                    start_time = time.time()
                    best_move, val = mcts.search(board) # <--- AQUI ESTÁ A MUDANÇA
                    elapsed = time.time() - start_time
                    
                    # Ajuste de perspectiva (se for Preto, inverte o valor)
                    if turn == chess.BLACK: val = -val
                    
                    status = "Neutro"
                    if val > 0.5: status = "Vantagem Brancas"
                    elif val < -0.5: status = "Vantagem Pretas"

                    print("\n" + "="*30)
                    print(f"🔥 MELHOR JOGADA: {best_move}")
                    print(f"⏱️ Tempo: {elapsed:.2f}s")
                    print(f"📈 Valor: {val:.2f} ({status})")
                    print("="*30)

            time.sleep(0.1) # Loop mais rápido
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Se der erro, imprime mas não crasha
            print(f"Erro no loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()