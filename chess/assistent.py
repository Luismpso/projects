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
# Tenta carregar o supervisionado primeiro (que é o base), ou o reinforcement se preferires
MODEL_PATH = "models/supervised.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMULATIONS = 100 # Aumentei um pouco para ser mais preciso

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model():
    """Carrega o modelo com gestão de erros robusta."""
    print(f"🧠 A preparar rede neural no dispositivo: {DEVICE}...")
    model = ChessNet().to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO CRÍTICO: O ficheiro '{MODEL_PATH}' não foi encontrado!")
        print("👉 Solução: Corre 'python train_supervised.py' para criares o cérebro da IA.")
        sys.exit(1)

    try:
        # weights_only=False silencia o aviso em versões novas do PyTorch, 
        # mas mantemos compatibilidade caso uses uma versão antiga.
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            # Fallback para versões antigas do PyTorch
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Modelo '{MODEL_PATH}' carregado com sucesso!")
        return model
    except RuntimeError as e:
        print(f"\n❌ ERRO DE COMPATIBILIDADE: {e}")
        print("⚠️  Isto acontece porque mudaste a estrutura da Rede (de 12 para 17 canais).")
        print("👉 Solução: Apaga o ficheiro .pth antigo e treina um novo!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro desconhecido ao carregar: {e}")
        sys.exit(1)

def get_board_state(driver):
    board = chess.Board()
    board.clear()
    
    try:
        # Script JS para extrair peças e highlights do Chess.com
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
            # Regex para detetar cor e peça (ex: 'wp', 'bk')
            color_match = re.search(r'([wb])([pnbrqk])', p_class)
            # Regex para detetar casa (ex: 'square-11' até 'square-88')
            sq_match = re.search(r'square-(\d)(\d)', p_class)
            
            if color_match and sq_match:
                color = chess.WHITE if color_match.group(1) == 'w' else chess.BLACK
                role = {'p':1, 'n':2, 'b':3, 'r':4, 'q':5, 'k':6}[color_match.group(2)]
                # Chess.com usa coordenadas 1-8, python-chess usa 0-7
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                board.set_piece_at(chess.square(f, r), chess.Piece(role, color))

        # 2. Descobrir Vez (Pelo highlight amarelo da última jogada)
        turn = chess.WHITE # Default
        for h_class in data['highlights']:
            sq_match = re.search(r'square-(\d)(\d)', h_class)
            if sq_match:
                f, r = int(sq_match.group(1)) - 1, int(sq_match.group(2)) - 1
                piece = board.piece_at(chess.square(f, r))
                if piece:
                    # Se há uma peça numa casa iluminada, foi essa que mexeu.
                    # Logo, a vez é do adversário dessa peça.
                    turn = chess.BLACK if piece.color == chess.WHITE else chess.WHITE
                    break
        
        board.turn = turn
        return board, turn
    except Exception:
        return None, None

def main():
    clear_screen()
    print("🚀 Espião V5 (Final & Robusto)...")
    
    # Carregar IA antes de abrir o browser para garantir que está tudo bem
    model = load_model()

    print("🌐 A abrir Browser...")
    options = webdriver.ChromeOptions()
    options.add_argument("--log-level=3") # Silenciar logs do Chrome
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://www.chess.com/play/computer")
    
    print("✅ Tudo pronto! Vai jogar.")
    
    last_key = ""

    while True:
        try:
            board, turn = get_board_state(driver)
            
            if board:
                # Cria uma chave única (FEN simplificado) para não recalcular a mesma posição
                key = f"{board.epd().split(' ')[0]} {turn}"
                
                if key != last_key:
                    last_key = key
                    clear_screen()
                    
                    who = "BRANCAS" if turn == chess.WHITE else "PRETAS"
                    print(f"📡 ESTADO DETETADO | Vez: {who}")
                    print(board) 
                    print(f"\n🧠 A pensar ({SIMULATIONS} simulações)...")
                    
                    # MCTS
                    mcts = MCTS(model, DEVICE, SIMULATIONS)
                    board.turn = turn 
                    
                    # Usamos search_return_root para ter acesso às probabilidades
                    root = mcts.search_return_root(board)
                    
                    # Processar resultados para visualização
                    suggestions = []
                    total_visits = sum(c.visit_count for c in root.children.values())
                    
                    if total_visits > 0:
                        for idx, child in root.children.items():
                            # IMPORTANTE: Passar o board para o decode_move tratar promoções
                            move = decode_move(idx, board)
                            
                            if move:
                                # Verifica legalidade (às vezes a rede sugere algo ilegal)
                                try:
                                    py_move = chess.Move.from_uci(move)
                                    if py_move in board.legal_moves:
                                        score = child.value_sum / child.visit_count
                                        conf = (child.visit_count / total_visits) * 100
                                        suggestions.append((move, conf, score))
                                except:
                                    continue
                    
                    # Ordenar por visitas (confiança)
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    
                    print("\n🔥 SUGESTÕES:")
                    if not suggestions:
                        print("⚠️ Nenhuma jogada válida encontrada (Rede confusa).")
                    
                    for i, (m, c, s) in enumerate(suggestions[:3]):
                        # Formatação do score (-1 a 1)
                        eval_txt = f"{s:.2f}"
                        if s > 0.5: eval_txt += " (Vantagem)"
                        elif s < -0.5: eval_txt += " (Desvantagem)"
                        
                        icon = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
                        print(f"{icon} {m} -> {c:.1f}% [{eval_txt}]")
                    
                    print("-" * 30)

            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n👋 A sair...")
            break
        except Exception as e:
            # Ignora erros momentâneos de leitura do browser
            pass

if __name__ == "__main__":
    main()