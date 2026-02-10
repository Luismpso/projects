import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ChessDataset
from src.model import ChessNet
import os
import glob
import chess
import chess.pgn
import io
from tqdm import tqdm

# Configurações gerais
BATCH_SIZE = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 

# Ficheiros de dados
LICHESS_FILE = "data/lichess_db_standard_rated_2019-07.pgn"
OPENINGS_DIR = "data/chess-openings"

MAX_ELITE_GAMES = 50000 
MIN_RATING = 2000

# Hiperparâmetros de treino
EPOCHS = 10
LEARNING_RATE = 0.001

def load_tsv_file(file_path, max_lines=5000):
    """ Lê ficheiros .tsv de aberturas (jogadas de livro). """
    games_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines and "ECO" in lines[0]: lines = lines[1:]
            
        count = 0
        for line in lines:
            if count >= max_lines: break
            parts = line.strip().split('\t')
            if len(parts) < 3: continue 
            
            pgn_string = parts[-1]
            try:
                game = chess.pgn.read_game(io.StringIO(pgn_string))
                if game is None: continue
                
                board = game.board()
                for move in game.mainline_moves():
                    games_data.append((board.fen(), move.uci(), 0.0)) 
                    board.push(move)
                count += 1
            except: continue
                
    return games_data

def load_pgn_turbo(file_path, target_games, min_rating):
    """
    Modo turbo: Lê o ficheiro como texto simples para encontrar jogos rápidos.
    Só processa o xadrez se o rating for bom.
    """
    games_data = []
    games_found = 0
    
    print(f"🚀 Modo turbo ativado: A filtrar {os.path.basename(file_path)}...")
    print(f"🎯 Objetivo: {target_games} jogos com Rating > {min_rating}")
    
    # Variáveis temporárias
    current_game_lines = []
    white_elo = 0
    black_elo = 0
    
    # Abre o ficheiro como texto
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(total=target_games, unit="jogos", desc="Jogos Elite")
        
        for line in f:
            # Se encontrarmos o início de um novo jogo...
            if line.startswith("[Event"):
                # ...processamos o jogo ANTERIOR (se ele existir e for bom)
                if current_game_lines and white_elo >= min_rating and black_elo >= min_rating:
                    try:
                        # Transforma o texto acumulado num jogo real
                        pgn_str = "".join(current_game_lines)
                        game = chess.pgn.read_game(io.StringIO(pgn_str))
                        
                        if game:
                            # Vê quem ganhou
                            result = game.headers.get("Result", "*")
                            if result == "1-0": val = 1.0
                            elif result == "0-1": val = -1.0
                            else: val = 0.0
                            
                            # Extrai as posições
                            board = game.board()
                            for move in game.mainline_moves():
                                games_data.append((board.fen(), move.uci(), val))
                                board.push(move)
                            
                            games_found += 1
                            pbar.update(1)
                            
                            if games_found >= target_games:
                                break
                    except:
                        pass # Se der erro num jogo, ignora e segue
                
                # Reseta as variáveis para começar a ler o NOVO jogo
                current_game_lines = [line]
                white_elo = 0
                black_elo = 0
            
            else:
                # Estamos a ler linhas de um jogo. Guardamos no buffer.
                current_game_lines.append(line)
                
                # Tenta pescar o Rating rapidinho
                if line.startswith("[WhiteElo"):
                    try: white_elo = int(line.split('"')[1])
                    except: pass
                elif line.startswith("[BlackElo"):
                    try: black_elo = int(line.split('"')[1])
                    except: pass

        pbar.close()

    return games_data

def train():
    print(f"🚀 A iniciar Sistema de Treino em: {DEVICE}")
    print(f"🚀 Placa Gráfica: {torch.cuda.get_device_name(0)}")
    print("--------------------------------------------------")
    
    raw_data = []

    # 1. Carregar teoria de aberturas
    if os.path.exists(OPENINGS_DIR):
        print(f"📂 A carregar teoria de aberturas...")
        opening_files = glob.glob(os.path.join(OPENINGS_DIR, "*.tsv"))
        for tsv_path in opening_files:
            games = load_tsv_file(tsv_path)
            raw_data.extend(games)
    
    # 2. Carregar jogos de alta qualidade do Lichess
    if os.path.exists(LICHESS_FILE):
        lichess_games = load_pgn_turbo(LICHESS_FILE, MAX_ELITE_GAMES, MIN_RATING)
        raw_data.extend(lichess_games)
    else:
        print(f"⚠️ Erro: Não encontrei {LICHESS_FILE}")

    # 3. Preparar Dataset
    total_positions = len(raw_data)
    print("--------------------------------------------------")
    print(f"📊 Dataset final: {total_positions} posições de xadrez de alta qualidade.")
    
    if total_positions == 0:
        print("❌ Erro: Sem dados. A abortar.")
        return
    
    dataset = ChessDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # 4. Modelo e Otimizador
    model = ChessNet().to(DEVICE)
    if os.path.exists("models/supervised.pth"):
        print("🔄 A continuar treino do modelo existente...")
        try:
            model.load_state_dict(torch.load("models/supervised.pth", map_location=DEVICE))
        except: print("   -> Modelo incompatível, a criar novo.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # 5. Loop de treino
    print("--------------------------------------------------")
    print("🏋️  A treinar...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Época {epoch+1}/{EPOCHS}", leave=True)
        
        for boards, moves, results in loop:
            boards, moves, results = boards.to(DEVICE), moves.to(DEVICE), results.to(DEVICE)
            
            optimizer.zero_grad()
            pred_policy, pred_value = model(boards)
            
            loss_p = policy_loss_fn(pred_policy, moves)
            loss_v = value_loss_fn(pred_value.squeeze(), results) 
            
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Época {epoch+1} concluída. Erro médio: {avg_loss:.4f}")

    # 6. Salvar modelo final
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/supervised.pth")
    print("\n💾 Treino concluído! Modelo salvo em models/supervised.pth")
    print("👉 Agora podes correr 'python app.py' para jogar!")

if __name__ == "__main__":
    train()