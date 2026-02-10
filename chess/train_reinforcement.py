import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import os
import time
from src.model import ChessNet
from src.mcts import MCTS
from src.board_encoding import encode_board
from src.dataset import encode_move
from tqdm import tqdm  

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ficheiros
SUPERVISED_MODEL = "models/supervised.pth" 
RL_MODEL_PATH = "models/reinforcement.pth"

# Hiperparâmetros
GENERATIONS = 10         # Quantas vezes a IA evolui
GAMES_PER_GEN = 100       # Jogos por geração
MCTS_SIMS = 100       # Simulações por jogada
EPOCHS_PER_GEN = 5       
BATCH_SIZE = 16

class RLDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        board, policy, value = self.examples[idx]
        return (
            torch.tensor(board, dtype=torch.float32), 
            torch.tensor(policy, dtype=torch.float32), 
            torch.tensor(value, dtype=torch.float32)
        )

def execute_episode(model):
    examples = []
    board = chess.Board()
    mcts = MCTS(model, DEVICE, num_simulations=MCTS_SIMS)
    
    while not board.is_game_over():
        root = mcts.search_return_root(board)
        
        policy_target = np.zeros(4096)
        visit_sum = sum(child.visit_count for child in root.children.values())
        
        if visit_sum > 0:
            for move_idx, child in root.children.items():
                policy_target[move_idx] = child.visit_count / visit_sum
        
        examples.append([encode_board(board), policy_target, 0])
        
        best_move_idx = -1
        max_v = -1
        for idx, child in root.children.items():
            if child.visit_count > max_v:
                max_v = child.visit_count
                best_move_idx = idx
        
        move_to_make = None
        for move in board.legal_moves:
            if encode_move(move.uci()) == best_move_idx:
                move_to_make = move
                break
        
        if move_to_make is None:
            move_to_make = list(board.legal_moves)[0]

        board.push(move_to_make)

    winner = 0.0
    result = board.result()
    if result == "1-0": winner = 1.0
    elif result == "0-1": winner = -1.0
    
    final_examples = []
    for board_state, policy, _ in examples:
        final_examples.append((board_state, policy, winner))
        
    return final_examples

def train_generation(model, examples):
    dataset = RLDataset(examples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_loss_fn = nn.MSELoss() 
    value_loss_fn = nn.MSELoss()
    
    model.train()
    total_loss = 0
    
    # Barra de progresso para o treino da rede
    for epoch in range(EPOCHS_PER_GEN):
        for boards, policies, values in dataloader:
            boards, policies, values = boards.to(DEVICE), policies.to(DEVICE), values.to(DEVICE)
            
            optimizer.zero_grad()
            pred_policy, pred_value = model(boards)
            
            pred_probs = torch.exp(pred_policy)
            loss_p = policy_loss_fn(pred_probs, policies)
            loss_v = value_loss_fn(pred_value.squeeze(), values)
            
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / (len(dataloader) * EPOCHS_PER_GEN)

def main():
    print(f"🤖 A iniciar treino em: ({DEVICE})")
    print("--------------------------------------------------")
    
    model = ChessNet().to(DEVICE)
    
    if os.path.exists(SUPERVISED_MODEL):
        print(f"✅ Carregando inteligência base: {SUPERVISED_MODEL}")
        try:
            model.load_state_dict(torch.load(SUPERVISED_MODEL, map_location=DEVICE))
        except:
             print("⚠️ Erro ao carregar. Verifica se o ficheiro existe.")
    elif os.path.exists(RL_MODEL_PATH):
        print(f"✅ Continuando treino anterior: {RL_MODEL_PATH}")
        model.load_state_dict(torch.load(RL_MODEL_PATH, map_location=DEVICE))

    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- 🧬 Geração {gen}/{GENERATIONS} ---")
        start_time = time.time()
        
        # A. SELF-PLAY (AGORA COM BARRA DE TEMPO!)
        print(f"   🎮 A jogar {GAMES_PER_GEN} partidas (Simulações: {MCTS_SIMS})...")
        all_examples = []
        model.eval()
        
        # O tqdm cria a barra de progresso aqui
        for i in tqdm(range(GAMES_PER_GEN), desc="Self-Play", unit="jogo"):
            game_data = execute_episode(model)
            all_examples.extend(game_data)
        
        # B. TREINO
        print(f"   🧠 A aprender com {len(all_examples)} novas posições...")
        avg_loss = train_generation(model, all_examples)
        print(f"      -> Erro de Treino: {avg_loss:.4f}")
        
        # C. SALVAR
        torch.save(model.state_dict(), RL_MODEL_PATH)
        print(f"   💾 Salvo em: {RL_MODEL_PATH}")
        
        elapsed = time.time() - start_time
        print(f"   ⏱️ Tempo da geração: {elapsed:.1f}s")

    print("\n--------------------------------------------------")
    print("🏆 Treino Concluído!")

if __name__ == "__main__":
    main()