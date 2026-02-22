import torch
from env import ChessEnv
from agent import DQNAgent

def play_game():
    env = ChessEnv()
    agent = DQNAgent(device="cpu")
    
    try:
        agent.policy_net.load_state_dict(torch.load("chess_dqn.pth", map_location="cpu"))
        print("Modelo treinado carregado com sucesso.")
    except Exception as e:
        print("Modelo 'chess_dqn.pth' não encontrado. O agente vai jogar com pesos aleatórios (não treinado).")
        
    state, legal_actions = env.reset()
    done = False
    
    print("-" * 30)
    print("Bem-vindo ao Xadrez RL!")
    print("0 - O Agente joga com as Brancas")
    print("1 - Tu jogas com as Brancas")
    
    try:
        choice = int(input("Escolhe (0 ou 1): "))
    except ValueError:
        choice = 0
        
    user_turn = (choice == 1)
    
    while not done:
        env.render()
        
        if user_turn:
            print("\nA tua vez!")
            valid_moves = [m.uci() for _, m in legal_actions]
            print(f"Algumas jogadas legais: {valid_moves[:5]} ... (total {len(valid_moves)})")
            
            valid = False
            while not valid:
                move_uci = input("Insere a tua jogada (formato e2e4): ")
                
                action = None
                for idx, m in legal_actions:
                    if m.uci() == move_uci:
                        action = idx
                        valid = True
                        break
                        
                if not valid:
                    print("Jogada inválida! Tenta novamente.")
                    
            state, reward, done, info = env.step(action)
            legal_actions = info["legal_actions"]
            user_turn = False
            
        else:
            print("\nO Agente está a pensar...")
            action = agent.select_action(state, legal_actions, epsilon=0.0)
            
            chosen_m = None
            for idx, m in legal_actions:
                if idx == action:
                    chosen_m = m
                    break
                    
            if chosen_m:
                print(f"O Agente jogou: {chosen_m.uci()}")
            else:
                # Should only happen if game is already over
                print("O Agente não tem jogadas possíveis.")
                break
                
            state, reward, done, info = env.step(action)
            legal_actions = info["legal_actions"]
            user_turn = True
            
    env.render()
    print("Fim do Jogo!")
    print("Resultado Final:", env.board.result())

if __name__ == "__main__":
    play_game()
