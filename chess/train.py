import torch
import random
from env import ChessEnv
from agent import DQNAgent

def train_agent(episodes=500):
    env = ChessEnv()
    agent = DQNAgent(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar usando o dispositivo: {agent.device}")
    
    # Parâmetros Epsilon-Greedy
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    for episode in range(episodes):
        state, legal_actions = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # TURNO DO AGENTE
            action = agent.select_action(state, legal_actions, epsilon)
            
            # Aplica a ação
            next_state, reward_agent, done, info = env.step(action)
            legal_actions = info["legal_actions"]
            
            if done:
                reward = reward_agent
                agent.store_transition(state, action, reward, next_state, done)
                # Otimização final caso perca/ganhe nesse turno
                agent.optimize_model()
                break
                
            # TURNO DO OPONENTE (Joga aleatoriamente para fins de treino inicial)
            opp_action = random.choice(legal_actions)[0]
            next_state, reward_opp, done, info = env.step(opp_action)
            legal_actions = info["legal_actions"]
            
            # A recompensa total da "maratona" dos dois turnos 
            # (Vantagem material do Agente MENOS a vantagem alcançada pelo oponente)
            reward = reward_agent - reward_opp
            
            # Armazena a transição do estado inicial do agente PARA o estado do agente
            # após a resposta do oponente
            agent.store_transition(state, action, reward, next_state, done)
            
            # Otimizar pesos da rede neural
            loss = agent.optimize_model()
            
            state = next_state
            total_reward += reward
            step += 1
            
        # Atualizar a Rede Alvo a cada 10 episódios
        if episode % 10 == 0:
            agent.update_target_network()
            
        # Reduzir Epsilon (explorar menos, explorar o conhecimento mais)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            print(f"Episódio {episode}: Passos={step}, Recompensa Material={total_reward}, Epsilon={epsilon:.3f}")
            
    # Guardar os pesos finais (o cérebro do agente)
    torch.save(agent.policy_net.state_dict(), "chess_dqn.pth")
    print("Treino concluído. Modelo guardado.")

if __name__ == "__main__":
    train_agent()
