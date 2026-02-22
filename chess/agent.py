import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import ChessDQN

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        # Rede que toma as ações
        self.policy_net = ChessDQN().to(self.device)
        # Rede Alvo (Target) estabiliza o treino (atualizada a cada N passos)
        self.target_net = ChessDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4) # Learning rate modesto
        self.memory = ReplayBuffer(10000) # Capacidade do buffer (quantas jogadas passadas recorda para treinar)
        self.batch_size = 64
        self.gamma = 0.99
        
    def select_action(self, state, legal_actions, epsilon):
        if not legal_actions:
            return None
            
        if random.random() < epsilon:
            # Explorar
            return random.choice(legal_actions)[0]
            
        # Exploração Avarenta (Greedy)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Retorna o Q-value previso para todas as 4096 ações possíveis
            q_values = self.policy_net(state_tensor).squeeze(0) 
            
            best_action = None
            best_q = -float('inf')
            
            # Masking the actions: Procuramos a melhor ação, APENAS dentro das válidas
            for action_idx, move in legal_actions:
                q = q_values[action_idx].item()
                if q > best_q:
                    best_q = q
                    best_action = action_idx
                    
            return best_action
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device) 
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calcular os Q(s, a) atuais
        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Calcular os max Q(s', a') usando a rede alvo
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            
        # Q-values esperados
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
        
        # Função de perda (Huber Loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        # Otimizar (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clipping para limitar a "explosão" dos gradientes na aprendizagem
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        return loss.item()
