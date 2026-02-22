import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessDQN(nn.Module):
    def __init__(self):
        super(ChessDQN, self).__init__()
        # Entrada: 12 canais (6 peças do agente, 6 do adversário) x 8 x 8
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # O tabuleiro é 8x8. Depois de 3 convs com padding=1, continua com 8x8.
        # 128 (canais) * 8 * 8 = 8192 atributos na matriz flattened
        self.fc1 = nn.Linear(8192, 1024)
        
        # Saída: 4096 ações possíveis (x * 64 + y, onde max_x=63 e max_y=63)
        self.fc2 = nn.Linear(1024, 4096)
        
    def forward(self, x):
        # x tem a forma (batch_size, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten para preparar para as Linear Layers
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        # Sem ReLU aqui porque os Q-values podem e devem ser negativos ou positivos
        x = self.fc2(x)
        return x
