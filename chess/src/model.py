import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual 
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=5): 
        super(ChessNet, self).__init__()
        
        # --- TEM DE SER 12 AQUI ---
        self.conv_input = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        # --------------------------
        
        self.bn_input = nn.BatchNorm2d(64)
        
        self.res_tower = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )
        
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096) 

        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_tower(x)
        
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 8 * 8)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1) 
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) 
        
        return p, v