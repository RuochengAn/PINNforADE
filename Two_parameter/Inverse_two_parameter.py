'''

Advection-Diffusion Equation with Thermal Effects (Dimensionless Form):
    Rd * u_t = u_zz + (M - Pe) * u_z
    
Initial Condition (No pollution):
    u(0, z) = 0
    
Top boundary condition (Dirichlet, Time-dependent boundary):
    u(t, 0) = 1 - exp(-alpha * t)
    
Bottom boundary conditons:
    Case 1: Dirichlet, zero-concentration boundary
        u = 0
    Case 2: Neumann, zero-flux boundary
        u_z = 0
    Case 3: Robin, mixed boundary
        u_z + beta * u = 0

'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self._init_weights()
        self.Rd = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def _init_weights(self):
        for m in self.nn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        NN = self.nn(torch.cat([t, z], dim=1))
        return NN

class AdaptiveTanh(nn.Module):
    def __init__(self, init_a=1.0):
        super(AdaptiveTanh, self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a))

    def forward(self, x):
        return torch.tanh(self.a * x)

class SAPINN(nn.Module):
    def __init__(self):
        super(SAPINN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(2, 64), AdaptiveTanh(init_a=1.0),
            nn.Linear(64, 64), AdaptiveTanh(init_a=1.0),
            nn.Linear(64, 64), AdaptiveTanh(init_a=1.0),
            nn.Linear(64, 1)
        )
        self._init_weights()
        self.Rd = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def _init_weights(self):
        for m in self.nn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        NN = self.nn(torch.cat([t, z], dim=1))
        D = t
        return NN * D