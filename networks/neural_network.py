import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstracts import BaseNetwork


class GymEnvModel(BaseNetwork):
    def __init__(self, num_state=8, num_action=4, discrete_action=True, gru=True):
        super(GymEnvModel, self).__init__()
        self.num_action = num_action
        self.use_gru = gru
        self.discrete_action = discrete_action
        self.hidden_dimension = 64
        # nn model --------------------------------------------------------------------------
        # adapted from the model of EPyMARL: https://github.com/uoe-agents/epymarl/blob/main/src/modules/agents/rnn_agent.py
        self.fc1 = nn.Linear(num_state, self.hidden_dimension) # Applies a linear transformation to the incoming data
        self.h = self.init_hidden() # hidden state
        if self.use_gru: # else: fc
            self.gru = nn.GRU(self.hidden_dimension, self.hidden_dimension) # multi-layer gated recurrent unit 
        else:
            self.fc2 = nn.Linear(self.hidden_dimension, self.hidden_dimension) 
        self.fc3 = nn.Linear(self.hidden_dimension, num_action)
        # -----------------------------------------------------------------------------------     

    def init_hidden(self):
        return torch.zeros([1, 1, 64], dtype=torch.float) 

    def forward(self, x, debug=False):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)
            
            x = self.fc1(x) # in
            x = F.relu(x) 
            if self.use_gru:
                x, self.h = self.gru(x, self.h) # multi-layer
            else:
                x = self.fc2(x)
                x = F.relu(x) 
            x = self.fc3(x) # out

            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)
            else:
                x = torch.tanh(x.squeeze())
            x = x.detach().cpu().numpy()

            return x

    def reset(self):
        if self.use_gru:
            self.h = self.init_hidden()

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def get_param_list(self):
        param_list = []
        for param in self.parameters():
            param_list.append(param.data.numpy())
        return param_list

    def apply_param(self, param_lst: list):
        count = 0
        for p in self.parameters():
            p.data = torch.tensor(param_lst[count]).float()
            count += 1
