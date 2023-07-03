import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstracts import BaseNetwork


class GymEnvModel(BaseNetwork):
    def __init__(self, num_state=8, num_action=4, discrete_action=True, gru=True):
        super(GymEnvModel, self).__init__()
        self.num_action = num_action
        print("\nGymEnvModel:")
        print("num_state=", num_state)
        print("num_actions=", num_action)
        print("discrete=", discrete_action)
        print("gru=", gru)
        print("\n")
        self.fc1 = nn.Linear(num_state, 32) # Applies a linear transformation to the incoming data
        self.use_gru = gru
        if self.use_gru:
            self.gru = nn.GRU(32, 32)
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)
        self.fc2 = nn.Linear(32, num_action)
        self.discrete_action = discrete_action

    def forward(self, x, debug=False):
        with torch.no_grad():
            if debug:
                print("1: ", x, np.array(x).shape)
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)
            if debug:
                print("2: ", x, np.array(x).shape)
            x_ = self.fc1(x)
            if debug:
                print("3: ", x_, np.array(x_).shape)

            x = torch.tanh(x_)
            if self.use_gru:
                x, self.h = self.gru(x, self.h)
                x = torch.tanh(x)
            x = self.fc2(x)
            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)
            else:
                x = torch.tanh(x.squeeze())
            x = x.detach().cpu().numpy()
            if debug:
                print(x)
            return x

    def reset(self):
        if self.use_gru:
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)

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
