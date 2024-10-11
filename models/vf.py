import torch
import torch.nn as nn
from dilo_utils import mlp

class TwinQ(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=256, n_hidden=2, layer_norm=False):
        super().__init__()
        dims = [state_dim+act_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True, layer_norm=layer_norm)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2, layer_norm=False):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True, layer_norm=layer_norm)

    def forward(self, state):
        return self.v(state)


class TwinV(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2, layer_norm=False):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = mlp(dims, squeeze_output=True, layer_norm=layer_norm)
        self.v2 = mlp(dims, squeeze_output=True, layer_norm=layer_norm)


    def both(self, state):
        return self.v1(state), self.v2(state)

    def forward(self, state):
        return torch.min(*self.both(state))
