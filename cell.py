import torch
from torch import nn

GRU = 'GRU'
LSTM = 'LSTM'

class GatedCell(nn.Module):
    def __init__(self, params):
        super().__init__()

        hidden_size = params['hidden_size']
        input_size = params['input_size']

        self.type = params['type']

        if self.type == GRU:
            self.cell = nn.GRUCell(input_size, hidden_size)
        elif self.type == LSTM:
            self.cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, input, state=None):
        return self.cell(input, state)

    def get_cell(self, state):
        if self.type == GRU:
            return state
        elif self.type == LSTM:
            return state[1]
