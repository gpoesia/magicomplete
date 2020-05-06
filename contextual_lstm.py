# LSTM Cell that takes a context tensor into account
# Idea from https://arxiv.org/pdf/1804.09661.pdf

import torch
from torch import nn

class ContextualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, context_size, context_rank):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wi = nn.Parameter(torch.Tensor(input_size, 4*hidden_size))
        self.Wi_b = nn.Parameter(torch.Tensor(4*hidden_size))

        self.Wh = nn.Parameter(torch.Tensor(hidden_size, 4*hidden_size))
        self.Wh_b = nn.Parameter(torch.Tensor(4*hidden_size))

        self.AZ_il = nn.Parameter(torch.Tensor(context_size,
                                               input_size,
                                               context_rank))
        self.AZ_ir = nn.Parameter(torch.Tensor(context_rank,
                                               4*hidden_size,
                                               context_size))

        self.AZ_hl = nn.Parameter(torch.Tensor(context_size,
                                               hidden_size,
                                               context_rank))
        self.AZ_hr = nn.Parameter(torch.Tensor(context_rank,
                                               4*hidden_size,
                                               context_size))

    def contextual_parameters(self):
        return iter((self.AZ_il, self.AZ_ir, self.AZ_hl, self.AZ_hr))

    def compute_context_weights(self, context):
        lhs_i = torch.tensordot(context, self.AZ_il, dims=([1], [0]))
        rhs_i = torch.tensordot(context, self.AZ_ir, dims=([1], [2]))

        A_i = torch.bmm(lhs_i, rhs_i)

        lhs_h = torch.tensordot(context, self.AZ_hl, dims=([1], [0]))
        rhs_h = torch.tensordot(context, self.AZ_hr, dims=([1], [2]))
        A_h = torch.bmm(lhs_h, rhs_h)

        return (A_i, A_h)

    def forward(self, x, state=None, context_weights=None):
        B, E = x.shape
        H = self.hidden_size

        if E != self.input_size:
            raise ValueError('Expected input size of {}, got {}'
                             .format(self.input_size, E))

        if state is None:
            h = torch.zeros((B, H), device=x.device, dtype=x.dtype,
                            requires_grad=True)
            c = torch.zeros((B, H), device=x.device, dtype=x.dtype,
                            requires_grad=True)
        else:
            h, c = state

        if context_weights is None:
            A_i1 = torch.zeros_like(self.Wi, requires_grad=True)
            A_h1 = torch.zeros_like(self.Wh, requires_grad=True)
            A_i = A_i1.expand((B, E, 4*H))
            A_h = A_h1.expand((B, H, 4*H))
        else:
            A_i, A_h = context_weights

        Wi = 0*A_i + self.Wi
        Wh = 0*A_h + self.Wh

        ifgo = (
                torch.bmm(x.unsqueeze(1), Wi).squeeze(1) + self.Wi_b +
                torch.bmm(h.unsqueeze(1), Wh).squeeze(1) + self.Wh_b
               )

        i, f, g, o = ifgo[:, :H], ifgo[:, H:2*H], ifgo[:, 2*H:3*H], ifgo[:, 3*H:]
        i = i.sigmoid()
        f = f.sigmoid()
        g = g.tanh()
        o = o.sigmoid()

        c_new = f * c + i * g
        h_new = o * c_new.tanh()

        return (h_new, c_new)
