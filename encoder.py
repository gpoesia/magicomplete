from base import AutoCompleteEncoder
import random
import torch
import torch.nn as nn

class NeuralEncoder(nn.Module):
    'Encodes a string by removing characters.'
    def __init__(self, alphabet, epsilon=None, hidden_size=100):
        super().__init__()

        self.hidden_size = hidden_size
        self.encoder_lstm = nn.LSTM(alphabet.embedding_size(), hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, 1)
        self.epsilon = epsilon

    def name(self):
        return 'NeuralEncoder(eps={:.2f})'.format(self.epsilon)

    def encode(self, batch):
        encoded_batch = self.alphabet.encode_batch(batch)
        encoded_batch_indices = self.alphabet.encode_batch_indices(batch)

        p_keep = self(encoded_batch, encoded_batch_indices)

        mask = torch.bernoulli(p_keep)
        return [''.join([batch[i][j]
                for j in range(len(batch[i]))
                if mask[i][j+1]])
                for i in range(len(batch))]

    def forward(self, encoded_batch, alphabet, encoded_batch_indices):

        encoder_hidden_states, final_state = self.encoder_lstm(encoded_batch) #(B,L,H), H
        p_keep = torch.sigmoid(self.output_proj(encoder_hidden_states)).squeeze(2) # (B,L,1)

        p_keep = p_keep.masked_fill(encoded_batch_indices == alphabet.start_token_index(), 1.0)
        p_keep = p_keep.masked_fill(encoded_batch_indices == alphabet.end_token_index(), 1.0)

        return p_keep

    def is_optimizeable(self):
        return True
