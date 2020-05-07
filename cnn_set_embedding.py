# Embeds a set of words using a convolutional architecture

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from alphabet import AsciiEmbeddedEncoding
from torch.nn.utils.rnn import pad_sequence

class CNNSetEmbedding(nn.Module):
    def __init__(self, device, output_size=50, kernel_size=5):
        super().__init__()

        self.alphabet = AsciiEmbeddedEncoding(device)
        self.words_conv = nn.Conv1d(self.alphabet.embedding_size(), output_size,
                              kernel_size, padding=(kernel_size // 2))
        self.word_gate = nn.Linear(output_size, output_size)
        self.word_proj = nn.Linear(output_size, output_size)

        self.set_conv = nn.Conv1d(output_size, output_size, kernel_size=1, padding=0)
        self.set_gate = nn.Linear(output_size, output_size)
        self.set_proj = nn.Linear(output_size, output_size)

    def forward(self, word_sets):
        word_sets = [['^'] + s for s in word_sets]
        set_lens = list(map(len, word_sets))
        set_begin = np.cumsum([0] + set_lens)

        encoded_chars = self.alphabet.encode_batch([w for s in word_sets for w in s])
        max_len = encoded_chars.shape[1]

        words, _ = self.words_conv(encoded_chars.transpose(1, 2)).relu().max(dim=2)
        words_proj = self.word_proj(words).relu()
        words_gate = self.word_gate(words).sigmoid()
        words_out = words_gate * words_proj + (1 - words_gate) * words

        word_sets_embedded = [words_out[set_begin[i]:set_begin[i+1]] for i in range(len(word_sets))]
        word_sets_padded = pad_sequence(word_sets_embedded, batch_first=True)

        sets, _ = self.set_conv(word_sets_padded.transpose(1, 2)).relu().max(dim=2)
        sets_proj = self.set_proj(sets).relu()
        sets_gate = self.set_gate(sets).sigmoid()
        sets_out = sets_gate * sets_proj + (1 - sets_gate) * sets

        return sets_out
