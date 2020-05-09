# Embeds a set of words using a convolutional architecture

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from alphabet import AsciiEmbeddedEncoding
from torch.nn.utils.rnn import pad_sequence
from util import broadcast_dot, Progress, batched

class CNNSetEmbedding(nn.Module):
    def __init__(self, device, output_size=50, kernel_size=5):
        super().__init__()

        self.alphabet = AsciiEmbeddedEncoding(device)
        self.words_conv = nn.Conv1d(self.alphabet.embedding_size(), output_size,
                              kernel_size, padding=(kernel_size // 2))
        self.set_conv = nn.Conv1d(output_size, output_size, kernel_size=1, padding=0)

    def embed_words(self, encoded_chars):
        words, _ = self.words_conv(encoded_chars.transpose(1, 2)).max(dim=2)
        return words

    def forward(self, word_sets):
        word_sets = [['^'] + s for s in word_sets]
        set_lens = list(map(len, word_sets))
        set_begin = np.cumsum([0] + set_lens)

        encoded_chars = self.alphabet.encode_batch([w for s in word_sets for w in s])
        words_out = self.embed_words(encoded_chars)
        word_sets_embedded = [words_out[set_begin[i]:set_begin[i+1]] for i in range(len(word_sets))]
        word_sets_padded = pad_sequence(word_sets_embedded, batch_first=True)

        sets, _ = self.set_conv(word_sets_padded.transpose(1, 2)).max(dim=2)

        return sets

    def query(self, set_embedding, words):
        encoded_chars = self.alphabet.encode_batch(words)
        word_embeddings = self.embed_words(encoded_chars)
        return broadcast_dot(word_embeddings, set_embedding).sigmoid()

    def train_model(self, dataset, parameters={}):
        lr = parameters.get('lr' , 1e-3)
        epochs = parameters.get('epochs' , 5)
        init_scale = parameters.get('init_scale' , 1e-2)
        set_batch_size = parameters.get('set_batch_size' , 32)
        query_batch_size = parameters.get('query_batch_size' , 32)
        log_every = parameters.get('log_every' , 100)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for p in self.parameters():
            p.data.uniform_(-init_scale, init_scale)

        losses, train_accuracies = [], []

        p = Progress(epochs * ((len(dataset) + set_batch_size - 1) //
                              set_batch_size))

        random.shuffle(dataset)

        for e in range(epochs):
            for batch in batched(dataset, set_batch_size):
                optimizer.zero_grad()
                set_embeddings = self(batch)

                queries = [s + random.choice(batch) for s in batch]

                predictions, y = [], []

                for i, s in enumerate(batch):
                    if len(queries[i]) > 0:
                        predictions.append(self.query(set_embeddings[i], queries[i]))
                        y.append(torch.tensor([float(w in s) for w in queries[i]],
                                           dtype=torch.float,
                                           device=predictions[-1].device))

                loss = F.binary_cross_entropy(torch.cat(predictions),
                                              torch.cat(y))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                p.tick()
                if p.current_iteration % log_every == 0:
                    acc = (torch.cat(predictions).round() == torch.cat(y)).float().mean()
                    train_accuracies.append(acc)
                    print(p.format(),
                          'loss = {:.3f}, train_acc = {:.3f}'
                          .format(losses[-1], acc))

        return losses, train_accuracies

    def dump(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, hidden_size=128, device=None):
        if device is None:
            device = torch.device('cpu')

        semb = SetEmbedding(device, hidden_size=hidden_size)
        semb.load_state_dict(torch.load(path, map_location=device))
        semb.to(device)

        return semb
