# Neural model that embeds a set of words

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util import broadcast_dot, batched, Progress
import alphabet

class SetEmbedding(nn.Module):
    def __init__(self, device, hidden_size=128, char_embedding_size=50):
        super().__init__()

        self.word_encoder = nn.LSTM(char_embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.word_embedding_proj = nn.Linear(4*hidden_size, hidden_size)
        self.set_embedding_proj = nn.Linear(2*hidden_size, hidden_size)
        self.set_encoder = nn.LSTM(hidden_size, hidden_size)
        self.alphabet = alphabet.AsciiEmbeddedEncoding(device)
        self.set_begin_word = '^'
        self.device = device

    def embed(self, word_sets):
        word_sets = [['^'] + s for s in word_sets]
        set_lens = list(map(len, word_sets))
        set_begin = np.cumsum([0] + set_lens)

        encoded_chars = self.alphabet.encode_batch([w for s in word_sets for w in s])

        max_len = encoded_chars.shape[1]

        _, (words_h_n, words_c_n)  = self.word_encoder(encoded_chars)

        D, W, E = words_h_n.shape

        word_embeddings = torch.cat((words_h_n.transpose(0, 1).reshape((W, -1)),
                                     words_c_n.transpose(0, 1).reshape(W, -1)), dim=1)

        word_embeddings = self.word_embedding_proj(word_embeddings)

        word_embeddings_by_set = nn.utils.rnn.pad_sequence([word_embeddings[set_begin[i]:set_begin[i+1]]
                                                            for i in range(len(word_sets))], batch_first=False)

        _, (sets_h_n, sets_c_n) = self.set_encoder(word_embeddings_by_set)

        set_embeddings = torch.cat((sets_h_n.transpose(0, 1).reshape((len(word_sets), -1)),
                                    sets_c_n.transpose(0, 1).reshape((len(word_sets), -1))), dim=1)
        set_embeddings = self.set_embedding_proj(set_embeddings)

        return set_embeddings

    def query(self, set_embedding, words):
        encoded_chars = self.alphabet.encode_batch(words)
        _, (words_h_n, words_c_n)  = self.word_encoder(encoded_chars)
        W = len(words)
        word_embeddings = torch.cat((words_h_n.transpose(0, 1).reshape((W, -1)),
                                     words_c_n.transpose(0, 1).reshape(W, -1)), dim=1)
        word_embeddings = self.word_embedding_proj(word_embeddings)

        return broadcast_dot(word_embeddings, set_embedding).sigmoid()

    def train(self, dataset, parameters={}):
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

        all_elements = list(set(w for s in dataset for w in s))

        p = Progress(epochs * ((len(dataset) + set_batch_size - 1) //
                              set_batch_size))

        for e in range(epochs):
            for batch in batched(dataset, set_batch_size):
                optimizer.zero_grad()
                set_embeddings = self.embed(batch)

                queries = [s + random.choice(batch) for s in batch]

                predictions, y = [], []

                for i, s in enumerate(batch):
                    if len(queries[i]) > 0:
                        predictions.append(self.query(set_embeddings[i], queries[i]))
                        y.append(torch.tensor([float(w in s) for w in queries[i]],
                                           dtype=torch.float,
                                           device=self.device))

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
