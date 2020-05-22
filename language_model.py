# Simple RNN-based language model

import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from cell import GatedCell
from alphabet import AsciiEmbeddedEncoding
from context import *
from cnn_set_embedding import CNNSetEmbedding
from util import Progress, batched

def split(batch):
    return zip(*[(r['l'], r['c'], r['i']) for r in batch])

class RNNLanguageModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()

        self.alphabet = AsciiEmbeddedEncoding(device)
        self.cells = nn.ModuleList()

        self.device = device
        layers = params.get('layers', 1)
        hidden_size = params.get('hidden_size', 128)
        context = Context.parse(params.get('context', 'NONE'))
        context_algorithm = ContextAlgorithm.parse(params.get('context_algorithm', 'NONE'))
        context_embedding_size = params.get('context_embedding_size', 50)

        context_input_size = (0
                              if context_algorithm != ContextAlgorithm.CNN
                              else context.count() * context_embedding_size)

        for i in range(layers):
            self.cells.append(
                nn.GRUCell(
                    (self.alphabet.embedding_size() + context_input_size) if i == 0 else hidden_size,
                    hidden_size
                )
            )

        if context_algorithm == ContextAlgorithm.CNN:
            self.context_cnn = CNNSetEmbedding(device, context_embedding_size)

        self.output = nn.Linear(hidden_size + context_input_size, self.alphabet.alphabet_size())
        self.context = context

        self.to(device)

    def encode(self, batch_l, batch_i, batch_c, partial=False):
        C_idx = self.alphabet.encode_batch_indices(batch_l, partial)

        if self.context:
            context_tensors = []

            if self.context & Context.IMPORTS:
                context_tensors.append(self.context_cnn(batch_i))

            if self.context & Context.IDENTIFIERS:
                context_tensors.append(self.context_cnn(batch_c))

            ctx = torch.cat(context_tensors, dim=1)
        else:
            ctx = torch.zeros((len(batch_l), 0),
                              dtype=torch.float,
                              device=self.device)

        return C_idx, ctx

    def forward(self, batch, context):
        output_prob = []
        state = [None] * len(self.cells)

        batch_emb = self.alphabet.embed(batch)
        expected = batch[:, 1:]
        B, L = batch.shape

        for i in range(L - 1):
            input_i = torch.cat([batch_emb[:, i, :], context], dim=1)

            for layer, cell in enumerate(self.cells):
                state[layer] = cell(input_i, state[layer])
                input_i = state[layer]

            output_i = F.log_softmax(self.output(torch.cat([input_i, context], dim=1)), dim=1)
            output_prob.append(output_i[torch.arange(0, B), expected[:, i]].unsqueeze(0))

        return torch.cat(output_prob).transpose(0, 1)

    def dump(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(params, path, device):
        lm = RNNLanguageModel(params, device)
        lm.load_state_dict(torch.load(path, map_location=device))
        lm.to(device)
        lm.eval()
        return lm

    def fit(self, dataset, tracker, params):
        learning_rate = params.get('learning_rate') or 1e-2
        batch_size = params.get('batch_size') or 32
        init_scale = params.get('init_scale') or 0.1
        epochs = params.get('epochs', 1)
        log_every = params.get('log_every') or 100

        validations_per_epoch = params.get('validations_per_epoch') or 10
        max_validation_examples = params.get('max_validation_examples') or 10**9
        loss_convergence_window = params.get('loss_convergence_window') or None
        loss_convergence_improvement = params.get('loss_convergence_improvement') or None

        training_set, validation_set = dataset['train'], dataset['dev'][:max_validation_examples]

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for p in self.parameters():
            p.data.uniform_(-init_scale, init_scale)

        batches_per_epoch = math.ceil(len(training_set) / batch_size)
        total_batches = epochs * batches_per_epoch
        validate_every = max(batches_per_epoch // validations_per_epoch, 1)

        p = Progress(total_iterations=total_batches if epochs else None)

        for e in range(epochs):
            random.shuffle(training_set)

            for i, batch in enumerate(batched(training_set, batch_size)):
                optimizer.zero_grad()
                log_ppl = self.compute_log_perplexity(batch).mean()
                log_ppl.backward()
                optimizer.step()

                ppl = log_ppl.exp().item()

                if p.tick() % log_every == 0:
                    print('Epoch {} batch {}: ppl = {:.3f}, {}'.format(e, i, ppl, p.format()))

                tracker.step()
                tracker.add_scalar('train/ppl', ppl)

                if (i + 1) % validate_every == 0:
                    tracker.add_scalar(
                            'val/ppl',
                            self.compute_log_perplexity(validation_set, batch_size, False)
                                .mean().exp().item())
                    tracker.checkpoint()

        tracker.add_scalar(
                'val/ppl',
                self.compute_log_perplexity(validation_set, batch_size, False)
                    .mean().exp().item())

    def compute_log_perplexity(self, dataset, batch_size=None, grad=True, partial=False):
        log_ppls = []

        with torch.autograd.set_grad_enabled(grad):
            for batch in batched(dataset, batch_size or len(dataset)):
                batch_l, batch_c, batch_i = split(batch)
                idx, ctx = self.encode(batch_l, batch_i, batch_c, partial)
                lengths = torch.tensor([len(i) + (not partial) for i in batch_l], device=self.device)

                log_probs = self(idx, ctx)
                log_probs.masked_fill_(idx[:, 1:] == self.alphabet.padding_token_index(), 0)
                log_ppl = log_probs.sum(dim=1) / lengths
                log_ppls.append(log_ppl)

            return -torch.cat(log_ppls)

class DiscriminativeLanguageModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()

        self.alphabet = AsciiEmbeddedEncoding(device)
        self.device = device

        hidden_size = params.get('hidden_size', 128)
        context = Context.parse(params.get('context', 'NONE'))
        context_algorithm = ContextAlgorithm.parse(params.get('context_algorithm', 'NONE'))
        context_embedding_size = params.get('context_embedding_size', 50)
        cell_params = params.get('cell', {'type': 'GRU'})

        context_input_size = (0
                              if context_algorithm != ContextAlgorithm.CNN
                              else context.count() * context_embedding_size)

        self.cell = GatedCell({
            **cell_params,
            'input_size': self.alphabet.embedding_size() + context_input_size,
            'hidden_size': hidden_size,
        })

        if context_algorithm == ContextAlgorithm.CNN:
            self.context_cnn = CNNSetEmbedding(device, context_embedding_size)

        self.output = nn.Linear(hidden_size, 1)
        self.context = context

        self.to(device)

    def encode(self, batch, partial=False):
        batch_l, batch_i, batch_c = split(batch)
        C_idx = self.alphabet.encode_batch_indices(batch_l, partial)

        if self.context:
            context_tensors = []

            if self.context & Context.IMPORTS:
                context_tensors.append(self.context_cnn(batch_i))

            if self.context & Context.IDENTIFIERS:
                context_tensors.append(self.context_cnn(batch_c))

            ctx = torch.cat(context_tensors, dim=1)
        else:
            ctx = torch.zeros((len(batch_l), 0),
                              dtype=torch.float,
                              device=self.device)

        return C_idx, ctx

    def forward(self, batch, context):
        output_prob = []
        state = None

        batch_emb = self.alphabet.embed(batch)
        B, L = batch.shape

        for i in range(L - 1):
            input_i = torch.cat([batch_emb[:, i, :], context], dim=1)
            state = self.cell(input_i, state)

        return self.output(self.cell.get_cell(state)).squeeze(1).sigmoid()

    def dump(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(params, path, device):
        dlm = DiscriminativeLanguageModel(params, device)
        dlm.load_state_dict(torch.load(path, map_location=device))
        dlm.to(device)
        dlm.eval()
        return dlm

    def fit(self, dataset, tracker, params):
        learning_rate = params.get('learning_rate') or 1e-2
        batch_size = params.get('batch_size') or 32
        init_scale = params.get('init_scale') or 0.1
        epochs = params.get('epochs', 1)
        log_every = params.get('log_every') or 100

        validations_per_epoch = params.get('validations_per_epoch') or 10
        max_validation_examples = params.get('max_validation_examples') or 10**9
        loss_convergence_window = params.get('loss_convergence_window') or None
        loss_convergence_improvement = params.get('loss_convergence_improvement') or None

        training_set, validation_set = dataset['train'], dataset['dev'][:max_validation_examples]

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for p in self.parameters():
            p.data.uniform_(-init_scale, init_scale)

        batches_per_epoch = math.ceil(len(training_set) / batch_size)
        total_batches = epochs * batches_per_epoch
        validate_every = max(batches_per_epoch // validations_per_epoch, 1)

        p = Progress(total_iterations=total_batches if epochs else None)

        for e in range(epochs):
            random.shuffle(training_set)

            for i, batch in enumerate(batched(training_set, batch_size)):
                optimizer.zero_grad()
                X, y = zip(*batch)
                X_i, X_c = self.encode(X)
                y_hat = self(X_i, X_c)
                y_true = torch.tensor(y, dtype=torch.float).to(self.device)

                loss = F.binary_cross_entropy(y_hat, y_true)
                loss.backward()
                optimizer.step()

                acc = (y_hat.round() == y_true).float().mean().item()

                if p.tick() % log_every == 0:
                    print('Epoch {} batch {}: acc = {:.3f}, {}'.format(e, i, acc, p.format()))

                tracker.step()
                tracker.add_scalar('train/loss', loss.item())
                tracker.add_scalar('train/acc', acc)

                if (i + 1) % validate_every == 0:
                    tracker.add_scalar('val/acc', self.compute_accuracy(validation_set, batch_size))
                    tracker.checkpoint()

        tracker.add_scalar('val/acc', self.compute_accuracy(validation_set, batch_size))

    def compute_accuracy(self, examples, batch_size=128):
        correct = 0

        with torch.no_grad():
            for batch in batched(examples, batch_size):
                X, y = zip(*batch)
                X_i, X_c = self.encode(X)
                y_hat = self(X_i, X_c)
                y_true = torch.tensor(y, dtype=torch.float).to(self.device)
                correct += int((y_hat.round() == y_true).sum().item())

        return correct / len(examples)
