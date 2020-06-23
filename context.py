# Context-related options and embedding.

from enum import Flag, auto
from torch import nn
from torch.nn import functional as F

class ContextAlgorithm(Flag):
    NONE = 0
    CONCAT_CELL = auto()
    FACTOR_CELL = auto()
    CNN = auto()

    @staticmethod
    def parse(s):
        return ({
            'NONE': ContextAlgorithm.NONE,
            'FACTOR_CELL': ContextAlgorithm.FACTOR_CELL,
            'CONCAT_CELL': ContextAlgorithm.CONCAT_CELL,
            'CNN': ContextAlgorithm.CNN,
            })[s]

class Context(Flag):
    NONE = 0
    IMPORTS = auto()
    IDENTIFIERS = auto()
    PREVIOUS_LINES = auto()

    def count(self):
        return sum(1
                   for v in [self.IMPORTS, self.IDENTIFIERS, self.PREVIOUS_LINES]
                   if self & v)

    @staticmethod
    def parse(s):
        d = ({
            'NONE': Context.NONE,
            'IMPORTS': Context.IMPORTS,
            'IDENTIFIERS': Context.IDENTIFIERS,
            'PREVIOUS_LINES': Context.PREVIOUS_LINES,
            })

        r = Context.NONE
        for t in s.split('+'):
            r = r | d[t]
        return r

class ContextEmbedding(nn.Module):
    def __init__(self, alphabet, params={}):
        super().__init__()

        self.context = Context.parse(params.get('context', 'NONE'))
        self.context_size = params.get('context_size', 128)
        self.n_previous_lines = params.get('n_previous_lines', 5)
        self.kernel_size = params.get('kernel_size', 5)
        self.alphabet = alphabet

        self.conv = nn.Conv1d(self.alphabet.embedding_size(),
                              self.context_size,
                              self.kernel_size,
                              padding=(self.kernel_size // 2))

    def forward(self, batch):
        contexts = [[] for _ in batch]
        context_lines = 0

        if self.context & Context.PREVIOUS_LINES:
            for i, r in enumerate(batch):
                contexts[i].extend(pad(r['p'], self.n_previous_lines))
            context_lines += self.n_previous_lines

        if self.context & Context.IMPORTS:
            for i, r in enumerate(batch):
                contexts[i].append(' '.join(r['i']))
            context_lines += 1

        all_strings = [s for c in contexts for s in c]
        enc = self.alphabet.encode_batch(all_strings)
        embedded_enc, _ = self.conv(enc.transpose(1, 2)).max(dim=2)
        ctx, _ = embedded_enc.reshape((len(batch), context_lines, -1)).max(dim=1)

        return ctx

def pad(l, n):
    return [''] * (n - len(l)) + l[-n:]
