# Convert between character and vector representations of strings.

import torch
from torch.nn import functional as F
import torch.nn as nn
import collections
import json

class AlphabetEncoding:
    def alphabet_size(self):
        'Returns the number of characters in the alphabet.'
        raise NotImplemented()

    def embedding_size(self):
        'returns number of dimensions in the embedding / encoding'
        raise NotImplemented()

    def encode(self, s):
        'Given a string, returns a 2D tensor representation of it.'
        raise NotImplemented()

    def decode(self, t):
        '''Given a 2D tensor where the last dimension corresponds to character embeddings,
        decodes the tensor into a string.'''
        raise NotImplemented()

    def get_padding_token(self):
        '''Returns the encoding of the padding token.'''
        raise NotImplemented()

    def get_start_token(self):
        '''Returns the encoding of the start token.'''
        raise NotImplemented()

    def get_end_token(self):
        '''Returns the encoding of the end token.'''
        raise NotImplemented()

    def get_copy_token(self):
        '''Returns the encoding of the special copy token.'''
        raise NotImplemented()

    def encode_batch(self, b):
        '''Takes a batch of sentences as a list of strings encodes it as a tensor.
        - b: a list of strings, of length B

        Returns: a tensor of shape B x L x D, where L is the length of the longest
        sentence in the batch and D is the dimensionality of the alphabet encoding.'''
        raise NotImplemented()

    def char_to_index(self, c):
        '''Returns the index of a character in the encoding.'''
        raise NotImplemented()

    def index_to_char(self, c):
        '''Returns the character in the encoding given its index.'''
        raise NotImplemented()

    def is_optimizeable(self):
        'Returns whether this encoder should be optimized in end-to-end training.'
        return False

class BytePairEncoding(AlphabetEncoding):
    NUM_ASCII = 128
    PADDING_INDEX = 0
    START_INDEX = 1
    END_INDEX = 2

    def __init__(self, params, device):
        super().__init__()
        self.device = device

        self.v_size = params.get('vocabulary_size', 1000)
        self.e_size = params.get('embedding_size', 256)
        self.vocab = {}
        self.piece2ind = {}
        self.embedding = nn.Embedding(
            self.v_size, self.e_size, padding_idx=self.PADDING_INDEX)

        if params.get('vocab_file'):
            self.load_vocabulary(params['vocab_file'])

    def compute_vocabulary(self, examples):
        frequencies = collections.defaultdict(int)

        for i in range(self.NUM_ASCII):
            self.vocab[i] = chr(i)
            self.piece2ind[chr(i)] = i

        ds = [list(e) for e in examples]

        while len(self.vocab) < self.v_size:
            freq = collections.defaultdict(int)

            for l in ds:
                for i in range(len(l) - 1):
                    freq[(l[i], l[i+1])] += 1

            pair, _ = max(list(freq.items()), key=lambda p: p[1])
            piece = pair[0] + pair[1]
            index = len(self.vocab)
            self.vocab[index] = piece
            self.piece2ind[piece] = index

            for i in range(len(ds)):
                j, new_l = 0, []
                s = ds[i]

                while j < len(s):
                    if j + 1 < len(s) and s[j] == pair[0] and s[j+1] == pair[1]:
                        new_l.append(piece)
                        j += 2
                    else:
                        new_l.append(s[j])
                        j += 1

                ds[i] = new_l

            print('New piece:', index, repr(piece))

    def alphabet_size(self):
        'Returns the number of characters in the alphabet.'
        return self.v_size

    def embedding_size(self):
        'Returns number of dimensions in the embedding / encoding'
        return self.e_size

    def encode_indices(self, s, partial=False):
        'Given a string, returns a 2D tensor representation of it.'
        i, l = 0, []

        while i < len(s):
            last = self.piece2ind[s[i]]

            for j in range(i+1, len(s) + 1):
                index = self.piece2ind.get(s[i:j])
                if index is None:
                    break
                last = index

            l.append(last)
            i = j

        l_t = torch.tensor([self.START_INDEX] +
                           l +
                           ([] if partial else [self.END_INDEX]),
                            dtype=torch.long, device=self.device)
        return l_t

    def encode(self, s):
        idxs = self.encode_indices(s)
        return self.embedding(idxs)

    def embed(self, indices):
        return self.embedding(indices)

    def decode(self, t):
        raise NotImplemented()

    def get_padding_token(self):
        return self.embedding(torch.tensor(self.PADDING_INDEX, device=self.device))

    def get_start_token(self):
        return self.embedding(torch.tensor(self.START_INDEX, device=self.device))

    def get_end_token(self):
        return self.embedding(torch.tensor(self.END_INDEX, device=self.device))

    def end_token_index(self):
        return self.END_INDEX

    def padding_token_index(self):
        return self.PADDING_INDEX

    def encode_batch(self, batch):
        if len(batch) == 0:
            return torch.zeros((0, 0, self.EMBEDDING_SIZE), device=self.device)

        return self.encode_tensor_indices(self.encode_batch_indices(batch))

    def encode_batch_indices(self, batch, partial=False):
        if len(batch) == 0:
            return torch.zeros((0, 0), device=self.device)

        indices = [self.encode_indices(s, partial)]

        max_length = max(map(len, indices))
        padding_tensor = torch.tensor([self.PADDING_INDEX],
                                      dtype=torch.long, device=self.device)
        return torch.stack(
                [torch.cat([idx, padding_tensor.repeat(max_length - len(s))])
                 for s in indices])

    def char_to_index(self, c):
        return self.piece2ind[c]

    def index_to_char(self, i):
        return self.vocab[i]

    def encode_tensor_indices(self, batch):
        return self.embedding(batch)

    def is_optimizeable(self):
        return True

    def dump_vocabulary(self, path):
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'piece2ind': self.piece2ind
            }, f)

    def load_vocabulary(self, path):
        with open(path) as f:
            v = json.load(f)
            self.vocab = v['vocab']
            self.piece2ind = v['piece2ind']

class AsciiOneHotEncoding(AlphabetEncoding):
    '''One-hot encoding that only takes ASCII characters.

    Uses control ASCII characters 0, 1 and 2 for padding, start and end tokens,
    respectively.'''

    ALPHABET_SIZE = 128

    PADDING_INDEX = 0
    START_INDEX = 1
    END_INDEX = 2
    COPY_INDEX = 3

    def __init__(self, device):
        self.device = device

        self.PADDING = F.one_hot(torch.scalar_tensor(0, dtype=torch.long, device=device), self.ALPHABET_SIZE).to(torch.float)
        self.START   = F.one_hot(torch.scalar_tensor(1, dtype=torch.long, device=device), self.ALPHABET_SIZE).to(torch.float)
        self.END     = F.one_hot(torch.scalar_tensor(2, dtype=torch.long, device=device), self.ALPHABET_SIZE).to(torch.float)
        self.COPY    = F.one_hot(torch.scalar_tensor(3, dtype=torch.long, device=device), self.ALPHABET_SIZE).to(torch.float)

    def alphabet_size(self):
        return self.ALPHABET_SIZE

    def embedding_size(self):
        return self.ALPHABET_SIZE

    def encode(self, s):
        b = s.encode('ascii')
        t = torch.zeros((len(s) + 2, self.ALPHABET_SIZE), device=self.device)
        t[0] = self.START
        t[range(1, len(s) + 1), list(b)] += 1
        t[-1] = self.END
        return t

    def encode_indices(self, s, partial=False):
        b = s.encode('ascii')
        return torch.tensor([self.START_INDEX] +
                             list(b) +
                            ([] if partial else [self.END_INDEX]),
                            dtype=torch.long, device=self.device)

    def decode(self, t):
        return ''.join(map(chr, torch.argmax(t, dim=1)))

    def get_padding_token(self):
        return self.PADDING

    def get_start_token(self):
        return self.START

    def get_end_token(self):
        return self.END

    def get_copy_token(self):
        return self.COPY

    def start_token_index(self):
        return self.START_INDEX

    def end_token_index(self):
        return self.END_INDEX

    def copy_token_index(self):
        return self.COPY_INDEX

    def padding_token_index(self):
        return self.PADDING_INDEX

    def encode_batch(self, batch):
        if len(batch) == 0:
            return torch.zeros((0, 0, self.ALPHABET_SIZE), device=self.device)

        max_length = max(map(len, batch))
        return torch.stack(
                [torch.cat([self.encode(s), self.PADDING.repeat(max_length - len(s), 1)])
                 for s in batch])

    def encode_batch_indices(self, batch, partial=False):
        if len(batch) == 0:
            return torch.zeros((0, 0), device=self.device)

        max_length = max(map(len, batch))
        padding_tensor = torch.tensor([self.PADDING_INDEX],
                                      dtype=torch.long, device=self.device)
        return torch.stack(
                [torch.cat([self.encode_indices(s, partial),
                            padding_tensor.repeat(max_length - len(s))])
                 for s in batch])

    def char_to_index(self, c):
        return ord(c)

    def index_to_char(self, i):
        return chr(i)

    def encode_tensor_indices(self, batch):
        """encode batched tensor of character indices as
        a tensor of one-hot encodings

        Arguments:
            batch {[tensor]} -- tensor of batched indices of size [batch]
        Returns a tensor of dimension [batch, embedding_size]
        """
        return F.one_hot(batch, num_classes=self.ALPHABET_SIZE).to(torch.float)


class AsciiEmbeddedEncoding(AlphabetEncoding, nn.Module):
    '''character embedding for ASCII characters.

    Uses control ASCII characters 0, 1 and 2 for padding, start and end tokens,
    respectively.'''

    EMBEDDING_SIZE = 50
    NUM_ASCII = 128
    PADDING_INDEX = 0
    START_INDEX = 1
    END_INDEX = 2
    COPY_INDEX = 3

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ascii_embedding = nn.Embedding(
            self.NUM_ASCII, self.EMBEDDING_SIZE, padding_idx=self.PADDING_INDEX)

    def alphabet_size(self):
        return self.NUM_ASCII

    def embedding_size(self):
        return self.EMBEDDING_SIZE

    def encode(self, s):
        idxs = self.encode_indices(s)
        return self.ascii_embedding(idxs)

    def embed(self, indices):
        return self.ascii_embedding(indices)

    def encode_indices(self, s, partial=False):

        b = s.encode('ascii')
        return torch.tensor([self.START_INDEX] +
                             list(b) +
                            ([] if partial else [self.END_INDEX]),
                            dtype=torch.long, device=self.device)

    def decode(self, t):
        raise NotImplemented()

    def get_padding_token(self):
        return self.ascii_embedding(torch.tensor(self.PADDING_INDEX, device=self.device))

    def get_start_token(self):
        return self.ascii_embedding(torch.tensor(self.START_INDEX, device=self.device))

    def get_end_token(self):
        return self.self.ascii_embedding(torch.tensor(self.END_INDEX, device=self.device))

    def get_copy_token(self):
        return self.self.ascii_embedding(torch.tensor(self.COPY_INDEX), device=self.device)

    def end_token_index(self):
        return self.END_INDEX

    def copy_token_index(self):
        return self.COPY_INDEX

    def padding_token_index(self):
        return self.PADDING_INDEX

    def encode_batch(self, batch):
        if len(batch) == 0:
            return torch.zeros((0, 0, self.EMBEDDING_SIZE), device=self.device)

        return self.encode_tensor_indices(self.encode_batch_indices(batch))

    def encode_batch_indices(self, batch, partial=False):
        if len(batch) == 0:
            return torch.zeros((0, 0), device=self.device)

        max_length = max(map(len, batch))
        padding_tensor = torch.tensor([self.PADDING_INDEX],
                                      dtype=torch.long, device=self.device)
        return torch.stack(
                [torch.cat([self.encode_indices(s, partial),
                            padding_tensor.repeat(max_length - len(s))])
                 for s in batch])

    def char_to_index(self, c):
        return ord(c)

    def index_to_char(self, i):
        return chr(i)

    def encode_tensor_indices(self, batch):
        """encode batched tensor of character indices as
        a tensor of one-hot encodings

        Arguments:
            batch {[tensor]} -- tensor of batched indices of size [batch]
        Returns a tensor of dimension [batch, embedding_size]
        """
        return self.ascii_embedding(batch)

    def is_optimizeable(self):
        'Returns whether this encoder should be optimized in end-to-end training.'
        return True
