# Seq2Seq decoder model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import broadcast_dot
from alphabet import AsciiEmbeddedEncoding, BytePairEncoding
from contextual_lstm import ContextualLSTMCell
from cnn_set_embedding import CNNSetEmbedding
from context import *

class AutoCompleteDecoderModel(nn.Module):
    def __init__(self, params, device=None):
        super().__init__()

        hidden_size = params.get('hidden_size', 100)
        max_test_length = params.get('max_test_length', 200)
        dropout_rate = params.get('dropout_rate', 0.2)
        context = Context.parse(params.get('context', 'NONE'))
        context_size = params.get('context_size', 128)

        alphabet_params = params.get('alphabet', {})
        if alphabet_params.get('type', 'CHAR') == 'CHAR':
            self.alphabet = alphabet = AsciiEmbeddedEncoding(device)
        else:
            self.alphabet = alphabet = BytePairEncoding(alphabet_params, device)

        self.hidden_size = hidden_size
        self.encoder_lstm = nn.LSTM(alphabet.embedding_size(), hidden_size,
                                    batch_first=True, bidirectional=True)

        self.context = context

        self.decoder_lstm = nn.LSTMCell(
                alphabet.embedding_size() + (context_size
                                             if context != Context.NONE
                                             else 0),
                hidden_size)

        if context.count():
            self.context_emb = ContextEmbedding(self.alphabet, params)

        self.h_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.c_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.attention_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(3*hidden_size, hidden_size, bias=False)
        self.vocab_proj = nn.Linear(hidden_size, alphabet.alphabet_size(), bias=False)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.max_test_length = max_test_length

        self.device = device
        if device:
            self.to(device)

        self.params = params

    def forward(self,
                compressed,
                context = None,
                expected = None,
                return_loss = True):
        '''Forward pass, for test time if expected is None, otherwise for training.

        Encodes the compressed input sentences in C and decodes them using
        the sentences in T as the target.

        - @param compressed: list of compressed strings.
        - @param expected: list of expected (expanded) output string.
        - @param return_loss: whether to return the loss or probabilities.
        - @returns if expected is passed and return_loss is True, returns the loss term.
                   if expected is passed and return_loss is False, returns the probabilities of each prediction.
                   Otherwise, returns a list with one string with the predictions for each input
                   compressed string.

        '''
        alphabet = self.alphabet

        B = len(compressed)
        is_training = expected is not None
        C = alphabet.encode_batch(compressed)
        C_indices = alphabet.encode_batch_indices(compressed)
        C_padding_tokens = torch.zeros((C.shape[0], C.shape[1]),
                                        dtype=torch.long,
                                        device=alphabet.device)
        for i in range(B):
            # Everything after string (+ 2 tokens for begin and end) gets set to 1.
            # Used to make attention ignore padding tokens.
            C_padding_tokens[i][len(compressed[i]) + 2:] = 1

        encoder_hidden_states, (enc_hn, enc_cn) = self.encoder_lstm(C)
        decoder_state = (self.h_proj(enc_hn.transpose(0, 1).reshape(B, -1)),
                         self.c_proj(enc_cn.transpose(0, 1).reshape(B, -1)))

        if is_training:
            E =  alphabet.encode_batch_indices(expected)
            E_emb =  alphabet.encode_batch(expected)
            predictions = []

        finished = torch.zeros(B, device=alphabet.device)
        decoded_strings = [[] for _ in range(B)]

        i = 0
        next_input = alphabet.get_start_token().repeat(B, 1)

        last_output = None
        all_finished = False

        encoder_hidden_states_proj = self.attention_proj(encoder_hidden_states)

        # If using context embeddings, compute context weights for the batch.
        using_context = (self.context != Context.NONE)

        if using_context:
            if context is None:
                raise ValueError('Expected a context matrix.')

        while not all_finished:
            if not using_context:
                decoder_state = (decoder_hidden, decoder_cell) = self.decoder_lstm(next_input, decoder_state)
            else:
                decoder_state = (decoder_hidden, decoder_cell) = self.decoder_lstm(
                        torch.cat([next_input, context], dim=1),
                        decoder_state)

            # decoder_hidden: (B, H)
            # encoder_hidden_states: (B, L, H)
            attention_scores = torch.squeeze(torch.bmm(encoder_hidden_states_proj, # (B, L, H)
                                                       torch.unsqueeze(decoder_hidden, -1) # (B, H, 1)
                                                       ), 2) # -> (B, L)

            # Set attention scores to -infinity at padding tokens.
            attention_scores.data.masked_fill_(C_padding_tokens.bool(), -float('inf'))

            attention_d = F.softmax(attention_scores, dim=1)
            attention_result = torch.squeeze(torch.bmm(torch.unsqueeze(attention_d, 1),
                                             encoder_hidden_states), dim=1)
            U = torch.cat([decoder_hidden, attention_result], dim=1)
            V = self.output_proj(U)

            timestep_out = self.dropout(V.tanh())
            proj = self.vocab_proj(timestep_out)
            last_output = F.softmax(proj, dim=1)

            if is_training:
                predictions.append(last_output)
                next_input = E_emb[:, i + 1]
            else:
                # At test time, set next input to last predicted character
                # (greedy decoding).
                predictions = last_output.argmax(dim=1)
                finished[predictions ==  alphabet.end_token_index()] = 1

                for idx in (finished == 0).nonzero():
                    decoded_strings[idx].append(
                             alphabet.index_to_char(predictions[idx]))

                next_input = torch.cat([
                     alphabet.encode_tensor_indices(predictions),
                    timestep_out,
                ], dim=1)

            i += 1

            if is_training:
                all_finished = (i + 1 == E.shape[1])
            else:
                all_finished = i == self.max_test_length or finished.sum() == B

        if is_training:
            predictions = torch.stack(predictions, dim=1)
            if return_loss:
                return (
                    F.nll_loss(
                        predictions.transpose(1, 2).log(),
                        E[:, 1:],
                        ignore_index= alphabet.padding_token_index(),
                        reduction='none')
                )
            else:
                return predictions
        else:
            return [''.join(s) for s in decoded_strings]

    def clone(self):
        c = AutoCompleteDecoderModel(self.params, self.device)
        c.load_state_dict(self.state_dict())
        return c

    def dump(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, params, device=None):
        if device is None:
            device = torch.device('cpu')

        decoder = AutoCompleteDecoderModel(params, device)
        decoder.load_state_dict(torch.load(path, map_location=device))
        decoder.to(device)
        decoder.eval()
        return decoder

    def encode_context(self, batch):
        if self.context.count():
            return self.context_emb(batch)
        return None
