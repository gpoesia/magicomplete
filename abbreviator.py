# System that takes a set of common strings and creates reliable abbreviations from them.

import random
from user import User
import torch
from torch.nn import functional as F
import numpy as np
from util import batched, split_at_identifier_boundaries, Progress, is_subsequence
import collections
import json
from models import load_from_run
from language_model import RNNLanguageModel, DiscriminativeLanguageModel
from decoder import AutoCompleteDecoderModel
import math
import time

class AbbreviationAlgorithm:
    def generate_alternatives(self, string):
        raise NotImplemented()

def is_identifier_char(c):
    return c.isalnum() or c == '_'

class EraseSuffixes(AbbreviationAlgorithm):
    def generate_alternatives(self, s):
        order = []

        for i, c in enumerate(s):
            if i > 0 and is_identifier_char(c) and is_identifier_char(s[i-1]):
                order.append(([i], (0, -i)))

        order.sort(key=lambda p: p[1])

        alternatives = []

        for (remove, _) in order:
            alternatives.append(''.join(c for i, c in enumerate(s) if i not in remove))

        return alternatives

class LanguageAbbreviator:
    def __init__(self, decoder, set_embedding, training_set, parameters={}):
        self.description = parameters.get('description')
        self.decoder = decoder.clone()
        self.set_embedding = set_embedding
        self.training_set = training_set
        self.user = User()

        self.minimum_validation_accuracy = parameters.get('minimum_validation_accuracy', 0.8)
        self.minimum_train_examples = parameters.get('minimum_train_examples', 64)
        self.val_examples = parameters.get('val_examples', 32)
        self.learning_rate = parameters.get('learning_rate', 0.1)
        self.max_steps = parameters.get('max_steps', 100)
        self.batch_size = parameters.get('batch_size', 32)
        self.rehearsal_batch_size = parameters.get('rehearsal_batch_size', 32)
        abbreviation_algorithm = parameters.get('abbreviation_algorithm', 'erase_suffixes')

        assert abbreviation_algorithm in ('erase_suffixes'), 'Unknown abbreviation algorithm'

        self.abbreviation_algorithm = {
            'erase_suffixes': EraseSuffixes
            }[abbreviation_algorithm]()

        self.rehearsal_examples = []

    def encode(self, batch):
        batch_l, batch_i, batch_c = zip(*[(r['l'], r['i'], r['c']) for r in batch])
        ctx = self.decoder.compute_context(batch_i, batch_c, self.set_embedding)
        return [{ **r, 'l': self.user.encode(r['l'])} for r in batch], ctx

    def decode(self, batch_encoded, ctx):
        batch_l = [r['l'] for r in batch_encoded]
        return self.decoder(batch_l, ctx)

    def name(self):
        return ('LanguageAbbreviator({}, min_val_acc={:.2f}, min_train={}, val_ex={}, lr={}, max_steps={}, rehearsal={})'
                .format(self.description,
                        self.minimum_validation_accuracy,
                        self.minimum_train_examples,
                        self.val_examples,
                        self.learning_rate,
                        self.max_steps,
                        self.rehearsal_batch_size))

    def find_abbreviation(self, string):
        abbreviation = string
        validation_accuracy = 1.0

        examples = list(s for s in self.training_set if s['l'].find(string) != -1)

        if len(examples) < self.minimum_train_examples + self.val_examples:
            return string

        train, val = examples[:-self.val_examples], examples[-self.val_examples:]
        initial = self.try_learn_abbreviation(string, abbreviation, train, val)

        alternatives = [abbreviation]

        if not initial:
            return abbreviation

        while len(alternatives) > 0:
            success = False

            for a in alternatives:
                success = self.try_learn_abbreviation(string, a, train, val)

                if success:
                    abbreviation = a
                    alternatives = self.abbreviation_algorithm.generate_alternatives(a)
                    break

            if not success:
                break

        if abbreviation != string:
            self.rehearsal_examples.extend(train)
            self.user.add_new_convention(string, abbreviation)

        return abbreviation

    def try_learn_abbreviation(self, string, abbreviation, training_set, val_set):
        new_decoder = self.decoder.clone()
        optimizer = torch.optim.SGD(new_decoder.parameters(), lr=self.learning_rate)

        encode = lambda s: self.user.encode(s).replace(string, abbreviation)

        for i in range(self.max_steps):
            new_decoder.train()
            optimizer.zero_grad()

            batch = random.sample(training_set, self.batch_size)

            batch.extend(random.sample(self.rehearsal_examples, min(len(self.rehearsal_examples),
                                                                        self.rehearsal_batch_size)))

            original = [s['l'] for s in batch]
            batch_encoded = [encode(s) for s in original]
            batch_imports, batch_ids = zip(*[(s['i'], s['c']) for s in batch])
            ctx = new_decoder.compute_context(batch_imports,
                                              batch_ids,
                                              self.set_embedding)

            loss = new_decoder(batch_encoded, ctx, original).mean()
            loss.backward()

            optimizer.step()

            new_decoder.eval()
            val_predictions = []

            for val_batch in batched(val_set, self.batch_size):
                val_l, val_i, val_c = zip(*[(s['l'], s['i'], s['c'])
                                            for s in val_batch])
                val_batch_encoded = [encode(l) for l in val_l]
                val_ctx = new_decoder.compute_context(val_i, val_c, self.set_embedding)
                predictions = new_decoder(val_batch_encoded, val_ctx)
                val_predictions.extend(p == s for p, s in zip(predictions, val_l))

            val_acc = np.mean(val_predictions)

            if val_acc >= self.minimum_validation_accuracy:
                self.decoder = new_decoder
                return True

        return False

    def dump(self, path):
        self.decoder.dump(path)

class LMRLanguageAbbreviator:
    def __init__(self, lm, training_set, parameters={}):
        self.parameters = parameters
        self.description = parameters.get('description') or 'LMR'
        self.lm = lm
        self.training_set = training_set

        self.training_set_tokens = [split_at_identifier_boundaries(r['l'])
                                    for r in training_set]

        self.abbreviation_table = {}
        self.inverted_abbreviations = collections.defaultdict(list)

        self.minimum_validation_accuracy = parameters.get('minimum_validation_accuracy') or 0.8
        self.val_examples = parameters.get('val_examples') or 64
        self.rehearsal_examples = parameters.get('rehearsal_examples') or 64
        self.batch_size = parameters.get('batch_size') or 128
        self.beam_size = parameters.get('beam_size') or 32
        self.evaluation_examples = []

    def encode_tokens(self, tokens):
        ts = []
        for t in tokens:
            v = self.abbreviation_table.get(t)
            ts.append(v or t)
        return ts

    def encode(self, batch):
        encoded_l = [{ **r,
                       'l': ''.join(self.encode_tokens(split_at_identifier_boundaries(r['l']))) }
                     for r in batch]

        return encoded_l, None

    def beam_search(self, line, imports, identifiers):
        i = 0
        tokens = split_at_identifier_boundaries(line)
        candidates = [()]

        for i, t in enumerate(tokens):
            next_candidates = []

            for c in candidates:
                for expansion in ([(t,)] + self.inverted_abbreviations.get(t, [])):
                    next_candidates.append(c + expansion)

            # Rank candidates either in the last iteration or if there are too many.
            if len(next_candidates) > 1 and \
               (i == len(tokens) - 1 or len(next_candidates) > self.beam_size):

                perplexities = self.lm.compute_log_perplexity(
                    [{ 'l': ''.join(c), 'i': imports, 'c': identifiers } for c in next_candidates],
                    self.batch_size,
                    grad=False,
                    partial=True)

                ppl_by_candidate = list(zip(perplexities, next_candidates))
                ppl_by_candidate.sort()
                next_candidates = [c for _, c in ppl_by_candidate][:self.beam_size]

            candidates = next_candidates

        return [''.join(c) for c in candidates]

    def decode(self, batch_encoded, idx_ctx=None):
        return [self.beam_search(row['l'], row['i'], row['c'])[0] for row in batch_encoded]

    def name(self):
        return ('LMRLanguageAbbreviator({}, min_val_acc={:.2f}, rehearsal_examples={})'
                .format(self.description,
                        self.minimum_validation_accuracy,
                        self.rehearsal_examples))

    def find_tokens(self, s, tokens):
        s_t = split_at_identifier_boundaries(s)

        for i in range(len(s)):
            if tokens == s_t[i:i+len(tokens)]:
                return i

        return -1

    def list_candidate_abbreviations(self, s):
        return [s[:l] for l in range(1, len(s)) if s[:l].isidentifier()]

    def find_abbreviation(self, string):
        string_tokens = split_at_identifier_boundaries(string)

        examples = []

        for s, t in zip(self.training_set, self.training_set_tokens):
            if self.find_tokens(s['l'], string_tokens) != -1:
                examples.append(s)
            if len(examples) == self.val_examples:
                break

        if len(examples) < self.val_examples:
            return string

        for abbreviation in self.list_candidate_abbreviations(string):
            accuracy = self.evaluate_new_abbreviation(string_tokens, abbreviation, examples)

            if accuracy >= self.minimum_validation_accuracy:
                self.abbreviation_table[string_tokens] = abbreviation
                self.inverted_abbreviations[abbreviation].append(string_tokens)
                self.evaluation_examples.extend(examples)
                return abbreviation

        return string

    def evaluate_new_abbreviation(self, string_tokens, abbreviation, val_set):
        rehearsal = random.sample(self.evaluation_examples,
                                  k=min(len(self.evaluation_examples),
                                        self.rehearsal_examples))
        examples = rehearsal + val_set

        self.abbreviation_table[string_tokens] = abbreviation
        self.inverted_abbreviations[abbreviation].append(string_tokens)

        encoded_examples, _ = self.encode(examples)
        decoded_examples = self.decode(encoded_examples)

        correct = [dec == original['l'] for dec, original in zip(decoded_examples, examples)]
        correct_val = correct[:len(rehearsal)]
        correct_new = correct[len(rehearsal):]

        accuracy_val = np.mean(correct_val) if len(correct_val) else 1.0
        accuracy_new = np.mean(correct_new)

        del self.abbreviation_table[string_tokens]
        self.inverted_abbreviations[abbreviation].pop()

        print('Accuracy with abbreviation {} => {} on {} examples: {:.2f}% val, {:.2f}% positive'
              .format(
                  ''.join(string_tokens),
                  abbreviation,
                  len(examples),
                  100*accuracy_val,
                  100*accuracy_new))

        errors = [(dec, original['l']) for dec, original in zip(decoded_examples, examples)
                  if dec != original['l']][:10]

        return min(accuracy_val, accuracy_new)

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump({#'parameters': self.parameters,
                       'abbreviation_table': { (''.join(k)): v for k, v in self.abbreviation_table.items() },
                       'inverted_abbreviations': self.inverted_abbreviations },
                      f)

    @staticmethod
    def load(params, path, device):
        with open(path) as f:
            tables = json.load(f)
        lm = load_from_run(RNNLanguageModel, params['lm'], device, 'model')
        abbreviator = LMRLanguageAbbreviator(lm, [], params)

        abbreviator.abbreviation_table = {
            k: v for k, v in tables['abbreviation_table'].items()
        }
        abbreviator.inverted_abbreviations = {
            k: [tuple(u) for u in v]
            for k, v in tables['inverted_abbreviations'].items()
        }

        return abbreviator

class CLMLanguageAbbreviator:
    def __init__(self, clm, abbreviation_targets, params={}):
        self.params = params
        self.clm = clm.clone()
        self.abbreviation_targets = abbreviation_targets
        self.abbreviation_targets_set = set(abbreviation_targets)
        self.inverted_abbreviations = collections.defaultdict(list)
        self.evaluation_examples = []
        self.rehearsal_examples = []

        for t in abbreviation_targets:
            self.inverted_abbreviations[t[0]].append(t)

        self._load_params()

    def _load_params(self):
        self.description = self.params.get('description') or 'CLM'
        self.val_examples = self.params.get('val_examples') or 512
        self.batch_size = self.params.get('batch_size') or 128
        self.beam_size = self.params.get('beam_size') or 32

    def list_candidate_expansions(self, token):
        candidates = [token]
        for t in self.abbreviation_targets:
            if t != token and is_subsequence(token, t):
                candidates.append(t)
        return candidates

    def encode_tokens(self, tokens):
        return [(t[0] if t in self.abbreviation_targets_set else t) for t in tokens]

    def encode(self, batch):
        encoded_l = [{ **r,
                       'l': ''.join(self.encode_tokens(split_at_identifier_boundaries(r['l']))) }
                     for r in batch]

        return encoded_l, None

    def compute_accuracy(self, examples):
        if len(examples) == 0:
            return 1.0
        queries = [{ **ex,
                     'l': self.abbreviate(ex['l'], self.abbreviation_targets) }
                     for ex in examples]
        predictions = self.beam_search(queries, beam_size=2)
        correct = [r['l'] == p[0] for r, p in zip(examples, predictions)]
        return sum(correct) / len(examples)
 
    def beam_search(self, queries, beam_size=None):
        if len(queries) == 0:
            return []

        with torch.no_grad():
            beam_size = beam_size or self.beam_size
            tokens = [split_at_identifier_boundaries(r['l']) for r in queries]
            candidates = [[()] for _ in queries]

            for t_i in range(max(len(t) for t in tokens)):
                need_reranking = []
                reranking_list = []

                for i in range(len(queries)):
                    if t_i < len(tokens[i]):
                        t = tokens[i][t_i]
                        expansions = self.list_candidate_expansions(t)
                        next_candidates = []

                        for c in candidates[i]:
                            for e in expansions:
                                next_candidates.append(c + (e,))

                        candidates[i] = next_candidates

                    if len(candidates[i]) > beam_size or \
                            (len(candidates[i]) > 1 and t_i + 1 == len(tokens[i])):
                        need_reranking.append((
                            i, 
                            len(reranking_list),
                            len(reranking_list) + len(candidates[i])))
                        reranking_list.extend({ 'l': ''.join(c), 
                                                's': queries[i]['l'],
                                                'p': queries[i].get('p', []),
                                                'i': queries[i]['i'],
                                                'c': queries[i]['c'] }
                                                for c in candidates[i])

                if len(need_reranking) > 0:
                    scores = []
                    for batch in batched(reranking_list, self.batch_size):
                        batch_l, batch_s = zip(*[(r['l'], r['s']) for r in batch])
                        context = self.clm.encode_context(batch)
                        lens = torch.tensor([len(l) + 1 for l in batch_l], device=self.clm.device)

                        scores.extend(list(-self.clm(batch_s, context, batch_l,
                                                     return_loss=True).sum(dim=1) / lens))

                    for i, lo, hi in need_reranking:
                        candidates_with_scores = list(zip(reranking_list[lo:hi],
                                                          scores[lo:hi]))
                        candidates_with_scores.sort(key=lambda cs: cs[1], reverse=True)
                        candidates[i] = [split_at_identifier_boundaries(c['l'])
                                         for c, _ in candidates_with_scores][:beam_size]

            return [[''.join(c) for c in candidates_i] for candidates_i in candidates]

    def decode(self, batch_encoded, ctx=None):
        return [r[0] for r in self.beam_search(batch_encoded)]

    def name(self):
        return ('CLMLanguageAbbreviator({}, pre-trained'
                .format(self.description))

    def find_token(self, s, t):
        return t in split_at_identifier_boundaries(s)

    def find_abbreviation(self, string):
        return (string[0]
                if string in self.abbreviation_targets_set
                else string)

    def dump(self, path):
        torch.save({
            'params': self.params,
            'abbreviation_targets': self.abbreviation_targets,
            'clm': self.clm.state_dict(),
            }, path)

    @staticmethod
    def load(path, device):
        m = torch.load(path, map_location=device)
        params = m['params']
        clm = AutoCompleteDecoderModel(params.get('clm', {}), device)
        clm.load_state_dict(m['clm'])
        abbreviator = CLMLanguageAbbreviator(
                clm, m['abbreviation_targets'], params)
        return abbreviator

    def abbreviate(self, line, targets_set):
        tokens = split_at_identifier_boundaries(line)
        abbrev_tokens = []

        for t in tokens:
            if t in targets_set:
                abbrev_tokens.append(t[0])
            else:
                abbrev_tokens.append(t)

        return ''.join(abbrev_tokens)

    def fit(self, tracker, dataset):
        learning_rate = self.params.get('learning_rate', 1e-3)
        batch_size = self.params.get('batch_size') or 32
        init_scale = self.params.get('init_scale') or 0.1
        epochs = self.params.get('epochs', 1)
        log_every = self.params.get('log_every') or 100
        validate_every = self.params.get('validate_every') or 1000
        val_examples = self.params.get('val_examples') or 1000
        lr_gamma = self.params.get('lr_gamma') or 1.0

        training_set, val_set = dataset['train'], dataset['dev']
        batches_per_epoch = math.ceil(len(training_set) * epochs / batch_size)
        total_batches = epochs * batches_per_epoch
        val_acc = 0

        random.shuffle(val_set)

        for p in self.clm.parameters():
            if len(p.shape) >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p, -init_scale, init_scale)

        optimizer = torch.optim.Adam(self.clm.parameters(),
                                     lr=learning_rate)
        p = Progress(total_iterations=total_batches)

        for e in range(epochs):
            random.shuffle(training_set)

            for i, batch in enumerate(batched(training_set, batch_size)):
                optimizer.zero_grad()
                batch_l = [r['l'] for r in batch]
                batch_enc = [r['l'] for r in self.encode(batch)[0]]
                lengths = torch.tensor([len(l) + 1 for l in batch_l],
                                       device=self.clm.device)
                context = self.clm.encode_context(batch)
                loss = (self.clm(batch_enc, context, batch_l, return_loss=True)
                        .sum(dim=1) / lengths).mean()
                loss.backward()
                optimizer.step()

                if p.tick() % log_every == 0:
                    print('Epoch {} batch {}: loss = {:.3f}, {}'
                          .format(e, i, loss.item(), p.format()))

                tracker.step()
                tracker.add_scalar('train/loss', loss.item())

                if tracker.current_step % validate_every == 0:
                    last_val, val_acc = val_acc, self.validate(val_set[:val_examples], tracker)
                    if val_acc < last_val:
                        pass # TODO: Remove some items from S based on confusion matrix?

            tracker.checkpoint()
        self.validate(val_set[:val_examples], tracker)

    def validate(self, examples, tracker):
        val_acc = self.compute_accuracy(examples)
        tracker.add_scalar('val/acc', val_acc)
        print('Validation accuracy: {:.2f}%'.format(100*val_acc))
        tracker.checkpoint()
        return val_acc

class DiscriminativeLanguageAbbreviator:
    def __init__(self, dlm, abbreviation_targets, params={}, training_set=[]):
        self.params = params
        self.dlm = dlm
        self.abbreviation_targets = abbreviation_targets
        self.abbreviation_targets_set = set(abbreviation_targets)
        self.inverted_abbreviations = collections.defaultdict(list)

        for t in abbreviation_targets:
            self.inverted_abbreviations[t[0]].append(t)

        self.evaluation_examples = []
        self.training_set = training_set
        self._load_params()

    def _load_params(self):
        self.description = self.params.get('description') or 'DLA'
        self.minimum_validation_accuracy = self.params.get('minimum_validation_accuracy') or 0.8
        self.val_examples = self.params.get('val_examples') or 64
        self.rehearsal_examples = self.params.get('rehearsal_examples') or 64
        self.batch_size = self.params.get('batch_size') or 128
        self.beam_size = self.params.get('beam_size') or 32
        self.max_abbrev_len = self.params.get('max_abbrev_len') or 1
        self.lambd = self.params.get('lambd') or 0.5

    def list_candidate_expansions(self, token):
        candidates = [token]
        for t in self.abbreviation_targets:
            if t != token and is_subsequence(token, t):
                candidates.append(t)
        return candidates

    def encode_tokens(self, tokens):
        ts = []
        for t in tokens:
            if t in self.abbreviation_targets_set:
                ts.append(t[0])
            else:
                ts.append(t)
        return ts

    def encode(self, batch):
        encoded_l = [{ **r,
                       'l': ''.join(self.encode_tokens(split_at_identifier_boundaries(r['l']))) }
                     for r in batch]

        return encoded_l, None

    def compute_accuracy(self, Xy):
        examples = [x for x, y in Xy]
        queries = [{ **ex,
                     'l': self.abbreviate(ex['l'], self.abbreviation_targets) }
                     for ex in examples]
        predictions = self.beam_search(queries, beam_size=2)
        correct = [r['l'] == p[0] for r, p in zip(examples, predictions)]
        return sum(correct) / len(examples)
 
    def beam_search(self, queries, beam_size=None):
        with torch.no_grad():
            beam_size = beam_size or self.beam_size
            tokens = [split_at_identifier_boundaries(r['l']) for r in queries]
            candidates = [[()] for _ in queries]

            for t_i in range(max(len(t) for t in tokens)):
                need_reranking = []
                reranking_list = []

                for i in range(len(queries)):
                    if t_i < len(tokens[i]):
                        t = tokens[i][t_i]
                        expansions = self.list_candidate_expansions(t)
                        next_candidates = []

                        for c in candidates[i]:
                            for e in expansions:
                                next_candidates.append(c + (e,))

                        candidates[i] = next_candidates

                    if len(candidates[i]) > beam_size or \
                            (len(candidates[i]) > 1 and t_i + 1 == len(tokens[i])):
                        need_reranking.append((
                            i, 
                            len(reranking_list),
                            len(reranking_list) + len(candidates[i])))
                        reranking_list.extend({ 'l': ''.join(c), 
                                                's': queries[i]['l'],
                                                'i': queries[i]['i'],
                                                'c': queries[i]['c'] }
                                                for c in candidates[i])

                if len(need_reranking) > 0:
                    scores = []
                    for batch in batched(reranking_list, self.batch_size):
                        encoded, ctx = self.dlm.encode(batch)
                        scores.extend(list(self.dlm(encoded, ctx)))

                    for i, lo, hi in need_reranking:
                        candidates_with_scores = list(zip(reranking_list[lo:hi],
                                                          scores[lo:hi]))
                        candidates_with_scores.sort(key=lambda cs: cs[1], reverse=True)
                        candidates[i] = [split_at_identifier_boundaries(c['l'])
                                         for c, _ in candidates_with_scores][:beam_size]

            return [[''.join(c) for c in candidates_i] for candidates_i in candidates]

    def decode(self, batch_encoded, ctx=None):
        return [r[0] for r in self.beam_search(batch_encoded)]

    def name(self):
        return ('DiscriminativeLanguageAbbreviator({}, min_val_acc={:.2f}, rehearsal_examples={})'
                .format(self.description,
                        self.minimum_validation_accuracy,
                        self.rehearsal_examples))

    def find_token(self, s, t):
        return t in split_at_identifier_boundaries(s)

    def find_abbreviation(self, string):
        examples = []

        for s in self.training_set:
            if self.find_token(s['l'], string) != -1:
                examples.append(s)
            if len(examples) == self.val_examples:
                break

        if len(examples) < self.val_examples:
            return string

        accuracy = self.evaluate_new_abbreviation(string, examples)

        if accuracy >= self.minimum_validation_accuracy:
            self.abbreviation_targets.append(string)
            self.abbreviation_targets_set.add(string)
            self.inverted_abbreviations[string[0]].append(string)
            self.evaluation_examples.extend(examples)
            return string[0]

        return string

    def evaluate_new_abbreviation(self, string, val_set):
        rehearsal = random.sample(self.evaluation_examples,
                                  k=min(len(self.evaluation_examples),
                                            self.rehearsal_examples))
        examples = rehearsal + val_set

        self.abbreviation_targets.append(string)
        self.abbreviation_targets_set.add(string)
        self.inverted_abbreviations[string[0]].append(string)

        encoded_examples, _ = self.encode(examples)
        decoded_examples = self.decode(encoded_examples)

        correct = [dec == original['l'] for dec, original in zip(decoded_examples, examples)]
        correct_val = correct[:len(rehearsal)]
        correct_new = correct[len(rehearsal):]

        accuracy_val = np.mean(correct_val) if len(correct_val) else 1.0
        accuracy_new = np.mean(correct_new)

        self.abbreviation_targets.pop()
        self.abbreviation_targets_set.remove(string)
        self.inverted_abbreviations[string[0]].pop()

        print('Accuracy with abbreviation {} => {} on {} examples: {:.2f}% val, {:.2f}% positive'
              .format(
                  string,
                  string[0],
                  len(examples),
                  100*accuracy_val,
                  100*accuracy_new))

        return min(accuracy_val, accuracy_new)

    def dump(self, path):
        torch.save({
            'params': self.params,
            'abbreviation_targets': self.abbreviation_targets,
            'dlm': self.dlm.state_dict(),
            }, path)

    @staticmethod
    def load(path, device):
        m = torch.load(path, map_location=device)
        params = m['params']
        dlm = DiscriminativeLanguageModel(params['dlm'], device)
        dlm.load_state_dict(m['dlm'])
        abbreviator = DiscriminativeLanguageAbbreviator(
                dlm, m['abbreviation_targets'], params)
        return abbreviator

    def random_abbreviate(self, line, targets_set):
        tokens = split_at_identifier_boundaries(line)
        abbrev_tokens = []

        for t in tokens:
            if t in targets_set:
                abbrev_tokens.append(t[:1])
            else:
                abbrev_tokens.append(t)

        return ''.join(abbrev_tokens)

    def generate_contrastive_examples(self, N, training_set, use_prefixes=True):
        positive_examples = random.sample(training_set, 2*N)
        examples = []
        examples_used, correct = 0, 0

        for ex in random.sample(training_set, min(2*N, len(training_set))):
            abbrev = self.random_abbreviate(ex['l'], self.abbreviation_targets)
            if abbrev != ex['l']:
                examples.append((ex, abbrev))
                if len(examples) == N:
                    break

        predictions = self.beam_search([{**ex, 'l': ab } for ex, ab in examples], 2)
        contrastive_examples = []

        for p, (ex, abbrev) in zip(predictions, examples):
            correct += p[0] == ex['l']
            negative = p[0 if p[0] != ex['l'] else 1]
            tokens_pos = split_at_identifier_boundaries(ex['l'])
            tokens_neg = split_at_identifier_boundaries(negative)
            different = False

            for i in range(len(tokens_pos)):
                if use_prefixes or i + 1 == len(tokens_pos):
                    different = different or tokens_pos[i] != tokens_neg[i]
                    if different:
                        contrastive_examples.append(
                                ({ **ex, 's': abbrev, 'l': ''.join(tokens_pos[:i+1]) }, 1))
                        contrastive_examples.append(
                                ({ **ex, 's': abbrev, 'l': ''.join(tokens_neg[:i+1]) }, 0))

        return contrastive_examples, correct / len(examples)

    def fit(self, tracker, training_set):
        learning_rate = self.params.get('learning_rate', 1e-3)
        momentum = self.params.get('momentum', 0.9)
        batch_size = self.params.get('batch_size') or 32
        init_scale = self.params.get('init_scale') or 0.1
        epochs = self.params.get('epochs', 1)
        eras = self.params.get('eras', 1)
        epoch_size = self.params.get('epoch_size', 10**4)
        min_epoch_size = self.params.get('min_epoch_size', batch_size)
        log_every = self.params.get('log_every') or 100
        accumulate_examples = self.params.get('accumulate_examples', True)
        epoch_size_multiplier = self.params.get('epoch_size_multiplier', 1.0)
        use_prefixes = self.params.get('use_prefixes', True)
        minimum_epoch_accuracy = self.params.get('minimum_epoch_accuracy', 0.0)

        active_training_set = []

        batches_per_epoch = math.ceil(epoch_size / batch_size)
        total_batches = epochs * batches_per_epoch * eras
        previous_examples = []

        for p in self.dlm.parameters():
            if len(p.shape) >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p, -init_scale, init_scale)

        p = Progress(total_iterations=total_batches if epochs else None)

        for e in range(epochs + 1):
            optimizer = torch.optim.Adam(self.dlm.parameters(),
                                         lr=learning_rate)

            print('Generating {} contrastive examples...'.format(epoch_size))
            before = time.time()
            active_training_set, acc = self.generate_contrastive_examples(
                    epoch_size, training_set, use_prefixes)

            print('Done in {:.1f}s. Accuracy: {:.2f}%'
                  .format(time.time() - before, 100*acc))
            tracker.add_scalar('train/epoch_acc', acc)

            if e == epochs:
                break

            era_training_set = (active_training_set + previous_examples)
            era = 0

            while era < eras or acc < minimum_epoch_accuracy:
                random.shuffle(era_training_set)

                for i, batch in enumerate(batched(era_training_set, batch_size)):
                    optimizer.zero_grad()
                    X, y = zip(*batch)
                    X_i, X_c = self.dlm.encode(X)
                    y_hat = self.dlm(X_i, X_c)
                    y_true = torch.tensor(y, dtype=torch.float).to(self.dlm.device)

                    loss = F.binary_cross_entropy(y_hat, y_true)
                    loss.backward()
                    optimizer.step()

                    acc = (y_hat.round() == y_true).float().mean().item()

                    if p.tick() % log_every == 0:
                        print('Epoch {}/{} batch {}: loss = {:.3f}, acc = {:.2f}%, {}'.format(e, era, i, loss.item(), 100*acc, p.format()))

                    tracker.step()
                    tracker.add_scalar('train/loss', loss.item())

                if minimum_epoch_accuracy > 0:
                    examples, acc = self.generate_contrastive_examples(
                            epoch_size,
                            [x for x, y in active_training_set if y == 1],
                            use_prefixes
                            )
                    era_training_set = examples + previous_examples
                    print('Accuracy after era {}: {:.2f}%'.format(era, 100*acc))
                    tracker.add_scalar('train/epoch_acc', acc)

                era += 1

            if accumulate_examples:
                previous_examples.extend(active_training_set)

            tracker.checkpoint()
            epoch_size = max(min_epoch_size, int(epoch_size * epoch_size_multiplier))

class AbbreviatorEvaluator:
    def __init__(self, common_strings, evaluation_set, batch_size=64):
        self.common_strings = common_strings
        self.batch_size = batch_size
        self.evaluation_set = evaluation_set

    def evaluate(self, abbreviator, progress=None, tracker=None):
        cases, successes, total_len, compressed_len = 0, 0, 0, 0
        abbreviations = []

        for s in self.common_strings:
            abbreviation = abbreviator.find_abbreviation(s)
            cases += 1
            successes += len(abbreviation) < len(s)
            total_len += len(s)
            compressed_len += len(abbreviation)
            abbreviations.append({'short': abbreviation, 'long': s})

            if tracker:
                tracker.add_scalar('abbreviation_success', len(abbreviation) < len(s))
                tracker.add_scalar('abbreviation_success_freq', successes / cases)
                tracker.add_scalar('abbreviation_compression', 1 - (compressed_len / total_len))
                tracker.step()

            if progress is not None:
                progress.tick()

        eval_successes, eval_len, eval_compressed_len, eval_examples = 0, 0, 0, []

        for batch in batched(self.evaluation_set, self.batch_size):
            encoded, ctx = abbreviator.encode(batch)
            encoded_l = [r['l'] for r in encoded]
            decoded = abbreviator.decode(encoded, ctx)
            eval_examples.extend(zip(batch, encoded_l, decoded))
            eval_len += sum(len(s['l']) for s in batch)
            eval_compressed_len += sum(len(s) for s in encoded_l)
            eval_successes += sum(o['l'] == d for o, d in zip(batch, decoded))

        scalar_results = {
            'accuracy': eval_successes / len(self.evaluation_set),
            'eval_compression': 1 - (eval_compressed_len / eval_len),
            'abbreviation_success_rate': successes / cases,
            'abbreviation_compression': 1 - (compressed_len / total_len),
        }

        if tracker:
            for k, v in scalar_results.items():
                tracker.report_scalar(k, v)
            tracker.extend_list('eval_examples', eval_examples)
            tracker.extend_list('abbreviations', abbreviations)

        return {
            **scalar_results,
            'eval_examples': eval_examples,
            'abbreviations': abbreviations,
        }

def split_tokens(s):
    '''Splits the string s into identifiers and non-identifiers.
    Identifiers always come into odd positions, and non-identifiers
    in even positions.
    '''

    t, last_t = [], []
    in_id = False
    for c in s:
        if c.isidentifier() or (in_id and c.isalnum()):
            if in_id:
                last_t.append(c)
            else:
                t.append(''.join(last_t))
                in_id = True
                last_t = [c]
        else:
            if in_id:
                t.append(''.join(last_t))
                in_id = False
                last_t = [c]
            else:
                last_t.append(c)
    if len(last_t):
        t.append(''.join(last_t))
    return t

def random_abbreviate(tokens, prefix_table):
    new_tokens = []

    for t in tokens:
        if t in prefix_table and random.randint(0, 1):
            new_tokens.append(random.choice(prefix_table[t]))
        else:
            new_tokens.append(t)

    return new_tokens
