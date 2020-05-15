# System that takes a set of common strings and creates reliable abbreviations from them.

import random
from user import User
import torch
import numpy as np
from util import batched, split_at_identifier_boundaries
import collections
import json

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
        return [self.user.encode(s) for s in batch_l], ctx

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
        self.evaluation_examples = []

    def encode_tokens(self, tokens):
        for i in range(len(tokens)):
            for k, v in self.abbreviation_table.items():
                if i + len(k) <= len(tokens) and tokens[i:i + len(k)] == k:
                    return tokens[:i] + v + self.encode_tokens(tokens[i + len(k):])
        return tokens

    def encode(self, batch):
        encoded_l = [{ **r,
                       'l': ''.join(self.encode_tokens(split_at_identifier_boundaries(r['l']))) }
                     for r in batch]

        return encoded_l, None

    def list_candidates(self, tokens):
        if len(tokens) == 0:
            return [()]

        answer = []
        for c in self.list_candidates(tokens[1:]):
            answer.append((tokens[0],) + c)

        for k, v in self.inverted_abbreviations.get(tokens[0], []):
            if tokens[:len(k)] == k:
                for c in self.list_candidates(tokens[len(k):]):
                    answer.append(v + c)

        return answer

    def decode(self, batch_encoded, idx_ctx=None):
        all_candidates, limits = [], [0]

        for row in batch_encoded:
            l = row['l']
            tokens = tuple(split_at_identifier_boundaries(l))
            candidates = [''.join(c) for c in self.list_candidates(tokens)]
            all_candidates.extend({'l': cand, 'i': row['i'], 'c': row['c']} for cand in candidates)
            limits.append(len(all_candidates))

        perplexities = self.lm.compute_log_perplexity(all_candidates, self.batch_size, False)

        decoded = []

        for i in range(len(batch_encoded)):
            min_idx = min(range(limits[i], limits[i+1]),
                          key=lambda c: perplexities[c])
            decoded.append(all_candidates[min_idx]['l'])

        return decoded

    def compute_accuracy(self, examples):
        encoded = self.encode(examples)

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
        return [s[:l] for l in range(1, len(s) + 1) if s[l-1].isidentifier()]

    def find_abbreviation(self, string):
        string_tokens = split_at_identifier_boundaries(string)
        examples = list(s for (s, t) in zip(self.training_set, self.training_set_tokens)
                        if self.find_tokens(s['l'], string_tokens) != -1)

        if len(examples) < self.val_examples:
            return string

        val = examples[:self.val_examples]

        for abbreviation in self.list_candidate_abbreviations(string):
            abbreviation_tokens = split_at_identifier_boundaries(abbreviation)
            accuracy = self.evaluate_new_abbreviation(string_tokens, abbreviation_tokens, val)

            if accuracy >= self.minimum_validation_accuracy:
                self.abbreviation_table[string_tokens] = abbreviation_tokens
                self.inverted_abbreviations[abbreviation_tokens[0]].append(
                    (abbreviation_tokens, string_tokens))
                self.evaluation_examples.extend(val)
                return abbreviation

        return string

    def evaluate_new_abbreviation(self, string_tokens, abbreviation_tokens, val_set):
        examples = self.evaluation_examples + val_set

        self.abbreviation_table[string_tokens] = abbreviation_tokens
        self.inverted_abbreviations[abbreviation_tokens[0]].append((abbreviation_tokens, string_tokens))

        encoded_examples, _ = self.encode(examples)
        decoded_examples = self.decode(encoded_examples)

        correct = [dec == original['l'] for dec, original in zip(decoded_examples, examples)]
        correct_val = correct[:len(self.evaluation_examples)]
        correct_new = correct[len(self.evaluation_examples):]

        accuracy_val = np.mean(correct_val) if len(correct_val) else 1.0
        accuracy_new = np.mean(correct_new)

        del self.abbreviation_table[string_tokens]
        self.inverted_abbreviations[abbreviation_tokens[0]].pop()

        print('Accuracy with abbreviation = {} on {} examples: {:.2f}% val, {:.2f}% positive'
              .format(
                  ''.join(abbreviation_tokens),
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
