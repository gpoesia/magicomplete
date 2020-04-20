# System that takes a set of common strings and creates reliable abbreviations from them.

import random
from user import User
import torch
import numpy as np
from util import batched

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
    def __init__(self, decoder, alphabet, training_set, parameters={}):
        self.decoder = decoder.clone(alphabet)
        self.alphabet = alphabet
        self.training_set = training_set
        self.user = User()

        self.minimum_validation_accuracy = parameters.get('minimum_validation_accuracy') or 0.8
        self.minimum_train_examples = parameters.get('minimum_train_examples') or 64
        self.val_examples = parameters.get('val_examples') or 32
        self.learning_rate = parameters.get('learning_rate') or 0.1
        self.max_steps = parameters.get('max_steps') or 100
        self.batch_size = parameters.get('batch_size') or 32
        self.rehearsal_batch_size = parameters.get('rehearsal_batch_size', 32)
        abbreviation_algorithm = parameters.get('abbreviation_algorithm') or 'erase_suffixes'

        assert abbreviation_algorithm in ('erase_suffixes'), 'Unknown abbreviation algorithm'

        self.abbreviation_algorithm = {
            'erase_suffixes': EraseSuffixes
            }[abbreviation_algorithm]()

        self.rehearsal_examples = []

    def encode(self, batch):
        return [self.user.encode(s) for s in batch]

    def decode(self, batch):
        return self.decoder(batch, self.alphabet)

    def name(self):
        return ('LanguageAbbreviator(min_val_acc={:.2f}, min_train={}, val_ex={}, lr={}, max_steps={}, rehearsal={})'
                .format(self.minimum_validation_accuracy,
                        self.minimum_train_examples,
                        self.val_examples,
                        self.learning_rate,
                        self.max_steps,
                        self.rehearsal_batch_size))

    def find_abbreviation(self, string):
        abbreviation = string
        validation_accuracy = 1.0

        examples = list({s for s in self.training_set if s.find(string) != -1})

        if len(examples) < self.minimum_train_examples + self.val_examples:
            print('Not enough examples for', string)
            return string

        train, val = examples[:-self.val_examples], examples[-self.val_examples:]
        initial = self.try_learn_abbreviation(string, abbreviation, train, val)

        alternatives = [abbreviation]

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
        new_decoder = self.decoder.clone(self.alphabet)
        optimizer = torch.optim.SGD(new_decoder.parameters(), lr=self.learning_rate)

        encode = lambda s: self.user.encode(s).replace(string, abbreviation)

        for i in range(self.max_steps):
            new_decoder.train()
            optimizer.zero_grad()

            batch = random.sample(training_set, self.batch_size)

            batch.extend(random.sample(self.rehearsal_examples, min(len(self.rehearsal_examples),
                                                                        self.rehearsal_batch_size)))

            batch_encoded = [encode(s) for s in batch]

            loss = new_decoder(batch_encoded, self.alphabet, batch).mean()
            loss.backward()

            optimizer.step()

            new_decoder.eval()
            val_predictions = []

            for val_batch in batched(val_set, self.batch_size):
                predictions = new_decoder([encode(s) for s in val_set], self.alphabet)
                val_predictions.extend(p == s for p, s in zip(predictions, val_batch))

            val_acc = np.mean(val_predictions)

            if val_acc >= self.minimum_validation_accuracy:
                self.decoder = new_decoder
                return True

        return False

class AbbreviatorEvaluator:
    def __init__(self, common_strings, evaluation_set, batch_size=64):
        self.common_strings = common_strings
        self.batch_size = batch_size
        self.evaluation_set = evaluation_set

    def evaluate(self, abbreviator, progress=None):
        cases, successes, total_len, compressed_len = 0, 0, 0, 0
        abbreviations = []

        for s in self.common_strings:
            abbreviation = abbreviator.find_abbreviation(s)
            cases += 1
            successes += len(abbreviation) < len(s)
            total_len += len(s)
            compressed_len += len(abbreviation)
            if len(abbreviation) < len(s):
                abbreviations.append({'short': abbreviation, 'long': s})

            if progress is not None:
                progress.tick()

        eval_successes, eval_len, eval_compressed_len, eval_examples = 0, 0, 0, []

        for batch in batched(self.evaluation_set, self.batch_size):
            encoded = abbreviator.encode(batch)
            decoded = abbreviator.decode(encoded)
            eval_examples.extend(zip(batch, encoded, decoded))
            eval_len += sum(len(s) for s in batch)
            eval_compressed_len += sum(len(s) for s in encoded)
            eval_successes += sum(o == d for o, d in zip(batch, decoded))

        return {
            'accuracy': eval_successes / len(self.evaluation_set),
            'eval_compression': 1 - (eval_compressed_len / eval_len),
            'eval_examples': eval_examples,
            'abbreviation_success_rate': successes / cases,
            'abbreviations': abbreviations,
            'abbreviation_compression': 1 - (compressed_len / total_len),
        }
