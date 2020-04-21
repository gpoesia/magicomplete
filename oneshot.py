# One-shot learning algorithms.

import random
import numpy as np
import torch
import torch.nn.functional as F

from util import batched, Progress
from user import User
from data import augment

class OneShotLearner:
    def name(self):
        raise NotImplemented()

    def learn(self, example):
        pass

    def test(self, examples):
        raise NotImplemented()

class OneShotEvaluator:
    def name(self):
        raise NotImplemented()

    def evaluate(self, learner, dataset, save_examples=False):
        raise NotImplemented()

class LearnEvalIterate(OneShotEvaluator):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def name(self):
        return "LearnEvalIterate"

    def evaluate(self, learner, dataset, save_examples=False):
        u = User(conventions_queue=[(row['string'], row['abbreviation']) for row in dataset])

        accuracies = []
        accuracies_positive, accuracies_negative = [], []
        examples = []
        p = Progress(len(dataset))

        for row in dataset:
            u.add_next_convention()

            training_example = row['positive_examples'][0]
            learner.learn((u.encode(training_example), training_example))

            test_positive = list(set(s for s in row['positive_examples'] if s != training_example))
            test_negative = list(set(row['negative_examples']))

            correct_positive = []

            for batch in batched(test_positive, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_positive.extend([int(p == s) for p, s in zip(batch, learner_prediction)])
                if save_examples:
                    examples.extend([{'long': l, 'short': s, 'prediction': p}
                                     for l, s, p in zip(batch, encoded_batch, learner_prediction)])

            correct_negative = []

            for batch in batched(test_negative, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_negative.extend([int(p == s) for p, s in zip(batch, learner_prediction)])
                if save_examples:
                    examples.extend([{'long': l, 'short': s, 'prediction': p}
                                     for l, s, p in zip(batch, encoded_batch, learner_prediction)])

            accuracies.append(np.mean(correct_positive + correct_negative))
            accuracies_positive.append(np.mean(correct_positive))
            accuracies_negative.append(np.mean(correct_negative))

        return {
            'accuracy': np.mean(accuracies),
            'accuracy_positive': np.mean(accuracies_positive),
            'accuracy_negative': np.mean(accuracies_negative),
            'per_example_accuracy': accuracies,
            'per_example_accuracy_positive': accuracies_positive,
            'per_example_accuracy_negative': accuracies_negative,
            'examples': examples,
        }

class LearnAllThenEval(OneShotEvaluator):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def name(self):
        return "LearnAllThenEval"

    def evaluate(self, learner, dataset, save_examples=False):
        u = User(conventions_queue=[(row['string'], row['abbreviation']) for row in dataset])

        accuracies = []
        accuracies_positive, accuracies_negative = [], []
        train = []
        examples = []
        p = Progress(len(dataset))

        for row in dataset:
            u.add_next_convention()

            training_example = row['positive_examples'][0]
            train.append(training_example)
            learner.learn((u.encode(training_example), training_example))

        for row, training_example in zip(dataset, train):
            test_positive = list(set(s for s in row['positive_examples'] if s != training_example))
            test_negative = list(set(row['negative_examples']))

            correct_positive = []

            for batch in batched(test_positive, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_positive.extend([int(p == s) for p, s in zip(batch, learner_prediction)])
                if save_examples:
                    examples.extend([{'long': l, 'short': s, 'prediction': p}
                                     for l, s, p in zip(batch, encoded_batch, learner_prediction)])

            correct_negative = []

            for batch in batched(test_negative, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_negative.extend([int(p == s) for p, s in zip(batch, learner_prediction)])
                if save_examples:
                    examples.extend([{'long': l, 'short': s, 'prediction': p}
                                     for l, s, p in zip(batch, encoded_batch, learner_prediction)])

            accuracies.append(np.mean(correct_positive + correct_negative))
            accuracies_positive.append(np.mean(correct_positive))
            accuracies_negative.append(np.mean(correct_negative))

        return {
            'accuracy': np.mean(accuracies),
            'accuracy_positive': np.mean(accuracies_positive),
            'accuracy_negative': np.mean(accuracies_negative),
            'per_example_accuracy': accuracies,
            'per_example_accuracy_positive': accuracies_positive,
            'per_example_accuracy_negative': accuracies_negative,
            'examples': examples,
        }


class PriorBaseline(OneShotLearner):
    def __init__(self, prior_decoder, alphabet, parameters={}):
        self.decoder = prior_decoder.clone(alphabet)
        self.alphabet = alphabet
        self.decoder.eval()

        self.batch_size = parameters.get('batch_size') or 64

    def name(self):
        return 'PriorBaseline'

    def learn(self, example):
        pass

    def test(self, examples):
        results = []

        for batch in batched(examples, self.batch_size):
            results.extend(self.decoder(batch, self.alphabet))

        return results

class KGradientSteps(OneShotLearner):
    def __init__(self, prior_decoder, alphabet, parameters={}):
        self.decoder = prior_decoder.clone(alphabet)
        self.alphabet = alphabet
        self.decoder.eval()

        self.batch_size = parameters.get('batch_size') or 64
        self.learning_rate = parameters.get('learning_rate') or 1e-2
        self.k = parameters.get('k') or 1

        self.optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

    def name(self):
        return 'KGradientSteps(K={}, lr={})'.format(self.k, self.learning_rate)

    def learn(self, example):
        self.decoder.train()

        short, long = example

        for i in range(self.k):
            self.optimizer.zero_grad()
            loss = self.decoder([short], self.alphabet, [long]).mean()
            loss.backward()
            self.optimizer.step()

    def test(self, examples):
        self.decoder.eval()

        results = []

        for batch in batched(examples, self.batch_size):
            results.extend(self.decoder(batch, self.alphabet))

        return results

def infer_abbreviation(short, long):
    # FIXME: This doesn't really disambiguate between all the possible abbreviations...
    # That doesn't seem possible to do deterministically with a single example.
    for prefix in range(len(short)):
        if short[prefix] != long[prefix]:
            break
        prefix += 1

    for suffix in range(len(short)):
        if short[-1 - suffix] != long[-1 - suffix]:
            break
        suffix += 1

    return (short[prefix:-suffix], long[prefix:-suffix])

class StepUntilCorrect(OneShotLearner):
    def __init__(self, prior_decoder, alphabet,  parameters={}, augmentation_dataset=[]):
        self.decoder = prior_decoder.clone(alphabet)
        self.alphabet = alphabet

        self.batch_size = parameters.get('batch_size') or 64
        self.learning_rate = parameters.get('learning_rate') or 1e-2
        self.max_steps = parameters.get('max_steps') or 8
        self.extra_steps = parameters.get('extra_steps') or 0
        self.data_augmentation = parameters.get('data_augmentation') or None
        self.rehearsal_examples = parameters.get('rehearsal_examples') or 0
        self.augmentation_dataset = augmentation_dataset

        self.optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.learning_rate)
        self.past_examples = []

    def name(self):
        return ('StepUntilCorrect(lr={}, max={}, extra={}, data_augmentation={}, reheasal_examples={})'
                .format(self.learning_rate,
                        self.max_steps,
                        self.extra_steps,
                        self.data_augmentation or 'no',
                        self.rehearsal_examples))

    def fetch_augmentation_examples(self, short, long):
        ab_short, ab_long = infer_abbreviation(short, long)
        for row in self.augmentation_dataset:
            if long in row['positive_examples']:
                return ([(s.replace(row['string'], row['abbreviation']), s)
                         for s in row['positive_examples_train']] +
                        [(s, s) for s in row['negative_examples_train']])

    def trim_examples(self, batch):
        return [batch[0]] + random.sample(batch[1:], min(len(batch) - 1, self.batch_size - 1))

    def learn(self, example):
        short, long = example

        if self.data_augmentation is None:
            batch = [example]
        elif self.data_augmentation in ('ast_only_short', 'ast_all'):
            batch = augment(short, long, only_shortened=(self.data_augmentation == 'only_short'))
        elif self.data_augmentation == 'fetch_examples':
            batch = [example] + self.fetch_augmentation_examples(short, long)

        correct_since = self.max_steps

        for i in range(self.max_steps):
            self.decoder.eval()
            prediction = self.decoder([short], self.alphabet)[0]

            if prediction == long:
                correct_since = i

            if i >= correct_since + self.extra_steps:
                break

            rehearsal_batch = random.sample(self.past_examples,
                                            min(len(self.past_examples), self.rehearsal_examples))

            rehearsal_short, rehearsal_long = (zip(*rehearsal_batch)
                                               if len(rehearsal_batch)
                                               else ((), ()))

            self.decoder.train()
            self.optimizer.zero_grad()

            batch_short, batch_long = zip(*self.trim_examples(batch))

            loss = self.decoder(batch_short + rehearsal_short,
                                self.alphabet,
                                batch_long + rehearsal_long).mean()
            loss.backward()
            self.optimizer.step()

        self.past_examples.extend(batch)

    def test(self, examples):
        self.decoder.eval()

        results = []

        for batch in batched(examples, self.batch_size):
            results.extend(self.decoder(batch, self.alphabet))

        return results
