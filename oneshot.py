# One-shot learning algorithms.

import random
import numpy as np
import torch
import torch.nn.functional as F

from util import batched, Progress
from user import User

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

    def evaluate(self, learner):
        raise NotImplemented()

class LearnEvalIterate(OneShotEvaluator):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def name(self):
        return "LearnEvalIterate"

    def evaluate(self, learner, dataset):
        u = User(conventions_queue=[(row['string'], row['abbreviation']) for row in dataset])

        accuracies = []
        accuracies_positive, accuracies_negative = [], []
        p = Progress(len(dataset))

        for row in dataset:
            u.add_next_convention()

            training_example = random.choice(row['positive_examples'])
            learner.learn((u.encode(training_example), training_example))

            test_positive = list(set(s for s in row['positive_examples'] if s != training_example))
            test_negative = list(set(row['negative_examples']))

            correct_positive = []

            for batch in batched(test_positive, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_positive.extend([int(p == s) for p, s in zip(batch, learner_prediction)])

            correct_negative = []

            for batch in batched(test_negative, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_negative.extend([int(p == s) for p, s in zip(batch, learner_prediction)])

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
        }

class LearnAllThenEval(OneShotEvaluator):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def name(self):
        return "LearnAllThenEval"

    def evaluate(self, learner, dataset):
        u = User(conventions_queue=[(row['string'], row['abbreviation']) for row in dataset])

        accuracies = []
        accuracies_positive, accuracies_negative = [], []
        train = []
        p = Progress(len(dataset))

        for row in dataset:
            u.add_next_convention()

            training_example = random.choice(row['positive_examples'])
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

            correct_negative = []

            for batch in batched(test_negative, self.batch_size):
                encoded_batch = [u.encode(s) for s in batch]
                learner_prediction = learner.test(encoded_batch)
                correct_negative.extend([int(p == s) for p, s in zip(batch, learner_prediction)])

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

class StepUntilCorrect(OneShotLearner):
    def __init__(self, prior_decoder, alphabet, parameters={}):
        self.decoder = prior_decoder.clone(alphabet)
        self.alphabet = alphabet

        self.batch_size = parameters.get('batch_size') or 64
        self.learning_rate = parameters.get('learning_rate') or 1e-2
        self.max_steps = parameters.get('max_steps') or 8
        self.extra_steps = parameters.get('extra_steps') or 0

        self.optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

    def name(self):
        return 'StepUntilCorrect(lr={}, max={}, extra={})'.format(self.learning_rate,
                                                                  self.max_steps,
                                                                  self.extra_steps)

    def learn(self, example):
        short, long = example

        correct_since = self.max_steps

        for i in range(self.max_steps):
            self.decoder.eval()
            prediction = self.decoder([short], self.alphabet)[0]

            if prediction == long:
                correct_since = i

            if i >= correct_since + self.extra_steps:
                break

            self.decoder.train()
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
