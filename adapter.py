# Implements continual adaptation update rules given a prior model.

import torch
import torch.nn.functional as F
import collections
import random

InteractionRecord = collections.namedtuple('InteractionLog', ('line', 'encoded', 'decoded'))

class ModelAdapter:
    def __init__(self,
                 user,
                 prior_encoder,
                 prior_decoder,
                 alphabet,
                 prior_dataset,
                 parameters={}):
        self.user = user
        self.prior_dataset = prior_dataset
        self.prior_encoder = prior_encoder
        self.prior_decoder = prior_decoder.clone(alphabet)
        self.adapted_decoder = prior_decoder.clone(alphabet)
        self.alphabet = alphabet

        self.interaction_history = []

        self.learning_rate = parameters.get('learning_rate') or 1e-4

        self.enable_adaptation = parameters.get('enable_adaptation') or False
        self.enable_divergence_prior = parameters.get('enable_divergence_prior') or False
        self.divergence_prior_weight = parameters.get('divergence_prior_weight') or 1e-4
        self.divergence_prior_batch_size = parameters.get('divergence_prior_batch_size') or 64

        self.enable_local_rehearsal = parameters.get('enable_local_rehearsal') or False
        self.local_rehearsal_weight = parameters.get('local_rehearsal_weight') or 1e-4
        self.local_rehearsal_batch_size = parameters.get('local_rehearsal_batch_size') or 64

        self.optimizer = torch.optim.SGD(self.adapted_decoder.parameters(), lr=self.learning_rate)

    def run_on_example(self, example, encoded=None):
        batch = [example]

        if encoded is None:
            encoded = self.user.encode(example)

        prediction = self.adapted_decoder(compressed=[encoded], alphabet=self.alphabet)[0]
        self.interaction_history.append(InteractionRecord(example, encoded, prediction))

        if self.enable_adaptation:
            self.optimizer.zero_grad()
            loss = self.adapted_decoder(compressed=[encoded], alphabet=self.alphabet, expected=batch).mean()

            if self.enable_divergence_prior:
                loss += self.divergence_prior()

            if self.enable_local_rehearsal and \
               len(self.interaction_history) >= self.local_rehearsal_batch_size:
                loss += self.local_rehearsal()

            loss.backward()
            self.optimizer.step()

        return example == prediction

    def divergence_prior(self):
        batch = random.sample(self.prior_dataset, self.divergence_prior_batch_size)
        encoded_batch = self.prior_encoder.encode_batch(batch)

        with torch.no_grad():
            prior_probabilities = self.prior_decoder(encoded_batch, self.alphabet, batch, return_loss=False)

        adapted_probabilities = self.adapted_decoder(encoded_batch, self.alphabet, batch, return_loss=False)

        return F.kl_div(prior_probabilities, adapted_probabilities) * self.divergence_prior_weight

    def local_rehearsal(self):
        batch, encoded_batch, _ = zip(*random.sample(self.interaction_history,
                                                     self.local_rehearsal_batch_size))

        loss = self.adapted_decoder(encoded_batch, self.alphabet, batch).mean()

        return loss * self.local_rehearsal_weight
