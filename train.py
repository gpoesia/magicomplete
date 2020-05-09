# Train an end-to-end model

import random
import itertools
import math
import json
import numpy as np
from decoder import Context

import torch
import torch.nn.functional as F
from util import Progress, batched

def encode_dataset(d, encoder, rnd=random):
    return [(encoder.encode(r['l'], rnd), r['l'], r['c'], r['i']) for r in d]

def compute_accuracy(encoder, decoder, set_embedding, dataset, batch_size):
    with torch.no_grad():
        correct = []
        for i, batch in enumerate(batched(encode_dataset(dataset, encoder, random.Random('validation')),
                                          batch_size)):
            batch_e, batch_l, batch_c, batch_i = zip(*batch)
            ctx = decoder.compute_context(batch_i, batch_c, set_embedding)
            predicted = decoder(compressed=batch_e, context=ctx)
            correct.extend((a == b) for a, b in zip(batch_l, predicted))
        return np.mean(correct)

def train(encoder,
          decoder,
          set_embedding,
          dataset,
          parameters,
          device,
          tracker):
    '''Trains the end-to-end model using the specified parameters.'''
    learning_rate = parameters.get('learning_rate') or 1e-2
    batch_size = parameters.get('batch_size') or 32
    init_scale = parameters.get('init_scale') or 0.1
    epochs = parameters.get('epochs', 1)
    verbose = parameters.get('verbose') or False
    log_every = parameters.get('log_every') or 100

    validations_per_epoch = parameters.get('validations_per_epoch') or 10
    max_validation_examples = parameters.get('max_validation_examples') or 10**9
    loss_convergence_window = parameters.get('loss_convergence_window') or None
    loss_convergence_improvement = parameters.get('loss_convergence_improvement') or None

    training_set, validation_set = dataset['train'][:1000], dataset['dev'][:max_validation_examples]

    log = print if verbose else lambda *args: None
    decoder.to(device)
    decoder.train()

    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for p in decoder.parameters():
        p.data.uniform_(-init_scale, init_scale)

    batches_per_epoch = math.ceil(len(training_set) / batch_size)
    total_batches = epochs * batches_per_epoch
    validate_every = max(batches_per_epoch // validations_per_epoch, 1)

    p = Progress(total_iterations=total_batches if epochs else None)
    e = 0

    while e < epochs:
        random.shuffle(training_set)

        for i, batch in enumerate(batched(encode_dataset(training_set, encoder), batch_size)):
            batch_e, batch_l, batch_c, batch_i = zip(*batch)
            ctx = decoder.compute_context(batch_i, batch_c, set_embedding)
            lengths = torch.tensor([len(i) for i in batch_l], device=device)
            optimizer_dec.zero_grad()

            per_prediction_loss = decoder(
                    compressed=batch_e,
                    context=ctx,
                    expected=batch_l)
            loss = (per_prediction_loss.sum(dim=1)/(lengths+1)).mean()
            loss.backward()

            optimizer_dec.step()

            if p.tick() % log_every == 0:
                log('Epoch {} batch {}: loss = {:.3f}, {}'.format(e, i, loss.item(), p.format()))

            tracker.step()
            tracker.add_scalar('train/loss', loss.item())

            if (i + 1) % validate_every == 0:
                tracker.add_scalar('val/accuracy',
                                   compute_accuracy(encoder, decoder, set_embedding,
                                                    validation_set, batch_size))

        if loss_convergence_improvement is not None and \
           tracker.loss_converged('train/loss',
                                  loss_convergence_window,
                                  loss_convergence_improvement):
            print('Loss converged after epoch', e)
            break

        e += 1

    tracker.add_scalar('val/accuracy',
                       compute_accuracy(encoder, decoder, set_embedding, validation_set, batch_size))
