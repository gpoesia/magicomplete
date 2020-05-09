# Train an end-to-end model

import random
import itertools
import time
import math
import json
from decoder import Context

import torch
import torch.nn.functional as F
from util import Progress, batched

def train(encoder,
          decoder,
          set_embedding,
          dataset,
          parameters,
          device,
          output_path):
    '''Trains the end-to-end model using the specified parameters.'''
    torch.manual_seed(0)

    learning_rate = parameters.get('learning_rate') or 1e-2
    batch_size = parameters.get('batch_size') or 32
    init_scale = parameters.get('init_scale') or 0.1
    timeout = parameters.get('timeout')
    epochs = None if timeout in parameters else (parameters.get('epochs') or 1)
    verbose = parameters.get('verbose') or False
    log_every = parameters.get('log_every') or 100
    lr_decay_step_size = parameters.get('lr_decay_step_size') or 4
    lr_decay_gamma = parameters.get('lr_decay_gamma') or 0.1
    context = Context(parameters.get('context') or 0)

    training_set = dataset['train']
    train_losses = []
    best_loss = math.inf

    log = print if verbose else lambda *args: None
    decoder.to(device)
    decoder.train()

    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler_dec = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=lr_decay_step_size, gamma=lr_decay_gamma)

    for p in decoder.parameters():
        p.data.uniform_(-init_scale, init_scale)

    examples_processed = 0
    total_batches = epochs * math.ceil(len(training_set) / batch_size) if epochs else math.inf

    p = Progress(
            total_iterations=total_batches if epochs else None,
            timeout=timeout)

    e = 0

    random.shuffle(training_set)

    while not p.timed_out() if timeout else e < epochs:
        for i, batch in enumerate(batched(training_set, batch_size)):
            if p.timed_out():
                break

            batch_l, batch_c, batch_i = zip(*[(r['l'], r['c'], r['i'])
                                              for r in batch])
            ctx = decoder.compute_context(batch_i, batch_c, set_embedding)
            lengths = torch.tensor([len(i) for i in batch_l], device=device)
            optimizer_dec.zero_grad()

            encoded_batch = encoder.encode_batch(batch_l)
            per_prediction_loss = decoder(
                    compressed=encoded_batch,
                    context=ctx,
                    expected=batch_l)
            loss = (per_prediction_loss.sum(dim=1)/(lengths+1)).mean()
            loss.backward()
            train_losses.append(loss.item())

            optimizer_dec.step()

            if p.tick() % log_every == 0:
                log('Epoch {} batch {}: loss = {:.3f}, {}'
                    .format(e, i, train_losses[-1], p.format()))
        e += 1
        scheduler_dec.step()

        if (train_losses[-1] < best_loss):
            decoder.dump(output_path)
            best_loss = train_losses[-1]

    return train_losses
