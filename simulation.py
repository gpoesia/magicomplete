# Simulates a user's interaction with an auto-complete system.

import random

def simulate(dataset,
             user,
             decoder,
             alphabet,
             parameters={}):

    new_convention_every = parameters.get('new_convention_every', 100)

    correct = []

    d = dataset[:]
    random.shuffle(d)

    for i, l in enumerate(d):
        enc = user.encode(l)
        dec = decoder([enc], alphabet)

        correct.append(int(dec[0] == l))

        user.remember_substrings(l)

        if (i + 1) % new_convention_every == 0:
            user.form_new_convention()

    return correct
