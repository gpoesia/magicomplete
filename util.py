# Various utilities.

import argparse
import datetime
import time
import numpy as np
from data import load_dataset
import user
import json
import random

class Progress:
    def __init__(self, total_iterations=None, timeout=None):
        self.total_iterations = total_iterations
        self.timeout = timeout
        self.begin = time.time()
        self.current_iteration = 0

    def tick(self):
        self.current_iteration += 1

    def format(self):
        now = time.time()
        elapsed = now - self.begin

        if self.total_iterations is not None:
            return format_eta(
                self.current_iteration,
                self.current_iteration / self.total_iterations,
                elapsed,
                (self.total_iterations - self.current_iteration) *
                (elapsed / max(1, self.current_iteration)))

        if self.timeout is not None:
            return format_eta(
                self.current_iteration,
                (now - self.begin) / self.timeout,
                elapsed,
                max(0, self.timeout - (now - self.begin)))

        return "???"

def format_eta(current_iteration, complete_fraction, elapsed, remaining):
    remaining = int(remaining)

    hours = remaining // (60*60)
    minutes = (remaining // 60) % 60
    seconds = remaining % 60

    t = ""

    if hours > 0:
        t += "{:02}h".format(hours)

    if minutes > 0 or hours > 0:
        t += "{:02}m".format(minutes)

    t += "{:02}s".format(seconds)

    return "{:.2f}% done, {} left, {:.1f} it/s".format(100*complete_fraction, t, current_iteration / elapsed)

def rolling_average(seq, w=500):
    s = np.cumsum(np.array(seq))
    return (s[w:] - s[:-w]) / w

def precompute_interactions(dataset, language, new_convention_every, one_convention=False):
    dataset = load_dataset(dataset)[language]['train']
    random.shuffle(dataset)

    u = user.User()
    events = []
    p = Progress(len(dataset))

    for i, l in enumerate(dataset):
        enc, conventions = u.encode(l, trace_conventions=True, one_convention=one_convention)
        events.append({'type': 'user_input', 'long': l, 'short': enc, 'conventions': conventions})

        u.remember_substrings(l)
        p.tick()

        if (i + 1) % new_convention_every == 0:
            s, c = u.form_new_convention()
            events.append({'type': 'convention', 'long': s, 'short': c})
            print(p.format())

    with open('interactions.json', 'w') as f:
        json.dump(events, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConadComplete utilities')

    parser.add_argument('--seed', default='conadcomplete', help='Random seed')
    parser.add_argument('--language', default='Python', help='Programming Language to use (Python|Haskell|Java).')
    parser.add_argument('--dataset', default='medium', help='Dataset to use')
    parser.add_argument('--new-convention-every', default=100, type=int, help='Iterations between new conventions')
    parser.add_argument('--precompute-interactions', action='store_const', const=True, default=False)
    parser.add_argument('--one-convention', help='Limit to applying at most one convention per input.',
                        action='store_const', const=True, default=False)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.precompute_interactions:
        precompute_interactions(args.dataset, args.language, args.new_convention_every, args.one_convention)
