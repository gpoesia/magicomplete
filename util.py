# Various utilities.

import argparse
import collections
import datetime
import time
import numpy as np
import math
from data import load_dataset
import user
import json
import random
from abbreviation import UniformAbbreviation

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

def build_oneshot_dataset(dataset, language, eval_examples, n_abbreviations,
                          abbreviation_strategy):
    print('Loading dataset.')
    dataset = load_dataset(dataset)[language]['train']
    print('Loaded. Counting substrings...')

    substring_counts = collections.defaultdict(int)
    p = Progress(len(dataset))

    # Get all candidates.
    for l in dataset:
        for sz in range(3, 40):
            for i in range(len(l) - sz + 1):
                ss = l[i:i+sz]
                if ss.strip() == ss and any(map(lambda c: c.isalnum(), ss)):
                    substring_counts[ss] += 1
        p.tick()

        if (p.current_iteration + 1) % 1000 == 0:
            print(p.format())

    candidates = [(k, v) for k, v in substring_counts.items() if v >= eval_examples]
    print('Counted. Initial candidates:', len(candidates))

    # Filter only maximal candidates.
    is_maximal = collections.defaultdict(lambda: True)

    for c, cnt in candidates:
        for c2, cnt2 in candidates:
            if c != c2:
                if c.find(c2) != -1:
                    is_maximal[c2] = False
                if c2.find(c) != -1:
                    is_maximal[c] = False

    candidates = [(k, v) for k, v in candidates if is_maximal[k]]

    print('Maximal candidates:', len(candidates))

    candidates.sort(key=lambda kv: len(kv[0]) * math.log(kv[1]), reverse=True)

    if len(candidates) < n_abbreviations:
        raise Exception("Not enough maximal candidates: asked for {}, have {}"
                        .format(n_abbreviations, len(candidates)))

    candidates = candidates[:n_abbreviations]

    oneshot_dataset = []

    good, bad = 0, 0

    print('Computing abbreviations and finding examples...')
    p = Progress(len(candidates))

    for k, v in candidates:
        best_abbreviation = None
        positive_examples = [s for s in dataset if s.find(k) != -1]
        best_negative_examples = []

        for i in range(10):
            abbreviation = abbreviation_strategy.abbreviate(k)
            negative_examples = [s for s in dataset if s.find(abbreviation) != -1]

            if best_abbreviation is None or len(negative_examples) > len(best_negative_examples):
                best_abbreviation = abbreviation
                best_negative_examples = negative_examples

        oneshot_dataset.append({
            'string': k,
            'abbreviation': best_abbreviation,
            'positive_examples': random.sample(positive_examples,
                                               min(len(positive_examples), eval_examples)),
            'negative_examples': random.sample(best_negative_examples,
                                               min(len(best_negative_examples), eval_examples)),
        })

        if len(best_negative_examples) >= eval_examples:
            good += 1
        else:
            bad += 1

        p.tick()
        if (p.current_iteration + 1) % 100 == 0:
            print(p.format())

    print('Good:', good, 'bad:', bad)

    with open('oneshot_dataset.json', 'w') as f:
        json.dump(oneshot_dataset, f)

def batched(l, batch_size):
    for b in range((len(l) + batch_size - 1) // batch_size):
        yield l[b*batch_size:(b+1)*batch_size]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConadComplete utilities')

    parser.add_argument('--seed', default='conadcomplete', help='Random seed')
    parser.add_argument('--language', default='Python', help='Programming Language to use (Python|Haskell|Java).')
    parser.add_argument('--dataset', default='medium', help='Dataset to use')
    parser.add_argument('--new-convention-every', default=100, type=int, help='Iterations between new conventions')
    parser.add_argument('--precompute-interactions', action='store_const', const=True, default=False)
    parser.add_argument('--build-oneshot-dataset', action='store_const', const=True, default=False)
    parser.add_argument('--oneshot-eval-examples', type=int, default=200, help='How many positive/negative examples to fetch for each convention in the one-shot dataset')
    parser.add_argument('--oneshot-abbreviations', type=int, default=1000, help='How many abbreviations scenarios to put in the one-shot dataset.')
    parser.add_argument('--one-convention', help='Limit to applying at most one convention per input.',
                        action='store_const', const=True, default=False)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.precompute_interactions:
        precompute_interactions(args.dataset, args.language, args.new_convention_every, args.one_convention)
    elif args.build_oneshot_dataset:
        build_oneshot_dataset(args.dataset, args.language,
                              args.oneshot_eval_examples, args.oneshot_abbreviations,
                              UniformAbbreviation(0.2))
