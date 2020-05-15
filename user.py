# User model for continual adaptation auto-complete.

import collections
import math
import random
from data import load_dataset
from util import replace_identifier

class User:
    def __init__(self, convention_compression=0.5, conventions_queue=None):
        self.conventions = {}
        self.convention_id = {}
        self.conventions_queue = list(reversed(conventions_queue or []))
        self.substring_count = collections.defaultdict(int)
        self.convention_compression = convention_compression

    def encode(self, s, trace_conventions=False,
               one_convention=False, only_identifiers=True):
        if trace_conventions:
            conventions_used = []

        replace = lambda s, k, v: (s.replace(k, v)
                                   if not only_identifiers
                                   else replace_identifier(s, k, v))

        for k, v in self.conventions.items():
            s_before = s
            s = s.replace(k, v)

            if s != s_before:
                if trace_conventions:
                    conventions_used.append(self.convention_id[k])
                if one_convention:
                    break

        if trace_conventions:
            return s, conventions_used

        return s

    def remember_substrings(self, s, min_length=4, max_length=20):
        for l in range(min_length, min(len(s), max_length)):
            for i in range(len(s) - l + 1):
                ss = s[i:i+l].strip()
                if len(ss) == l:
                    self.substring_count[ss] += 1

    def form_new_convention(self):
        all_conventions = [(s, cnt)
                           for s, cnt in self.substring_count.items()
                           if cnt > 1 and not any(conv.find(s) != -1 for conv in self.conventions.keys())]

        if len(all_conventions) == 0:
            print('No new conventions to be formed.')
            return

        best_s, best_cnt = max(all_conventions, key=lambda s: len(s[0]) * s[1])

        convention = ''
        while len(convention) in (0, len(best_s)):
            convention = ''.join(c for c in best_s if random.random() < self.convention_compression)

        self.conventions[best_s] = convention
        self.convention_id[best_s] = len(self.conventions) - 1

        print('New convention:', repr(best_s), '==>', repr(convention), '(', best_cnt, 'occurrences)')

        self.prune()

        return (best_s, convention)

    def add_next_convention(self):
        s, c = self.conventions_queue.pop()
        self.conventions[s] = c

    def add_new_convention(self, string, abbreviation):
        self.conventions[string] = abbreviation
        self.convention_id[string] = len(self.conventions) - 1

    def prune(self):
        if len(self.substring_count) > 10**5:
            all_substrings = [(k, v) for k, v in self.substring_count.items()]
            all_substrings.sort(key=lambda kv: -kv[1])
            self.substring_count = collections.defaultdict(int, all_substrings[:10**4])

if __name__ == '__main__':
    u = User()

    data = load_dataset('small')['Python']['train']
    random.shuffle(data)

    print(len(data), 'lines of code.')

    for i, l in enumerate(data):
        enc = u.encode(l)

        u.remember_substrings(enc)

        if (i + 1) % 100 == 0:
            u.form_new_convention()
