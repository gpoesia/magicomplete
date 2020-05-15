# Various utilities.

import datetime
import time
import numpy as np
import random

class Progress:
    def __init__(self, total_iterations=None, timeout=None, print_every=None):
        self.total_iterations = total_iterations
        self.timeout = timeout
        self.begin = time.time()
        self.current_iteration = 0
        self.print_every = print_every

    def tick(self, inc=1):
        self.current_iteration += inc
        if self.print_every is not None and self.current_iteration % self.print_every == 0:
            print(self.format())
        return self.current_iteration

    def timed_out(self):
        return (self.timeout is not None and
                time.time() - self.begin > timeout)

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

def batched(l, batch_size):
    for b in range((len(l) + batch_size - 1) // batch_size):
        yield l[b*batch_size:(b+1)*batch_size]

def broadcast_dot(m, v):
    'torch.dot() broadcasting version'
    return m.mm(v.view(-1, 1)).squeeze(1)

def trim_str_to_id(s):
    first_id, last_id = None, None

    for i, c in enumerate(s):
        if c.isidentifier():
            if first_id is None:
                first_id = i
            last_id = i

    if first_id is None:
        return None
    return s[first_id:(last_id + 1)]

def random_hex_string(k=5):
    CHARS = '0123456789abcdef'
    return ''.join(random.choices(CHARS, k=k))

def replace_identifier(s, id, val, last_was_id=False):
    '''Replaces occurrences of `id` in string `s` by `val` only where `id` is surrounded by
    non-identifier characters.

    For example, for s = 'l(collision)', id = 'l', val = 'len', the result will be 'len(collision)',
    since the l's inside collision are not replaced.
    '''

    tokens = split_at_identifier_boundaries(s)
    for i in range(len(tokens)):
        if tokens[i] == id:
            tokens[i] = val

    return ''.join(tokens)

def split_at_identifier_boundaries(s):
    '''Returns a list of strings obtained by splitting s at identifier boundaries.

    Example: split_at_identifier_boundaries('self.x0 += 2') == ('self', '.', 'x0', ' += 2')
    '''

    tokens = []
    is_in_id = False

    for c in s:
        if c.isidentifier() or (is_in_id and c.isdigit()):
            if not is_in_id:
                tokens.append('')
                is_in_id = True
        else:
            if is_in_id or len(tokens) == 0:
                tokens.append('')
                is_in_id = False
        tokens[-1] += c

    return tuple(tokens)
