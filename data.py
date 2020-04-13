import json
import os
import sys
import ast

SMALL, MEDIUM, LARGE = 'small.json', 'medium.json', 'large.json'

def filter_ascii(strings):
    'Returns only the strings that can be encoded in ASCII.'
    l = []
    for s in strings:
        try:
            s.encode('ascii')
            s = s.strip()
            if 10 <= len(s) <= 80:
                l.append(s)
        except UnicodeEncodeError:
            pass

    return l

def load_dataset(size=SMALL):
    root = os.path.dirname(__file__)
    with open(os.path.join(root, 'dataset', '{}.json'.format(size))) as f:
        data = json.load(f)

        for language in data.keys():
            for split in data[language].keys():
                data[language][split] = filter_ascii(data[language][split])

        return data

def generate_sanity_check_dataset():
    'Returns a dataset where all examples are the same string, which consists of 10 times the same letter.'

    SIZE = 200
    l = []

    for i in range(SIZE):
        l.append(random.choice('abcdefghijklmnopqrstuvwxyz') * random.choice([5, 10]))

    return {'train': l, 'dev': l, 'test': l}

def get_kept_character_indices(original, short):
    kept = []
    i_original, i_short = 0, 0

    while i_short < len(short) and i_original < len(original):
        if short[i_short] == original[i_original]:
            kept.append(i_original)
            i_short += 1

        i_original += 1

    return kept

def infer_token_limits(s):
    limits = [0]
    for i in range(1, len(s)):
        if s[i].isalnum() != s[i-1].isalnum():
            limits.append(i)
    limits.append(len(s))
    return limits

def augment(short, original, only_shortened=True):
    kept = set(get_kept_character_indices(original, short))
    limits = infer_token_limits(original)

    examples = [(short, original)]

    for i, begin in enumerate(limits):
        for end in limits[i+1:]:
            ss = original[begin:end]
            if len(ss.strip()) == len(ss):
                try:
                    ast.parse(ss)
                    ss_short = ''.join(original[i] for i in range(len(original))
                                       if begin <= i < end and i in kept)
                    if not only_shortened or len(ss_short) < len(ss):
                        examples.append((ss_short, ss))
                except:
                    pass

    return examples
