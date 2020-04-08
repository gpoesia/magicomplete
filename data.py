import json
import os
import sys

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
