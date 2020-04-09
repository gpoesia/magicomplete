# Implementation of various abbreviation rules.

import random

class AbbreviationStrategy:
    def name(self):
        raise NotImplemented()

    def abbreviate(self, s):
        raise NotImplemented()

class UniformAbbreviation(AbbreviationStrategy):
    def __init__(self, p):
        self.p = p

    def name(self):
        return "UniformAbbreviation({})".format(self.p)

    def abbreviate(self, s):
        empty = True
        while empty:
            abb = ''.join(c for c in s if random.random() < self.p)
            empty = len(abb) == 0
        return abb
