# Context-related enums.

from enum import Flag, auto

class ContextAlgorithm(Flag):
    NONE = 0
    CONCAT_CELL = auto()
    FACTOR_CELL = auto()
    CNN = auto()

    @staticmethod
    def parse(s):
        return ({
            'NONE': ContextAlgorithm.NONE,
            'FACTOR_CELL': ContextAlgorithm.FACTOR_CELL,
            'CONCAT_CELL': ContextAlgorithm.CONCAT_CELL,
            'CNN': ContextAlgorithm.CNN,
            })[s]

class Context(Flag):
    NONE = 0
    IMPORTS = auto()
    IDENTIFIERS = auto()

    def count(self):
        return sum(1 for v in [self.IMPORTS, self.IDENTIFIERS] if self & v)

    @staticmethod
    def parse(s):
        return ({
            'NONE': Context.NONE,
            'IMPORTS': Context.IMPORTS,
            'IDENTIFIERS': Context.IDENTIFIERS,
            'IMPORTS+IDENTIFIERS': Context.IMPORTS | Context.IDENTIFIERS,
            })[s]
