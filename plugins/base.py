import torch
from abbreviator import *
from language_model import *
from models import *

def split_strip(l):
    s = l.strip()
    idx = l.index(s)
    return l[:idx], l[idx:idx + len(s)], l[idx + len(s):]

class TextEditorPluginBackend:
    def __init__(self):
        return
        print('Loading...')
        self.abbreviator = CLMLanguageAbbreviator.load('models/934c7.model', device=torch.device('cpu'))
        self.shown_hints = set()
        print('Done!')

    def get_hint(self, line):
        return
        line = line.strip()
        shortened = self.abbreviator.encode([{'l': line, 'i': [], 'c': []}])[0][0]['l']

        if shortened != line:
            return 'Try typing \'' + shortened + '\' and pressing Ctrl-J! I\'ll expand it to \'' + line + '\''
        return None

    def expand(self, buffer, current_line_index):
        return
        before, line, after = split_strip(line)
        expansions = self.abbreviator.decode(
            [{ 'l': line, 'i': imports, 'p': identifiers }])
        return expansions[0], before, after

    def record_hint(self, hint):
        return
        self.shown_hints.add(hint)

    def hint_was_shown(self, hint):
        return
        return hint in self.shown_hints
