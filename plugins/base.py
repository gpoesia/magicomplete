from abbreviator import *
from language_model import *
from models import *

def split_strip(l):
    s = l.strip()
    idx = l.index(s)
    return l[:idx], l[idx:idx + len(s)], l[idx + len(s):]

class TextEditorPluginBackend:
    def __init__(self):
        print('Loading...')
        self.abbreviator = load_from_run(DiscriminativeLanguageAbbreviator,
                                         '4bfc0',
                                         model_key='abbreviator')
        self.shown_hints = set()
        print('Done!')

    def get_hint(self, line):
        line = line.strip()
        shortened = self.abbreviator.encode([{'l': line, 'i': [], 'c': []}])[0][0]['l']

        if shortened != line:
            return 'Try typing \'' + shortened + '\' and pressing Ctrl-J! I\'ll expand it to \'' + line + '\''
        return None

    def expand(self, line, imports, identifiers):
        before, line, after = split_strip(line)
        expansions = self.abbreviator.decode(
            [{ 'l': line, 'i': imports, 'c': identifiers }])
        return expansions[0], before, after

    def record_hint(self, hint):
        self.shown_hints.add(hint)

    def hint_was_shown(self, hint):
        return hint in self.shown_hints
