from abbreviator import *
from language_model import *
from models import *

class TextEditorPluginBackend:
    def __init__(self):
        print('Loading...')
        self.abbreviator = load_from_run(LMRLanguageAbbreviator,
                                         '9d606',
                                         model_key='abbreviator')
        print('Done!')

    def get_hint(self, line):
        shortened = self.abbreviator.encode([{'l': line, 'i': [], 'c': []}])[0][0]['l']

        if shortened != line:
            return 'Try typing \'' + shortened + '\' and pressing Ctrl-J! I\'ll expand it to \'' + line + '\''
        else:
            return '{}'.format(self.abbreviator.abbreviation_table['def'])

    def expand(self, line, imports, identifiers):
        expansions = self.abbreviator.decode(
            [{ 'l': line, 'i': imports, 'c': identifiers }])
        return expansions[0]
