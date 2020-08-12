import json
import argparse
import random
from util import Progress
import ast

parser = argparse.ArgumentParser(
        'Build user study dataset from contextualized lines dataset.')

parser.add_argument('-l', '--lines-dataset', help='Path to the contextualized lines dataset.',
                    default=None)
parser.add_argument('-f', '--files-dataset', help='Path to the files dataset.',
                    default=None)
parser.add_argument('--language', help='Language (needed for files dataset).',
                    default='Python')
parser.add_argument('-n', '--n-lines', help='Number of examples to extract.', type=int, default=20)
parser.add_argument('-m', '--min-lines', help='Minimum lines in each example.', type=int, default=5)
parser.add_argument('-s', '--seed', help='Random seed.', default='user-study')
parser.add_argument('-o', '--output', help='Output file.', default='user-study.json')
parser.add_argument('--dont-parse', help='Do not try to parse snippet (for languages other than Python).', dest='parse', action='store_const', const=False, default=True)

opt = parser.parse_args()

random.seed(opt.seed)

def format(p):
    min_indent = min(l.index(l.strip()) for l in p)
    return '\n'.join(l[min_indent:] for l in p)

def extract_from_lines_dataset():
    with open(opt.lines_dataset) as f:
        print('Loading', opt.lines_dataset, '...')
        d = json.load(f)
        ex = []
        total_lines, total_characters = 0, 0
        random.shuffle(d['test'])
    
        for r in d['test']:
            if len(r['p']) < opt.min_lines:
                continue
    
            t = format(r['p'])
    
            try:
                if False and opt.parse:
                    ast.parse(t)
                ex.append(t)
                total_lines += len(r['p'])
                total_characters += len(t)
                if len(ex) == opt.n_lines:
                    break
            except:
                pass
    
        with open(opt.output, 'w') as f:
            json.dump(ex, f)
    
            print('Wrote', len(ex), 'examples.')
            for i, e in enumerate(ex):
                print('\n\nExample #{}:'.format(i+1))
                print(e)
    
            print('\n\n' + '#' * 10)
            print('{} lines, {} characters in total.'.format(total_lines, total_characters))

def extract_from_files_dataset():
    with open(opt.files_dataset) as f:
        print('Loading', opt.files_dataset, '...')
        d = json.load(f)[opt.language]
        ex = []
        total_lines, total_characters = 0, 0
        random.shuffle(d['test'])

        for s in d['test']:
            lines = s.split('\n')
            idx = random.randint(0, len(lines) - 1)
            example = lines[idx:idx+opt.min_lines]

            if len(example) < opt.min_lines:
                continue

            t = format(example)

            try:
                if opt.parse:
                    ast.parse(t)
                ex.append(t)
                total_lines += len(example)
                total_characters += len(t)
                if len(ex) == opt.n_lines:
                    break
            except:
                pass
    
        with open(opt.output, 'w') as f:
            json.dump(ex, f)
    
            print('Wrote', len(ex), 'examples.')
            for i, e in enumerate(ex):
                print('\n\nExample #{}:'.format(i+1))
                print(e)
    
            print('\n\n' + '#' * 10)
            print('{} lines, {} characters in total.'.format(total_lines, total_characters))

if opt.lines_dataset is not None:
    extract_from_lines_dataset()
else:
    extract_from_files_dataset()
