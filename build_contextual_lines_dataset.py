import json
import argparse
import random
from util import Progress

parser = argparse.ArgumentParser(
        'Build contextualized lines of code dataset from dataset of files.')

parser.add_argument('-f', '--files-dataset', help='Path to the files dataset.',
                    default='dataset-files.json')
parser.add_argument('-c', '--context-size', help='Maximum number of identifiers in the context of a line.', type=int, default=10)
parser.add_argument('-p', '--previous', help='Number of previous lines to include for each example.', type=int, default=10)
parser.add_argument('-m', '--min-length', help='Minimum line length.', type=int, default=4)
parser.add_argument('-M', '--max-length', help='Maximum line length.', type=int, default=80)
parser.add_argument('-N', '--max-name-length', help='Maximum name of a name in context or imports.', type=int, default=20)
parser.add_argument('-e', '--examples-from-file', help='Maximum number of examples to extract from a single file.', type=int, default=20)
parser.add_argument('-n', '--n-files', help='Number of files to use.', type=int, default=1000)
parser.add_argument('-l', '--language', help='Language to build the dataset for.', default='Python')
parser.add_argument('-o', '--output', help='Output file.', default='lines.json')

opt = parser.parse_args()

def extract_identifiers(s):
    ids = []
    is_in_id = False

    for c in s:
        if c.isidentifier() or (is_in_id and c.isdigit()):
            if not is_in_id:
                ids.append('')
                is_in_id = True
            ids[-1] += c
        else:
            is_in_id = False
    return ids

def extract_imports(l):
    if l.startswith('import'):
        tokens = l.split()
        if len(tokens) >= 2:
            return [tokens[1].split('.')[0]]

    if l.startswith('from') and l.find('import') != -1:
        tokens = l.split()

        if len(tokens) >= 2:
            return [tokens[1].split('.')[0]]

    return []

def example_is_valid(context, imports, line):
    if not (opt.min_length <= len(line) <= opt.max_length):
        return False

    try:
        line.encode('ascii')

        for s in context + imports:
            if len(s) > opt.max_name_length:
                return False
            s.encode('ascii')
    except UnicodeEncodeError:
        return False

    return True

def build_examples_from_file(f):
    ids, imports, lines, examples = [], set(), [], []

    for line_raw in f.split('\n'):
        line = line_raw.strip()

        # Comments and doc strings
        if (line.startswith('#') or
                line.startswith('"') or
                line.startswith("'")):
            ids.extend(extract_identifiers(line))
            continue

        c, i = ids[-opt.context_size:], list(imports)
        if example_is_valid(c, i, line):
            lines.append(line_raw)
            examples.append({ 'c': c, 'i': i, 'l': line, 'p': lines[-opt.previous:]})

        ids.extend(extract_identifiers(line))
        imports.update(extract_imports(line))

    return examples

def build_dataset(files, progress):
    examples = []

    for f in files:
        f_examples = build_examples_from_file(f)
        examples.extend(
                random.sample(f_examples,
                              min(len(f_examples), opt.examples_from_file)))
        progress.tick()

        if progress.current_iteration % 100 == 0:
            print(progress.format())

    return examples

if __name__ == '__main__':
    with open(opt.files_dataset) as f:
        print('Loading', opt.files_dataset, '...')
        files_dataset = json.load(f)[opt.language]
        files_count = sum(map(len, files_dataset.values()))
        n = min(files_count, opt.n_files)
        print(files_count, 'files in the dataset. Using', n)
        dataset = {}

        p = Progress(n)

        for split, files in files_dataset.items():
            dataset[split] = build_dataset(
                    random.sample(files, int(n / files_count * len(files))), p)

        print('Split sizes:')
        for split, examples in dataset.items():
            print('{}: {} examples'.format(split, len(examples)))

        with open(opt.output, 'w') as f:
            json.dump(dataset, f)
