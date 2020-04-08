import subprocess
import json
import sys
import os.path
import random
import collections

LANGUAGES = ('Java', 'Python', 'Haskell')
EXTENSIONS = ('.java', '.py', '.hs')
SPLITS = {'train': 0.8, 'dev': 0.1, 'test': 0.1}

random.seed(224)

def is_comment(line):
    return line.startswith('//') or line.startswith('/*') or line.startswith('#') or line.startswith('-- ') or line.startswith('* ')

def clean_line(line):
    line = line.strip()
    return (line or None) if not is_comment(line) else None

def sample_lines_from_repo(path):
    MAX_LINES_PER_REPO = 100
    lines = set()

    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            if any(fn.endswith(ext) for ext in EXTENSIONS):
                try:
                    with open(os.path.join(dirpath, fn)) as f:
                        f_lines = f.readlines()
                        lines.update(clean_line(l) for l in f_lines)
                except:
                    pass

    lines.discard(None)
    lines = list(lines)
    random.shuffle(lines)
    return lines[:MAX_LINES_PER_REPO]

with open(sys.argv[1]) as f:
    repos = json.load(f)

downloads, skipped = 0, 0

lines_by_language = collections.defaultdict(list)
DESIRED_LINES = 10**6

random.shuffle(repos)

try:
    for row in repos:
        language = row["language"]
        if language not in LANGUAGES or len(lines_by_language[language]) >= DESIRED_LINES:
            continue

        url = row["url"]
        dest = url[1:].replace('/', '__')

        if not os.path.exists(dest):
            subprocess.call("timeout 20 git clone --depth 1 https://github.com{} {}".format(url, dest), shell=True)
            downloaded = True
            downloads += 1
        else:
            skipped += 1
            downloaded = False

        lines = sample_lines_from_repo(dest)
        lines_by_language[row["language"]].extend(lines)

        if downloaded:
            subprocess.call("rm -rf {}".format(dest), shell=True)

        print('# lines by language:', {l:len(lines_by_language[l]) for l in LANGUAGES})
except KeyboardInterrupt:
    pass

dataset = collections.defaultdict(lambda: collections.defaultdict(list))

for l in LANGUAGES:
    lines = lines_by_language[l]
    random.shuffle(lines)
    assigned_lines = 0
    for s, f in SPLITS.items():
        split_limit = assigned_lines + int(len(lines) * f)
        dataset[l][s] = lines[assigned_lines:split_limit]
        assigned_lines = split_limit

with open('dataset.json', 'w') as f:
    json.dump(dataset, f)

print("{} repositories downloaded, {} skipped.".format(downloads, skipped))
