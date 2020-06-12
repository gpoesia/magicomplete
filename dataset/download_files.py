import subprocess
import json
import sys
import os.path
import random
import collections

LANGUAGES = ('Python',)
EXTENSIONS = ('.py',)
SPLITS = {'train': 0.8, 'dev': 0.1, 'test': 0.1}

random.seed('autocomplete')

def is_comment(line):
    return line.startswith('#') or line.startswith("'") or line.startswith('"')

def clean_line(line):
    return (line or None) if not is_comment(line.strip()) else None

def sample_files_from_repo(path):
    MAX_FILES_PER_REPO = 100
    files = set()

    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            if any(fn.endswith(ext) for ext in EXTENSIONS):
                try:
                    with open(os.path.join(dirpath, fn)) as f:
                        f_lines = f.readlines()
                        files.add('\n'.join(filter(None,
                                                   (clean_line(l) for l in f_lines))))
                except:
                    pass

    files = list(files)
    random.shuffle(files)
    return files[:MAX_FILES_PER_REPO]

with open(sys.argv[1]) as f:
    repos = json.load(f)

downloads, skipped = 0, 0

files_by_language = collections.defaultdict(set)
DESIRED_FILES = 10**5

random.shuffle(repos)

try:
    for row in repos:
        language = row["language"]
        if language not in LANGUAGES or len(files_by_language[language]) >= DESIRED_FILES:
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

        files = sample_files_from_repo(dest)
        files_by_language[row["language"]].update(files)

        if downloaded:
            subprocess.call("rm -rf {}".format(dest), shell=True)

        print('# files by language:', {l:len(files_by_language[l]) for l in LANGUAGES})
except KeyboardInterrupt:
    pass

dataset = collections.defaultdict(lambda: collections.defaultdict(list))

for l in LANGUAGES:
    files = list(files_by_language[l])
    random.shuffle(files)
    assigned_files = 0
    for s, f in SPLITS.items():
        split_limit = assigned_files + int(len(files) * f)
        dataset[l][s] = files[assigned_files:split_limit]
        assigned_files = split_limit

with open('100k-files.json', 'w') as f:
    json.dump(dataset, f)

print("{} repositories downloaded, {} skipped.".format(downloads, skipped))
