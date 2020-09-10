# Build dataset for fine-tuning to a particular repo

import random
from abbreviator import CLMLanguageAbbreviator
from build_contextual_lines_dataset import build_dataset
from do import build_abbreviation_targets, cap_collisions
import logging

DEFAULT_SPLITS = {
    'train': 0.8,
    'test': 0.2,
}

def build_repo_dataset(repo_files, splits=DEFAULT_SPLITS):
    random.shuffle(repo_files)

    last_end = 0
    dataset = {}

    for k, v in splits.items():
        begin = last_end
        end = min(int(begin + v * len(repo_files)), len(repo_files))
        dataset[k] = build_dataset(repo_files[begin:end], examples_from_file=10**10)
        last_end = end

    return dataset

def _compute_compression(abbreviator, examples):
    encoded, _ = abbreviator.encode(examples)
    return 1 - (sum(len(r['l']) for r in encoded) /
                sum(len(r['l']) for r in examples))

def specialize_abbreviator(abbreviator, repo_dataset):
    compression_before = _compute_compression(abbreviator, repo_dataset['test'])
    logging.debug('Building specialized abbreviations...')
    specialized_targets = build_abbreviation_targets(
            len(abbreviator.abbreviation_targets), repo_dataset['train'])
    logging.debug('Finished building specialized abbreviations.')

    if abbreviator.params.get('max_collisions'):
        specialized_targets = cap_collisions(
                specialized_targets, abbreviator.params['max_collisions'])

    specialized_abbreviator = CLMLanguageAbbreviator(
            abbreviator.clm, specialized_targets,
            {
                **abbreviator.params,
                'log_every': 1,
            })
    compression_after = _compute_compression(specialized_abbreviator, repo_dataset['test'])
    accuracy_before_ft = specialized_abbreviator.compute_accuracy(repo_dataset['test'])
    specialized_abbreviator.fit(None, repo_dataset, fine_tuning=True)
    accuracy_after_ft = specialized_abbreviator.compute_accuracy(repo_dataset['test'])

    return {
            'compression_before': compression_before, 
            'compression_after': compression_after, 
            'accuracy_before': abbreviator.compute_accuracy(repo_dataset['test']), 
            'accuracy_before_ft': accuracy_before_ft,
            'accuracy_after_ft': accuracy_after_ft,
           }
