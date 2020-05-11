# CLI tool that automates frequent project-related tasks.

import argparse
from util import *
import torch
import baseline
from set_embedding import SetEmbedding
import collections
import datetime
import time
import numpy as np
import math
from data import load_dataset
from decoder import AutoCompleteDecoderModel, Context, ContextAlgorithm
from train import train
import user
import json
import random
from abbreviation import UniformAbbreviation
from abbreviator import *
from slack import send_message
from run_tracker import RunTracker
from language_model import RNNLanguageModel

def precompute_interactions(dataset, language, new_convention_every, one_convention=False):
    dataset = load_dataset(dataset)[language]['train']
    random.shuffle(dataset)

    u = user.User()
    events = []
    p = Progress(len(dataset))

    for i, l in enumerate(dataset):
        enc, conventions = u.encode(l, trace_conventions=True, one_convention=one_convention)
        events.append({'type': 'user_input', 'long': l, 'short': enc, 'conventions': conventions})

        u.remember_substrings(l)
        p.tick()

        if (i + 1) % new_convention_every == 0:
            s, c = u.form_new_convention()
            events.append({'type': 'convention', 'long': s, 'short': c})
            print(p.format())

    with open('interactions.json', 'w') as f:
        json.dump(events, f)

def build_oneshot_dataset(dataset, language, eval_examples, n_abbreviations,
                          abbreviation_strategy):
    print('Loading dataset.')
    dataset = load_dataset(dataset)[language]['train']
    print('Loaded. Counting substrings...')

    substring_counts = collections.defaultdict(int)
    p = Progress(len(dataset))

    # Get all candidates.
    for l in dataset:
        for sz in range(3, 40):
            for i in range(len(l) - sz + 1):
                ss = l[i:i+sz]
                if ss.strip() == ss and any(map(lambda c: c.isalnum(), ss)):
                    substring_counts[ss] += 1
        p.tick()

        if (p.current_iteration + 1) % 1000 == 0:
            print(p.format())

    candidates = [(k, v) for k, v in substring_counts.items() if v >= eval_examples]
    print('Counted. Initial candidates:', len(candidates))

    # Filter only maximal candidates.
    is_maximal = collections.defaultdict(lambda: True)

    for c, cnt in candidates:
        for c2, cnt2 in candidates:
            if c != c2:
                if c.find(c2) != -1:
                    is_maximal[c2] = False
                if c2.find(c) != -1:
                    is_maximal[c] = False

    candidates = [(k, v) for k, v in candidates if is_maximal[k]]

    print('Maximal candidates:', len(candidates))

    candidates.sort(key=lambda kv: len(kv[0]) * math.log(kv[1]), reverse=True)

    if len(candidates) < n_abbreviations:
        raise Exception("Not enough maximal candidates: asked for {}, have {}"
                        .format(n_abbreviations, len(candidates)))

    oneshot_dataset = []

    print('Computing abbreviations and finding examples...')
    p = Progress(n_abbreviations)

    for k, v in candidates:
        best_abbreviation = None
        positive_examples = [s for s in dataset if s.find(k) != -1]
        best_negative_examples = []

        for i in range(10):
            abbreviation = abbreviation_strategy.abbreviate(k)
            negative_examples = list({s for s in dataset if s.find(abbreviation) != -1})

            if best_abbreviation is None or len(negative_examples) > len(best_negative_examples):
                best_abbreviation = abbreviation
                best_negative_examples = negative_examples

        positive_examples = list(set(positive_examples))
        negative_examples = list(set(best_negative_examples))

        if min(len(positive_examples), len(negative_examples)) >= eval_examples:
            random.shuffle(positive_examples)
            random.shuffle(negative_examples)

            oneshot_dataset.append({
                'string': k,
                'abbreviation': best_abbreviation,
                'positive_examples': positive_examples[:eval_examples],
                'negative_examples': negative_examples[:eval_examples],
                'positive_examples_train': positive_examples[eval_examples:2*eval_examples],
                'negative_examples_train': negative_examples[eval_examples:2*eval_examples],
            })

            p.tick()
            if (p.current_iteration + 1) % 100 == 0:
                print(p.format())

        if len(oneshot_dataset) == n_abbreviations:
            break

    with open('oneshot_dataset.json', 'w') as f:
        json.dump(oneshot_dataset, f)

def train_set_embedding(
        contextual_dataset_path,
        device,
        output_path,
        epochs):
    print('Loading dataset...')

    with open(contextual_dataset_path) as f:
        dataset = json.load(f)['train']

    training_set = [r['c'] for r in dataset] + [r['i'] for r in dataset]
    random.shuffle(training_set)

    print('Pre-training set embedding. {} examples'.format(len(training_set)))
    print('Using device', device)
    semb = SetEmbedding(device, hidden_size=128)
    loss, acc = semb.train(training_set, {'epochs': epochs, 'lr': 1e-3})
    semb.dump(output_path)
    print('Wrote', output_path)
    print('Last loss:', loss[-1])
    print('Last accuracy:', acc[-1])

def train_language_model(params_path, device, contexts_to_run):
    print('Using device', device)
    with open(params_path) as f:
        params = json.load(f)

    print('Loading dataset...')

    with open(params['dataset']) as f:
        dataset = json.load(f)

    contexts_to_run = contexts_to_run.split(',')

    for i, ctx_value in enumerate(contexts_to_run):
        params['model']['context'] = ctx_value
        lm = RNNLanguageModel(params['model'], device)

        print('{}/{} Training with context = {}'.format(
              i+1, len(contexts_to_run), ctx_value))

        tracker = RunTracker(lm, params)
        tracker.start()

        lm.fit(dataset, tracker, params)
        tracker.close()

def train_decoders(params_path,
                   device,
                   contexts_to_run):
    print('Using device', device)
    with open(params_path) as f:
        params = json.load(f)

    print('Loading dataset...')

    with open(params['dataset']) as f:
        dataset = json.load(f)

    if params.get('initial_set_embedding') is not None:
        print('Using ConcatCell with pre-trained set embedding.')
        set_embedding = SetEmbedding.load(params.get('initial_set_embedding'), device=device)
    else:
        print('Training with Context CNN end-to-end')
        set_embedding = None

    contexts_to_run = contexts_to_run.split(',')


    for i, ctx_value in enumerate(contexts_to_run):
        params['decoder']['context'] = ctx_value

        print('{}/{} Training with context = {}'.format(
              i+1, len(contexts_to_run), ctx_value))
        encoder = baseline.create_encoder(params['encoder']['type'], params['encoder']['params'])
        decoder = AutoCompleteDecoderModel(params['decoder'], device)

        tracker = RunTracker(decoder, params)
        tracker.start()

        train(encoder,
              decoder,
              set_embedding,
              dataset,
              params,
              device,
              tracker)

        tracker.close()

def build_abbreviation_targets(n_abbreviations, dataset):
    candidates = find_common_identifiers(dataset)
    targets = []

    for c, _, _ in candidates:
        seen_before = False
        for current in targets:
            if current.find(c) != -1:
                seen_before = True
                break
        if seen_before:
            continue
        targets.append(c)
        if len(targets) == n_abbreviations:
            break
    return targets

def run_abbreviator_experiment(
        dataset_path,
        set_embedding_path,
        n_targets,
        device,
        decoders_prefix,
        contexts):
    print('Loading dataset {}...'.format(dataset_path))
    with open(dataset_path) as f:
        ds = json.load(f)

    print(len(ds['dev']), 'examples in the validation set.')

    targets = build_abbreviation_targets(n_targets, ds['train'])

    set_embedding = (SetEmbedding.load(set_embedding_path, device=device)
                     if set_embedding_path is not None
                     else None)

    results = {}
    evaluator = AbbreviatorEvaluator(targets, ds['dev'])
    summary = ''

    abbreviators = [
        LanguageAbbreviator(
            'ctx=nil',
            AutoCompleteDecoderModel.load(
                decoders_prefix + '_ctx0.model', Context.NONE, ContextAlgorithm.CNN, device=device),
            set_embedding,
            ds['train'],
            { 'learning_rate': 0.1, 'rehearsal_batch_size': 64,
                'minimum_validation_accuracy': 0.8 },
        ),
        LanguageAbbreviator(
            'ctx=imports',
            AutoCompleteDecoderModel.load(
                decoders_prefix + '_ctx1.model', Context.IMPORTS, ContextAlgorithm.CNN, device=device),
            set_embedding,
            ds['train'],
            { 'learning_rate': 0.1, 'rehearsal_batch_size': 64,
                'minimum_validation_accuracy': 0.8 },
        ),
        LanguageAbbreviator(
            'ctx=identifiers',
            AutoCompleteDecoderModel.load(
                decoders_prefix + '_ctx2.model', Context.IDENTIFIERS, ContextAlgorithm.CNN, device=device),
            set_embedding,
            ds['train'],
            { 'learning_rate': 0.1, 'rehearsal_batch_size': 64,
                'minimum_validation_accuracy': 0.8 },
        ),
        LanguageAbbreviator(
            'ctx=imports+identifiers',
            AutoCompleteDecoderModel.load(
                decoders_prefix + '_ctx3.model',
                Context.IMPORTS | Context.IDENTIFIERS,
                ContextAlgorithm.CNN,
                device=device),
            set_embedding,
            ds['train'],
            { 'learning_rate': 0.1, 'rehearsal_batch_size': 64,
                'minimum_validation_accuracy': 0.8 },
        ),
    ]

    contexts = list(map(int, contexts.split(','))) or list(range(4))
    abbreviators = [a for a in abbreviators if a.decoder.context.value in contexts]

    p = Progress(len(abbreviators) * len(targets), print_every=5)

    for ab in abbreviators:
        print('Evaluating', ab.name())
        results[ab.name()] = ab_results = evaluator.evaluate(ab, p)
        ab_summary = ('{} - accuracy: {:.2f}%, success rate: {:.2f}%, compression: {:.2f}%, abbreviation compression: {:.2f}%\n'
                    .format(ab.name(),
                            100*ab_results['accuracy'],
                            100*ab_results['abbreviation_success_rate'],
                            100*ab_results['eval_compression'],
                            100*ab_results['abbreviation_compression']))
        summary += ab_summary
        print(summary)

    path = 'results/abbreviator_experiment_{}.json'.format(datetime.datetime.now().isoformat())

    with open(path, 'w') as f:
        json.dump(results, f)

    print('Language Abbreviator ({} abbreviations) experiment finished. Results saved to {}. Summary: \n{}'
                 .format(len(targets), path, summary))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConadComplete utilities')

    parser.add_argument('--seed', help='Random seed')
    parser.add_argument('--language', default='Python', help='Programming Language to use (Python|Haskell|Java).')
    parser.add_argument('--dataset', default='medium', help='Dataset to use')
    parser.add_argument('--oneshot-dataset', help='One-shot dataset to use')
    parser.add_argument('--new-convention-every', default=100, type=int, help='Iterations between new conventions')
    parser.add_argument('--precompute-interactions', action='store_const', const=True, default=False)
    parser.add_argument('--build-oneshot-dataset', action='store_const', const=True, default=False)
    parser.add_argument('--train-decoders',
                        help='Pre-train all 4 decoders with different contexts.',
                        action='store_const', const=True, default=False)
    parser.add_argument('--train-set-embedding',
                        help='Pre-train set embedding.',
                        action='store_const', const=True, default=False)
    parser.add_argument('--train-abbreviator',
                        help='Run LanguageAbbreviator experiment.',
                        action='store_const', const=True, default=False)
    parser.add_argument('--train-lm',
                        help='Train Language Model.',
                        action='store_const', const=True, default=False)
    parser.add_argument('--oneshot-eval-examples', type=int, default=100, help='How many positive/negative examples to fetch for each convention in the one-shot dataset')
    parser.add_argument('--abbreviations', type=int, default=1000, help='How many abbreviations scenarios to put in the one-shot dataset/use in abbreviator.')
    parser.add_argument('--one-convention', help='Limit to applying at most one convention per input.',
                        action='store_const', const=True, default=False)
    parser.add_argument('--contextual-dataset',
                        help='Path to dataset with contextualized lines.')
    parser.add_argument('--set-embedding',
                        help='Path to pre-trained Set Embedding model.')
    parser.add_argument('--contexts', default='',
                        help='Comma-separated list of context flags to consider. Default: all.')
    parser.add_argument('--decoders',
                        help='Prefix of path to decoders.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')
    parser.add_argument('--device',
                        default='cpu',
                        help='Device to use for PyTorch.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train for.')
    parser.add_argument('--params',
                        help='JSON file to read parameters from.')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.precompute_interactions:
        precompute_interactions(args.dataset, args.language, args.new_convention_every, args.one_convention)
    elif args.build_oneshot_dataset:
        build_oneshot_dataset(args.dataset, args.language,
                              args.oneshot_eval_examples, args.abbreviations,
                              UniformAbbreviation(0.2))
    elif args.train_set_embedding:
        train_set_embedding(
                args.contextual_dataset,
                torch.device(args.device),
                args.output,
                args.epochs)
    elif args.train_decoders:
        train_decoders(
                args.params,
                torch.device(args.device),
                args.contexts,
        )
    elif args.train_lm:
        train_language_model(
                args.params,
                torch.device(args.device),
                args.contexts,
        )
    elif args.train_abbreviator:
        run_abbreviator_experiment(
                args.dataset,
                args.set_embedding,
                args.abbreviations,
                torch.device(args.device),
                args.decoders,
                args.contexts,
        )
