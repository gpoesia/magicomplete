# Script to train all models and run all experiments that will go in the paper.

import torch
import json
import argparse
import do
import copy
from models import load_from_run
from language_model import RNNLanguageModel
from util import random_hex_string
from abbreviator import LMRLanguageAbbreviator, CLMLanguageAbbreviator
import plots

class Experiment:
    '''An experiment encodes the process of running it, the state transitions,
    saves the state to disk to output_path, and can load the last saved state
    and continue from it.'''

    def __init__(self, id):
        self.id = id

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def state_description(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def get_path(self):
        return 'experiments/{}.json'.format(self.id)

def format_table_entry(entry, precision=3, bold=False):
    if type(entry) is float:
        s = ('{:.' + str(precision) + 'f}').format(entry)
    else:
        s = str(entry)
    return s if not bold else '\\textbf{' + s + '}'

class AccuracyExperiment(Experiment):
    '''
    Runs models and compute test accuracy.

    settings:
        - run_clm: whether or not to run the CLM + ranking model
        - clm_contexts: if running CLM, list of context settings to run it.
        - run_lm: whether or not to run the LM + ranking model
        - clm_params: params to the clm model
        - lm_params: params to the lm model
    '''
    def __init__(self, id):
        super().__init__(id)
        self.settings = {}
        self.state = {}
        self.results = {}
        self.targets = {}
        self.datasets = {}

    def load(self):
        with open(self.get_path()) as f:
            data = json.load(f)
            self.settings = data['settings']
            self.state = data.get('state', {})
            self.results = data.get('results', {})

    def save(self):
        with open(self.get_path(), 'w') as f:
            json.dump({
                'settings': self.settings,
                'state': self.state,
                'results': self.results,
                }, f, indent=4)

    def _get_model_name(self, type, language, ctx_lines=None):
        language_prefix = '{}|'.format(language) if language else ''

        if type == 'CLM':
            return '{}CLM, ctx size {}'.format(language_prefix, ctx_lines)
        else:
            return '{}LM'.format(language_prefix)

    def state_description(self):
        lines = ['Accuracy Experiment \'{}\''.format(self.id)]

        evaluated = []

        if self.settings.get('run_clm'):
            for language in self.settings['languages']:
                for ctx in self.settings.get('clm_context_lines', []):
                    model_id = self._get_model_name('CLM', language, ctx)
                    lines.append('{} trained: {}'
                            .format(
                                model_id,
                                self.state.get('clm_trained[{}]'.format(model_id), 'no')))
                    if self.results.get(model_id):
                        evaluated.append(model_id)
        else:
            lines.append('Not running CLM')

        if self.settings.get('run_lm'):
            for language in self.settings['languages']:
                model_id = self._get_model_name('LM', language)
                lines.append('{} trained: {}'.format(
                    model_id,
                    self.state.get('lm_trained[{}]'.format(model_id), 'no')))

                if self.results.get(model_id):
                    evaluated.append(model_id)
        else:
            lines.append('Not running LM')

        lines.append('Models evaluated: {}'.format(', '.join(evaluated)))
        return '\n'.join(lines)

    def _get_clm_params(self, language, context_lines):
        params = copy.deepcopy(self.settings['params']['clm'])
        params['dataset'] = self.settings['languages'][language]
        params['clm']['n_previous_lines'] = context_lines
        params['n_targets'] = self.settings['n_targets']
        return params

    def _get_lm_params(self, language):
        params = copy.deepcopy(self.settings['params']['lm'])
        params['dataset'] = self.settings['languages'][language]
        return params

    def _make_lmr_abbreviator(self, lm_id, language, device):
        lm = load_from_run(RNNLanguageModel, lm_id, device)
        abbreviator = LMRLanguageAbbreviator(lm, self.datasets[language]['train'],
                                             self.settings['params']['lmr'])
        abbreviator.build_optimal_abbreviation_table(self.targets[language])
        return abbreviator

    def _load_data(self):
        for language in self.settings['languages']:
            with open(self.settings['languages'][language]) as f:
                dataset = json.load(f)
                self.datasets[language] = dataset
                self.targets[language] = do.build_abbreviation_targets(
                        self.settings['n_targets'], dataset['train'])

    def run(self, device):
        self._load_data()

        ####
        #### Step 1: train models.
        ####

        # CLM
        if self.settings.get('run_clm'):
            for language in self.settings['languages']:
                for ctx_lines in self.settings.get('clm_context_lines', []):
                    id = self._get_model_name('CLM', language, ctx_lines)
                    params = self._get_clm_params(language, ctx_lines)

                    trained_key = 'clm_trained[{}]'.format(id)

                    if not self.state.get(trained_key):
                        print('Training', id)
                        run_id = do.train_clm_abbreviator(
                                params,
                                device,
                                'NONE' if ctx_lines == 0 else 'PREVIOUS_LINES',
                                False)[0]

                        self.state[trained_key] = run_id
                        self.save()
                    else:
                        print('Skipping already trained', id)
        # LM
        if self.settings.get('run_lm'):
            for language in self.settings['languages']:
                id = self._get_model_name('LM', language)
                trained_key = 'lm_trained[{}]'.format(id)

                if not self.state.get(trained_key):
                    print('Training', id)
                    params = self._get_lm_params(language)
                    run_id = do.train_language_model(params, device, 'NONE')[0]
                    self.state[trained_key] = run_id
                    self.save()
                else:
                    print('Skipping already trained', id)

        ####
        # Step 2: evaluate
        ####
        if self.settings.get('run_clm'):
            for language in self.settings['languages']:
                for ctx_lines in self.settings.get('clm_context_lines', []):
                    id = self._get_model_name('CLM', language, ctx_lines)

                    if not self.results.get(id):
                        print('Evaluating', id)
                        run_id = self.state['clm_trained[{}]'.format(id)]
                        abbrev = CLMLanguageAbbreviator.load('models/{}.model'.format(run_id),
                                                             device)

                        eval_id, results = do.evaluate_abbreviator(
                                self.datasets[language],
                                self.targets[language],
                                abbrev,
                                abbrev.params,
                                'test')

                        self.results[id] = {
                                **results,
                                'run_id': eval_id,
                                }
                        self.save()
                    else:
                        print('Skipping already evaluated', id)

        if self.settings.get('run_lm'):
            for language in self.settings['languages']:
                id = self._get_model_name('LM', language)

                if not self.results.get(id):
                    print('Evaluating', id)
                    trained_id = self.state['lm_trained[{}]'.format(id)]

                    abbrev = self._make_lmr_abbreviator(trained_id, language, device)

                    eval_id, results = do.evaluate_abbreviator(
                            self.datasets[language],
                            self.targets[language],
                            abbrev,
                            abbrev.parameters,
                            'test')

                    self.results[id] = {
                            **results,
                            'run_id': eval_id,
                            }
                    self.save()
                else:
                    print('Skipping already evaluated', id)

        ####
        # Step 3: generate LaTeX table
        ####
        print('Generating results table...')
        table = self.generate_latex_table()

        with open('results/{}_results_table.tex'.format(self.id), 'w') as f:
            f.write(table)

    def generate_latex_table(self):
        model_names = []
        max_by_column = [0] * len(self.settings['languages'])
        rows = []

        if self.settings.get('run_lm'):
            id = self._get_model_name('LM', None)
            row = [id]

            for i, language in enumerate(self.settings['languages']):
                id_lang = self._get_model_name('LM', language)
                acc = self.results[id_lang]['accuracy']
                if acc > max_by_column[i]:
                    max_by_column[i] = acc
                row.append(acc)

            rows.append(row)

        if self.settings.get('run_clm'):
            for ctx_lines in self.settings.get('clm_context_lines', []):
                id = self._get_model_name('CLM', None, ctx_lines)
                row = [id]

                for i, language in enumerate(self.settings['languages']):
                    id_lang = self._get_model_name('CLM', language, ctx_lines)
                    acc = self.results[id_lang]['accuracy']
                    if acc > max_by_column[i]:
                        max_by_column[i] = acc
                    row.append(acc)
                rows.append(row)

        # Format table line by line
        lines = []
        lines.append('\\begin{tabular}{|c|' +
                     ('c|' * len(self.settings['languages'])) +
                     '}')
        lines.append('    \\hline & ' + ' & '.join(self.settings['languages'].keys()) + ' \\\\')

        for row in rows:
            lines.append(
                    '    \\hline ' +
                    ' & '.join([format_table_entry(v, 3, i >= 1 and v == max_by_column[i-1])
                                     for i, v in enumerate(row)]) +
                    ' \\\\')

        lines.append('\\hline')
        lines.append('\\end{tabular}')
        lines.append('')
        return '\n'.join(lines)

class AmbiguityExperiment(Experiment):
    '''
    Runs models with varying numbers of maximum collisions,
    compute test accuracy and compression.

    settings:
        - max_collisions_cap: list of number of max collisions to run with.
        - languages: list of languages to run.
        - clm_params: params to the clm model
    '''
    def __init__(self, id):
        super().__init__(id)
        self.settings = {}
        self.state = {}
        self.results = {}
        self.targets = {}
        self.datasets = {}

    def load(self):
        with open(self.get_path()) as f:
            data = json.load(f)
            self.settings = data['settings']
            self.state = data.get('state', {})
            self.results = data.get('results', {})

    def save(self):
        with open(self.get_path(), 'w') as f:
            json.dump({
                'settings': self.settings,
                'state': self.state,
                'results': self.results,
                }, f, indent=4)

    def _get_model_name(self, language, max_collisions=None):
        if max_collisions == 1:
            return '{}|CLM: no collision'.format(language)
        else:
            return '{}|CLM: {} collisions'.format(language, max_collisions)

    def state_description(self):
        lines = ['Ambiguity Experiment \'{}\''.format(self.id)]

        evaluated = []

        for language in self.settings['languages']:
            for max_collisions in self.settings.get('max_collisions_cap', []):
                model_id = self._get_model_name(language, max_collisions)
                lines.append('{} trained: {}'
                            .format(
                                model_id,
                                self.state.get('clm_trained[{}]'.format(model_id), 'no')))
                if self.results.get(model_id):
                    evaluated.append(model_id)

        lines.append('Models evaluated: [{}]'.format(', '.join(evaluated)))
        return '\n'.join(lines)

    def _get_clm_params(self, language, max_collisions):
        params = copy.deepcopy(self.settings['params']['clm'])
        params['dataset'] = self.settings['languages'][language]
        params['max_collisions'] = max_collisions
        return params

    def _load_data(self):
        for language in self.settings['languages']:
            with open(self.settings['languages'][language]) as f:
                dataset = json.load(f)
                self.datasets[language] = dataset
                self.targets[language] = do.build_abbreviation_targets(
                        self.settings['n_targets'], dataset['train'])

    def run(self, device):
        ####
        #### Step 0: load data (if needed).
        ####
        need_data = False

        for language in self.settings['languages']:
            for max_collisions in self.settings.get('max_collisions_cap', []):
                id = self._get_model_name(language, max_collisions)

                if not self.results.get(id):
                    need_data = True

        if need_data:
            self._load_data()

        ####
        #### Step 1: train models.
        ####

        # CLM
        for language in self.settings['languages']:
            for max_collisions in self.settings.get('max_collisions_cap', []):
                id = self._get_model_name(language, max_collisions)
                params = self._get_clm_params(language, max_collisions)

                trained_key = 'clm_trained[{}]'.format(id)

                if not self.state.get(trained_key):
                    print('Training', id)
                    run_id = do.train_clm_abbreviator(
                            params,
                            device,
                            'PREVIOUS_LINES',
                            False)[0]

                    self.state[trained_key] = run_id
                    self.save()
                else:
                    print('Skipping already trained', id)

        ####
        # Step 2: evaluate
        ####
        for language in self.settings['languages']:
            for max_collisions in self.settings.get('max_collisions_cap', []):
                id = self._get_model_name(language, max_collisions)

                if not self.results.get(id):
                    print('Evaluating', id)
                    run_id = self.state['clm_trained[{}]'.format(id)]
                    abbrev = CLMLanguageAbbreviator.load('models/{}.model'.format(run_id), device)

                    eval_id, results = do.evaluate_abbreviator(
                            self.datasets[language],
                            self.targets[language],
                            abbrev,
                            abbrev.params,
                            'test')

                    self.results[id] = {
                            **results,
                            'run_id': eval_id,
                            }
                    self.save()
                else:
                    print('Skipping already evaluated', id)

        ####
        # Step 3: generate plots
        ####
        for i, language in enumerate(self.settings['languages']):
            is_first, is_last = i == 0, i + 1 == len(self.settings['languages'])
            accuracy, compression = [], []

            for max_collisions in self.settings.get('max_collisions_cap', []):
                id = self._get_model_name(language, max_collisions)
                accuracy.append(self.results[id]['accuracy'])
                compression.append(self.results[id]['eval_compression'])

            output_path = 'experiments/{}-{}.png'.format(self.id, language)

            plots.plot_accuracy_compression(accuracy, compression, language,
                                            y_label=is_first, legend=is_last,
                                            output=output_path)

            print('Generated', output_path)

def run_accuracy_experiment(id, device):
    e = AccuracyExperiment(id)
    e.load()
    print(e.state_description())
    e.run(device)

def run_ambiguity_experiment(id, device):
    e = AmbiguityExperiment(id)
    e.load()
    print(e.state_description())
    e.run(device)

parser = argparse.ArgumentParser('Driver for all experiments in the paper.')
parser.add_argument('--id', required=True, help='Experiment ID.')
parser.add_argument('-a', '--accuracy', help='Run accuracy experiment (#1)',
                    action='store_const', const=True, default=False)
parser.add_argument('-m', '--ambiguity', help='Run ambiguity experiment (#2)',
                    action='store_const', const=True, default=False)
parser.add_argument('--device', help='Device to run on', default='cpu')

opt = parser.parse_args()

if opt.accuracy:
    run_accuracy_experiment(opt.id, torch.device(opt.device))
elif opt.ambiguity:
    run_ambiguity_experiment(opt.id, torch.device(opt.device))
