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
import dateutil.parser
import numpy as np
import pandas as pd
import repos
import logging
import random
import statsmodels.api as sm

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
                self.targets[language] = do.cap_collisions(
                        do.build_abbreviation_targets(
                            self.settings['n_targets'], dataset['train']),
                        self.settings.get('max_collisions', self.settings['n_targets']))

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
        self.loaded = False

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

        # If all models were evaluated, we don't need to load data anymore.
        if len(evaluated) == len(self.settings['languages']) * len(self.settings['max_collisions_cap']):
            self.loaded = True

        return '\n'.join(lines)

    def _get_clm_params(self, language, max_collisions):
        params = copy.deepcopy(self.settings['params']['clm'])
        params['dataset'] = self.settings['languages'][language]
        params['max_collisions'] = max_collisions
        return params

    def _load_data(self):
        if self.loaded:
            return

        for language in self.settings['languages']:
            with open(self.settings['languages'][language]) as f:
                dataset = json.load(f)
                self.datasets[language] = dataset
                self.targets[language] = do.build_abbreviation_targets(
                        self.settings['n_targets'], dataset['train'])

        self.loaded = True

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

        ls = list(self.settings['languages'].keys())
        data = [[ (self.results[self._get_model_name(l, c)]['accuracy'],
                    self.results[self._get_model_name(l, c)]['eval_compression'])
                    for c in self.settings['max_collisions_cap']]
                    for l in ls]
        return data

        plots.plot_accuracy_compression_combined(
                data,
                ls,
                output='experiments/{}-combined.png'.format(self.id))

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

class FineTuningExperiment(Experiment):
    '''
    Runs the fine tuning experiment.
    Parameters:
    - repos_dataset: dataset with repositories
    - top_n: how many of the top repositories to filter
    - n_repos: how many repositories to run on
    - pretrained: path to the pre-trained model
    '''
    def __init__(self, id):
        super().__init__(id)
        self.settings = {}
        self.loaded = False

    def load(self):
        with open(self.get_path()) as f:
            data = json.load(f)
            self.settings = data['settings']
            self.state = data.get('state', {})
            self.results = data.get('results', {})

    def state_description(self):
        lines = ['Fine Tuning Experiment: {} of the top {} repos from {}'
                 .format(self.settings['n_repos'], self.settings['top_n'],
                         self.settings['repos_dataset'])]

        lines.append('Evaluated: {}/{}'.format(len(self.results), self.settings['n_repos']))

        return '\n'.join(lines)

    def save(self):
        with open(self.get_path(), 'w') as f:
            json.dump({
                'settings': self.settings,
                'state': self.state,
                'results': self.results,
                }, f, indent=4)

    def run(self, device):
        if not self.loaded:
            logging.debug('Loading dataset...')
            self.dataset = json.load(open(self.settings['repos_dataset']))
            logging.debug('Loading pre-trained model...')
            self.clm = CLMLanguageAbbreviator.load(self.settings['pretrained'], device)
            self.loaded = True
            dataset, clm = self.dataset, self.clm

        # Step 1: Choose repositories to run on
        if not self.state.get('repos_chosen'):
            repos_by_size = [(k, len(v)) for k, v in dataset.items()]
            repos_by_size.sort(key=lambda kv: kv[1], reverse=True)
            largest_repos = [k for (k, v) in repos_by_size[:self.settings['top_n']]]

            self.state['repos_chosen'] = random.sample(largest_repos, self.settings['n_repos'])
            logging.debug('Chose repositories: {}'.format(self.state['repos_chosen']))
            self.save()

        # Step 2: Fine-tune and evaluate on each repository.
        results = []
        for i, repo in enumerate(self.state['repos_chosen']):
            if repo in self.results:
                results.append(self.results[repo])
            else:
                logging.debug('Running %d/%d (%s)', 
                              i+1, len(self.state.get('repos_chosen')), repo)
                repo_dataset = repos.build_repo_dataset(dataset[repo])
                r = repos.specialize_abbreviator(clm, repo_dataset)
                self.results[repo] = r
                self.save()
                results.append(r)

                logging.debug('Finished %s. Compression: %.3f => %.3f. Accuracy: %.3f => %.3f',
                              repo, r['compression_before'], r['compression_after'],
                              r['accuracy_before'], r['accuracy_after_ft'])

        # Step 3: Generate plots
        acc_original = [r['accuracy_before'] for r in results]
        acc_before_ft = [r['accuracy_before_ft'] for r in results]
        acc_after_ft = [r['accuracy_after_ft'] for r in results]

        comp_original = [r['compression_before'] for r in results]
        comp_after_ft = [r['compression_after'] for r in results]

        accuracy_path = '{}-accuracy.png'.format(self.id)
        compression_path = 'experiments/{}-compression-combined.png'.format(self.id)

        plots.plot_accuracy_distributions_fine_tuning(
                acc_original, acc_before_ft, acc_after_ft, accuracy_path)

        plots.plot_compression_distributions_fine_tuning(comp_original, comp_after_ft, 
                compression_path)

        print('Accuracy median before:', np.median(acc_original))
        print('Accuracy median before ft:', np.median(acc_before_ft))
        print('Accuracy median after ft:', np.median(acc_after_ft))

        print('Compression median before:', np.median(comp_original))
        print('Compression median after:', np.median(comp_after_ft))

        logging.info('Generated plots:', accuracy_path, compression_path)

class UserStudyExperiment(Experiment):
    '''
    Generates results from the raw user study data.
    Since it involves no compute, it doesn't save any state, simply
    computing the outputs every time it is ran.
    '''
    def __init__(self, id):
        super().__init__(id)
        self.settings = {}
        self.loaded_data = False

    def load(self):
        with open(self.get_path()) as f:
            data = json.load(f)
            self.settings = data['settings']

    def state_description(self):
        return 'User Study Experiment (no state)'

    def run(self, device=None, plot=True):
        if not self.loaded_data:
            with open(self.settings['raw_data']) as f:
                self.raw_data = json.load(f)
            self.loaded_data = True

        raw_data = self.raw_data

        SETTINGS = [
            'None',
            'VSCode',
            'Pragmatic',
            'VSCode+Pragmatic'
        ]

        LETTER_KEYS = set('Key' + chr(l) for l in range(ord('A'), ord('Z') + 1))

        def get_settings_order(session):
            order = []

            for l in raw_data['logs_by_session'][session]:
                s = l['events'][0]['setting']
                if s not in order:
                    order.append(s)

            return ''.join(map(str, order))

        typing_speeds_unnorm = [[] for _ in range(len(SETTINGS))]
        typing_speeds = [[] for _ in range(len(SETTINGS))]
        keystrokes_unnorm = [[] for _ in range(len(SETTINGS))]
        keystrokes = [[] for _ in range(len(SETTINGS))]
        user_task_baseline_speed = {}
        user_task_baseline_keystrokes = {}

        data_points = []

        for u, user in enumerate(raw_data['complete_sessions']):
            for s, setting in enumerate(user):
                for t, task in enumerate(setting):
                    events, target, session = task['events'], task['target'], task['session']

                    start_event = next(filter(lambda e: e['type'] == 'start', events))
                    end_event = next(filter(lambda e: e['type'] == 'end', events))
                    n_keystrokes = len(list(filter(lambda e: e['type'] == 'key-press', events)))
                    n_submits = len(list(filter(lambda e: e['type'] == 'submit', events)))
                    n_backspace = len(list(filter(lambda e: e['type'] == 'key-press' and
                                      e['key'] == 'Backspace', events)))
                    n_letters = len(list(filter(lambda e: e['type'] == 'key-press' and
                                      e['key'] in LETTER_KEYS, events)))

                    start_event_time = dateutil.parser.parse(start_event['timestamp'])
                    end_event_time = dateutil.parser.parse(end_event['timestamp'])
                    len_seconds = (end_event_time - start_event_time).total_seconds()

                    # Compute effective typing speed as characters-per-minute (CPM):
                    user_task_speed = 60 * len(task['target']) / len_seconds
                    user_task_keystrokes = n_keystrokes

                    # Normalize against setting 0.
                    if s == 0:
                        user_task_baseline_speed[u, target] = user_task_speed
                        user_task_baseline_keystrokes[u, target] = user_task_keystrokes

                    if t >= 3:
                        typing_speeds[s].append(user_task_speed /
                                                user_task_baseline_speed[u, target])
                        keystrokes[s].append(user_task_keystrokes /
                                             user_task_baseline_keystrokes[u, target])

                    typing_speeds_unnorm[s].append(user_task_speed)

                    printable_characters = sum(1 for c in task['target'] if c.isprintable())

                    data_points.append(
                        (
                            s,           # Autocomplete setting.
                            t,           # Number of the snippet.
                            u,           # User.
                            get_settings_order(session), # Order of settings that user got.
                            user_task_speed, # Typing speed (unnormalized).
                            user_task_speed / # Typing speed (normalized).
                                user_task_baseline_speed[u, target], 
                            user_task_keystrokes, # Keystrokes (unnormalized).
                            user_task_keystrokes / # Keystrokes (normalized).
                                user_task_baseline_keystrokes[u, target],
                            n_submits,   # Number of times user tried to submit the code (1 if no mistakes).
                            n_backspace, # Number of times the user pressed the backspace key.
                            n_letters /  # Number of times the user pressed a letter key.
                                printable_characters,
                        )
                    )



        dataset = pd.DataFrame(
            data_points,
            columns=['Setting', 'Snippet', 'User', 'Order', 'TypingSpeed', 'NormalizedTypingSpeed',
                     'Keystrokes', 'NormalizedKeystrokes',
                     'Submits', 'Backspace', 'LetterKeys']
        )

        dataset['VSCode'] = (dataset['Setting'] == 1) | (dataset['Setting'] == 3)
        dataset['Pragmatic'] = (dataset['Setting'] == 2) | (dataset['Setting'] == 3)

        for s, sname in enumerate(SETTINGS):
            dataset.loc[dataset['Setting'] == s, 'Autocomplete'] = sname

        ts_mean, ts_std = [], []
        ts_bs_lo, ts_bs_hi = [], []
        ks_mean, ks_std = [], []
        ks_bs_lo, ks_bs_hi = [], []

        for i, s in enumerate(SETTINGS):
            ts_mean.append(np.mean(typing_speeds[i]))
            ts_std.append(np.std(typing_speeds[i]) / np.sqrt(len(typing_speeds[i])))
            ks_mean.append(np.mean(keystrokes[i]))
            ks_std.append(np.std(keystrokes[i]) / np.sqrt(len(keystrokes[i])))

            print(s, 'typing speed:', np.mean(typing_speeds[i]), '+-', np.std(typing_speeds[i]),
                    'keystrokes:', np.mean(keystrokes[i]), '+-', np.std(keystrokes[i]))

        ts_mean = np.array(ts_mean)
        ts_std = np.array(ts_std)
        ks_mean = np.array(ks_mean)
        ks_std = np.array(ks_std)

        if plot:
            plots.plot_user_study_metric('Typing Speed', SETTINGS, ts_mean, ts_std, 'results/{}-typingspeed.png'.format(self.id))
            plots.plot_user_study_metric('Keystrokes', SETTINGS, ks_mean, ks_std, 'results/{}-keystrokes.png'.format(self.id))

        plots.plot_user_study_typing_speed(
                dataset, 'experiments/{}-typing-speed.png'.format(self.id))
        plots.plot_user_study_keystrokes(
                dataset, 'experiments/{}-keystrokes.png'.format(self.id))

        return dataset
 
def run_experiment(class_, id, device):
    e = class_(id)
    e.load()
    print(e.state_description())
    e.run(device)

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s]   %(message)s')
    logging.getLogger().setLevel(0)

    parser = argparse.ArgumentParser('Driver for all experiments in the paper.')
    parser.add_argument('--id', required=True, help='Experiment ID.')
    parser.add_argument('-a', '--accuracy', help='Run accuracy experiment (#1)',
                        action='store_const', const=True, default=False)
    parser.add_argument('-m', '--ambiguity', help='Run ambiguity experiment (#2)',
                        action='store_const', const=True, default=False)
    parser.add_argument('-f', '--fine-tuning', help='Fine tuning experiment (#3)',
                        action='store_const', const=True, default=False)
    parser.add_argument('-u', '--user-study', help='Run user study experiment (#4)',
                        action='store_const', const=True, default=False)
    parser.add_argument('--device', help='Device to run on', default='cpu')

    opt = parser.parse_args()

    if opt.accuracy:
        run_experiment(AccuracyExperiment, opt.id, torch.device(opt.device))
    elif opt.ambiguity:
        run_experiment(AmbiguityExperiment, opt.id, torch.device(opt.device))
    elif opt.fine_tuning:
        run_experiment(FineTuningExperiment, opt.id, torch.device(opt.device))
    elif opt.user_study:
        run_experiment(UserStudyExperiment, opt.id, torch.device(opt.device))
