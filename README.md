# Pragmatic Code Autocomplete

The main model in the paper is implemented in abbreviator.py,
in the class CLMLanguageAbbreviator.

To run all experiments we reported, there are scripts that will
do everything from training the models from scratch to generating
LaTeX tables, plots and printing the numerical results
we mention. These are concentrated in experiments.py

As a sample dataset, we're providing lines-small-py.json,
which contains lines of code in Python with a context of 10 lines.

Here's how to run each of the 4 experiments (substitute cuda:0 for
the appropriate GPU). Each experiment is fully described in a JSON
file in the 'experiments' directory.

## Experiment 1 - accuracy and context ablation

$ python experiments.py --accuracy --id accuracy-small --device cuda:0

## Experiment 2 - varying the level of ambiguity

$ python experiments.py --ambiguity --id ambiguity-small --device cuda:0

## Experiment 3 - fine-tuning to different repositories

$ gunzip repos-python-small.json.gz
$ python experiments.py --fine-tuning --id fine-tuning-small --device cuda:0

## Experiment 4 - user study

All the raw data we collected from our participants is in
results/user-study-data.json.2020-08-31T23-31-35.578975

To run the analyses, simply run:

$ python experiments.py --user-study --id user-study

The two pre-trained models that are needed for the fine-tuning experiment and
in the user study are included in models/ (the configuration files already point to them,
so there's nothing to do).
