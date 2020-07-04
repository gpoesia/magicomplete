import torch
import json
from abbreviator import *
from flask import Flask, request

print('Loading model...')
# 9ce32: 98.96% accuracy (CLM: 64d88)
# 2dcfe: 97.80% accuracy
# c90f3: 98.56% accuracy
# dce22: 97.28% accuracy
# 5fa60: 96.09% accuracy
abbreviator = CLMLanguageAbbreviator.load('models/64d88.model', device=torch.device('cpu'))

app = Flask(__name__)

@app.route('/keywords')
def get_keywords():
    return json.dumps(abbreviator.abbreviation_targets)

@app.route('/context_size')
def get_context_size():
    return json.dumps(abbreviator.clm.context.n_previous_lines)

@app.route('/complete')
def complete():
    l = request.args.get('l', '')
    p = json.loads(request.args.get('p', '[]'))
    input = {'l': l, 'p': p, 'i': [], 'c': []}
    completions = abbreviator.beam_search([input], 2)[0]
    return json.dumps(completions[0])
