# Tracks the run of a model, saves results and metrics, checkpoints, etc.

import threading
import json
import numpy as np
import datetime
from pprint import pformat

from util import random_hex_string
from torch.utils.tensorboard import SummaryWriter

class RunTracker:
    def __init__(self, model, parameters):
        self.data = {}
        self.run_id = random_hex_string()
        self.writer = SummaryWriter(comment=" run_id={}".format(self.run_id))
        self.model = model
        self.current_step = 0
        self.n_checkpoints = 0
        self.parameters = parameters

        self.timer = None

    def start(self):
        self.timer = threading.Timer(60, self.flush)
        self.timer.start()
        self.data['tracker:start-time'] = datetime.datetime.now().isoformat()

        self.data['params'] = self.parameters
        self.writer.add_text('params', pformat(self.parameters))

    def step(self):
        self.current_step += 1

    def add_to_list(self, tag, item):
        if tag not in self.data:
            self.data[tag] = []
        self.data[tag].append(item)

    def add_scalar(self, tag, value):
        self.writer.add_scalar(tag, value, self.current_step)
        self.add_to_list(tag, {'step': self.current_step, 'value': value})

    def report_scalar(self, tag, value):
        self.data[tag] = value
        self.writer.add_text(tag, '{:.4f}'.format(value))

    def checkpoint(self):
        self.n_checkpoints += 1
        self.model.dump('models/{}_check{}.model'.format(self.run_id, self.n_checkpoints))

    def loss_converged(self, loss_tag, window_size=10000, minimum_improvement=0.98):
        losses = self.data.get(loss_tag) or []
        if len(losses) < 2*window_size:
            return False
        return (np.mean(losses[-window_size:]) >
                minimum_improvement * np.mean(losses[-2*window_size:-window_size]))

    def flush(self):
        self.writer.flush()
        with open('runs/{}.json'.format(self.run_id), 'w') as f:
            json.dump(self.data, f)

    def close(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        self.data['tracker:end-time'] = datetime.datetime.now().isoformat()
        self.model.dump('models/{}.model'.format(self.run_id))
        self.flush()
        self.writer.close()
