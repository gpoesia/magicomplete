# Utility to record progress and measure ETA of long-running tasks.

import datetime
import time

class Progress:
    def __init__(self, total_iterations=None, timeout=None):
        self.total_iterations = total_iterations
        self.timeout = timeout
        self.begin = time.time()
        self.current_iteration = 0

    def tick(self):
        self.current_iteration += 1

    def eta(self):
        now = time.time()

        if self.total_iterations is not None:
            return format_eta((self.total_iterations - self.current_iteration) *
                                   (now - self.begin) / max(1, self.current_iteration))

        if self.timeout is not None:
            return format_eta(max(0, self.timeout - (now - self.begin)))

        return "???"

def format_eta(seconds):
    hours = seconds // (60*60)
    minutes = seconds // 60
    seconds %= 60

    s = ""

    if hours > 0:
        s += "{:02}h".format(hours)

    if minutes > 0 or hours > 0:
        s += "{:02}m".format(minutes)

    s += "{:02}s".format(seconds)

    return s
