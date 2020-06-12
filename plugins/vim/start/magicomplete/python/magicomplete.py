import vim
import sys
import os

def add_base_to_path():
    LEVELS_ABOVE = 6
    root = os.path.abspath(__file__)

    for i in range(LEVELS_ABOVE):
        root, _ = os.path.split(root)

    sys.path.append(root)

add_base_to_path()
from plugins.base import TextEditorPluginBackend

backend = TextEditorPluginBackend()

def show_message(message):
    vim.command('call popup_atcursor(' + repr(message) + ', {"time": 5000})')

def handle_expand():
    row, col = vim.current.window.cursor
    if row >= 1:
        current_line = vim.current.buffer[row-1]
        new_l, b, a = backend.expand(current_line, [], [])
        backend.record_hint(new_l)
        vim.current.buffer[row-1] = b + new_l + a

def handle_change():
    row, col = vim.current.window.cursor
    if row >= 2:
        last_line = vim.current.buffer[vim.current.window.cursor[0] - 2]
        current_line = vim.current.buffer[vim.current.window.cursor[0] - 1]
        last_line_core = last_line.strip()

        if len(current_line.strip()) == 0 and \
                not backend.hint_was_shown(last_line.strip()):
            hint = backend.get_hint(last_line)
            if hint is not None:
                show_message('Hint: ' + hint)
                backend.record_hint(last_line.strip())
