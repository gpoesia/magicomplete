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
    vim.command('call popup_atcursor(' + repr(message) + ', {"time": 3000})')

def handle_expand():
    row, col = vim.current.window.cursor
    current_line = vim.current.buffer[row-2]
    vim.current.buffer[row-2] = backend.expand(current_line, [], [])
    show_message('Magic!')

def handle_newline():
    line = vim.current.buffer[vim.current.window.cursor[0] - 2]
    hint = backend.get_hint(line)
    if hint is not None:
        show_message('Hint:' + hint)
