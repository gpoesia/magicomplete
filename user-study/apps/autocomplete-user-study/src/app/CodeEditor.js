import React from 'react';
import MonacoEditor from 'react-monaco-editor';

export const AutocompleteSetting = {
    NONE: 0,
    DEFAULT: 1,
    PRAGMATIC: 2,
};

const trim = (l) => {
    const tl = l.trimLeft();
    return [l.substr(0, l.length - tl.length), tl];
}

export default class CodeEditor extends React.Component {
    constructor() {
        super();
        this.contextSize = 5;

        this.state = {
            code: ''
        };

        this.keywords = new Set();
        this.line = 1;
    }

    onChange(newValue, e) {
        this.setState({
            code: newValue,
        });
        if (this.props.onChange) {
            this.props.onChange(newValue);
        }
    }

    mountEditor(e) {
        e.focus();
        e.onDidChangeCursorPosition(evt => {
            this.line = evt.position.lineNumber;
            if (this.props.setting == AutocompleteSetting.PRAGMATIC) {
                this.updateHighlighting(e, evt.position);
            }
        });
        if (this.props.setting == AutocompleteSetting.PRAGMATIC) {
            e.addAction({
            id: 'magicomplete',
            label: 'Pragmatic Code Autocomplete feature',
            keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
            contextMenuGroupId: 'navigation',
            contextMenuOrder: 1.5,
            run: ed => this.complete(ed),
          });
        }
        if (this.props.submitCode) {
            e.addAction({
              id: 'submit',
              label: 'Submit the code currently typed out.',
              keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KEY_S],
              contextMenuGroupId: 'navigation',
              contextMenuOrder: 1.5,
              run: ed => {
                  if (this.props.submitCode(this.state.code)) {
                      this.setState({ code: '' });
                  }
                }
            });
        }
    }

    updateHighlighting(e, position) {
        const line = position.lineNumber;
        const model = e.getModel();
        for (let i = Math.max(1, line - this.contextSize); i <= line; i++) {
            updateLineDecorations(i, model, this.keywords);
        }
    }

    componentDidMount() {
        this._loadKeywords();
    }

    _loadKeywords() {
        fetch('/keywords')
            .then(r => r.json())
            .then(keywords => {
                console.log('Received ', keywords.length, 'keywords');
                console.log(keywords);
                this.keywords = new Set(keywords);
            });
    }

    complete(editor) {
        const model = editor.getModel();
        const lineContents = model.getLineContent(this.line);

        let previous = [];

        for (let i = Math.max(1, this.line - this.contextSize); i < this.line; i++) {
            previous.push(model.getLineContent(i));
        }

        const [indent, trimmedL] = trim(lineContents);
        const l = encodeURIComponent(trimmedL);
        const p = encodeURIComponent(JSON.stringify(previous));

        fetch(`/complete?l=${l}&p=${p}`)
            .then(r => r.json())
            .then(completion => {
                editor.executeEdits(
                    'Magicomplete',
                    [{
                        range: new monaco.Range(this.line, 1,
                                                this.line, lineContents.length + 1),
                        text: indent + completion,
                        forceMoveMarkers: true,
                    }],
                    () => [new monaco.Selection(this.line, lineContents.length + 1,
                                                this.line, lineContents.length + 1)],
                );
                editor.getAction('editor.action.insertLineAfter').run();
            });
    }

    render() {
        const { setting } = this.props;
        const defaultAutocomplete = setting == AutocompleteSetting.DEFAULT;

        return (
            <MonacoEditor
                language="python"
                theme="vs-dark"
                value={this.state.code}
                onChange={(nV, e) => this.onChange(nV, e)}
                editorDidMount={e => this.mountEditor(e)}
                automaticLayout={true}
                options={{
                    tabCompletion: defaultAutocomplete ? "on" : "off",
                    quickSuggestions: defaultAutocomplete,
                    snippetSuggestions: defaultAutocomplete ? true : "none",
                }}
            />
        );
    }
}

const IDENTIFIER_CHAR = /[a-zA-Z_]/;
const DIGIT = /[0-9]/;

function splitAtIdentifierBoundaries(s) {
    const tokens = [];
    let isInId = false;

    s.split('').forEach(c => {
        if (c.match(IDENTIFIER_CHAR) || (isInId && c.match(DIGIT))) {
            if (!isInId) {
                tokens.push('');
                isInId = true;
            }
        } else if (isInId || tokens.length == 0) {
            tokens.push('');
            isInId = false;
        }
        tokens[tokens.length - 1] += c;
    });

    return tokens;
}

function updateLineDecorations(line, model, strings) {  
    let decorations = model.getLineDecorations(line).map(d => d.id);
    let newDecorations = [];

    const tokens = splitAtIdentifierBoundaries(
        model.getLineContent(line));

    let column = 0;
    tokens.forEach(t => {
        if (strings.has(t)) {
            newDecorations.push({
                range: new monaco.Range(line, column + 1, line, column + t.length + 1),
                options: { inlineClassName: 'magicompleteHighlight' }},
            );
        }
        column += t.length;
    });

    model.deltaDecorations(decorations, newDecorations);
}