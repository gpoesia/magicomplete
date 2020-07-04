import React from 'react';
import Progress from './Progress.js';
import MonacoEditor from 'react-monaco-editor';
import CodeEditor, { AutocompleteSetting } from './CodeEditor';

import leven from 'leven';
import _ from 'underscore';

const MAX_ERRORS = 5;

function computeErrors(code, target) {
    return leven(code.replace(/\s/g, ''),
                 target.replace(/\s/g, ''));
}

const NUMBER_OF_SETTINGS = _.keys(AutocompleteSetting).length;

class TypingTask extends React.Component {
    constructor(props) {
        super(props);
        this.events = [];

        this.settingsOrder = _.shuffle(_.range(NUMBER_OF_SETTINGS));

        this.state = {
            started: false,
            currentSetting: 0,
            currentTarget: 0,
            instructionsRead: false,
            taskBegin: new Date(),
        };

        this.session = Math.random().toString();
    }

    onChange(newValue) {
        this._pushEvent({ 'type': 'change', 'value': newValue });
    }

    componentDidMount() {
        fetch('/dataset')
            .then((r) => r.json())
            .then(dataset => {
                dataset = dataset.slice(0, 1);
                this.setState({ dataset });
            });
        this.updateInterval = window.setInterval(() => this.forceUpdate(), 500);
    }

    componentWillUnmount() {
        clearInterval(this.updateInterval);
    }

    _start() {
        this._pushEvent({ 'type': 'start', setting: this._currentSetting() });
        this.setState({
            started: true,
            taskBegin: new Date(),
        });
    }

    saveEvents(target) {
        return fetch('/save-events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                'session': this.session,
                'target': target,
                'setting': this._currentSetting(),
                'events': this.events, 
            }),
        }).then(() => {
            this.events = [];
        });
    }

    submitCode(code) {
        const target = this.state.dataset[this.state.currentTarget];
        this._pushEvent({ 'type': 'submit' });

        if (computeErrors(code, target) <= MAX_ERRORS) {
            this._pushEvent({ 'type': 'end' });
            this.saveEvents(target).then(() => {
                // After the last code snippet, move to the next setting.
                if (this.state.currentTarget + 1 == this.state.dataset.length) {
                    this.setState({
                        currentTarget: 0,
                        currentSetting: this.state.currentSetting + 1,
                        started: false,
                    });
                } else {
                    // Otherwise, just move to the next snippet.
                    this.setState({
                        currentTarget: this.state.currentTarget + 1,
                    });
                    this._start();
                }
            });
            return true;
        } else {
            this.setState({ error: 'Your code is too different from the target to submit.' });
            return false;
        }
    }

    _pushEvent(event) {
        this.events.push({ 
            ...event, 
            timestamp: new Date().toISOString()
        });
    }

    _currentSetting() {
        return this.settingsOrder[this.state.currentSetting];
    }

    render() {
        if (!this.state.dataset) {
            return <p>Loading dataset...</p>
        }

        if (this.state.currentSetting == NUMBER_OF_SETTINGS) {
            return <p>Thank you for participating! You can now close this page.</p>
        }

        const enableCompletion = (this.settingsOrder[this.state.currentSetting] == AutocompleteSetting.DEFAULT);

        const settingDescription = {
            [AutocompleteSetting.NONE]: "No autocomplete",
            [AutocompleteSetting.DEFAULT]: "VSCode's default autocomplete",
            [AutocompleteSetting.PRAGMATIC]: "Pragmatic autocomplete",
        }[this._currentSetting()];

        const settingInstructions = {
            [AutocompleteSetting.NONE]:
                <p>In this setting, the editor's autocomplete feature is disabled</p>,
            [AutocompleteSetting.DEFAULT]: 
                <p>
                    In this setting, you'll use VSCode's default autocomplete feature.
                    It suggests completions as you type, and you can press Enter to accept them.
                </p>,
            [AutocompleteSetting.PRAGMATIC]:
                <p>
                    In this setting, a set of common Python keywords and identifiers such as 'self', 'import'
                    and 'append') can be abbreviated to just its initial letter.
                    You can type an entire line using abbreviations, and use Ctrl+Enter (or Cmd+Enter on MacOS)
                    to expand the abbreviations you used. For example, typing the line 'i sys' and pressing
                    Ctrl+Enter will replace the line by 'import sys'.
                    The keywords that can be abbreviated in this manner will be highlighted in yellow when you
                    type them, so that you know you can abbreviate them the next time.
                </p>
        }[this._currentSetting()];

        const settingQuickInstructions = {
            [AutocompleteSetting.NONE]: null,
            [AutocompleteSetting.DEFAULT]: null,
            [AutocompleteSetting.PRAGMATIC]:
                <p>
                    Press Ctrl+Enter after typing an entire line of code using abbreviations to expand them.
                    Keywords highlighted in yellow can be abbreviated by just its initials.
                </p>
        }[this._currentSetting()];

        const settingHeader = <h2>Setting { this.state.currentSetting + 1 }/{ NUMBER_OF_SETTINGS } - { settingDescription } </h2>;

        if (!this.state.started) {
            return (
              <div className="instructions-container">
                {settingHeader}
                <p>
                  In this study, we want to evaluate the effectiveness of
                  different autocomplete systems.
                </p>
                <p>
                  You'll be shown code on the left, and your goal is to type it
                  on the right correctly as fast as possible.
                </p>
                <p>
                  Once you're done typing, press Ctrl+S to submit (Cmd+S on
                  MacOS).
                </p>
                <p>
                  Differences in spaces (including indentation) do not matter. A
                  small number of errors is also tolerated.
                </p>
                <h2>Next setting - {settingDescription}</h2>
                {settingInstructions}
                <p>
                  Press the button below to start the typing task.
                  Feel free to take a break while you're on this screen, but please
                  do not stop once you advance to the next screen.
                </p>
                <p>
                    <input
                        type="checkbox"
                        onChange={e =>
                            this.setState({ instructionsRead: e.target.checked })
                        }
                        value={this.state.instructionsRead}
                    />
                    I've read the instructions above and am ready to start.
                </p>
                <button
                  disabled={!this.state.instructionsRead}
                  onClick={() => this._start()}
                >
                  Start
                </button>
              </div>
            );
        }

        return (
            <div>
                { settingHeader }
                <Progress current={this.state.currentTarget} total={this.state.dataset.length} />

                <p>Type the code on the right. Press Ctrl+S when you're done.</p>
                { settingQuickInstructions }
                <p className="TypingTask-error"> { this.state.error || ' ' } </p>
                <p className="task-timer">
                    {((new Date() - this.state.taskBegin) / 1000).toFixed(0)}s
                </p>

                <div className="TypingTask-columns">
                    <div className="TypingTask-code-container TypingTask-column">
                        <MonacoEditor
                            language="python"
                            value={this.state.dataset[this.state.currentTarget]}
                            options={{ readOnly: true }}
                            theme="vs-dark"
                            automaticLayout={true}
                        />
                    </div>

                    <div className="TypingTask-code-container TypingTask-column">
                        <CodeEditor
                            setting={this._currentSetting()}
                            code={this.state.code}
                            mountEditor={e => this.mountEditor(e)}
                            onChange={(nV, e) => this.onChange(nV, e)}
                            submitCode={c => this.submitCode(c)}
                        />
                    </div>
                </div>

            </div>
        )
    }
}

export default TypingTask;