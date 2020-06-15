import React from 'react';
import Progress from './Progress.js';
import MonacoEditor from 'react-monaco-editor';

import leven from 'leven';
import _ from 'underscore';

const MAX_ERRORS = 5;

function computeErrors(code, target) {
    return leven(code.replace(/\s/g, ''),
                 target.replace(/\s/g, ''));
}

const AutocompleteSetting = {
    NONE: 0,
    DEFAULT: 1,
//    PRAGMATIC: 2,
};

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
            code: "",
        };

        this.session = Math.random().toString();
    }

    onChange(newValue, event) {
        this._pushEvent({ 'type': 'change', 'value': newValue });

        this.setState({
            code: newValue,
            error: '',
        });
    }

    componentDidMount() {
        fetch('/dataset')
            .then((r) => r.json())
            .then(dataset => {
                this.setState({ dataset: dataset.slice(0, 10) });
            });
    }

    _start() {
        this._pushEvent({ 'type': 'start', setting: this._currentSetting() });
        this.setState({
            started: true,
        });
    }

    handleKeyEvent(e) {
        if (e.ctrlKey && e.code == 'Enter') {
            e.stopPropagation();
            this.submitCode();
        }
    }

    saveEvents(target) {
        fetch('/save-events', {
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

    submitCode() {
        const target = this.state.dataset[this.state.currentTarget];
        this._pushEvent({ 'type': 'submit' });

        if (computeErrors(this.state.code, target) <= MAX_ERRORS) {
            this._pushEvent({ 'type': 'end' });
            this.saveEvents(target);

            // After the last code snippet, move to the next setting.
            if (this.state.currentTarget + 1 == this.state.dataset.length) {
                this.setState({
                    code: "",
                    currentTarget: 0,
                    currentSetting: this.state.currentSetting + 1,
                    started: false,
                });
            } else {
                // Otherwise, just move to the next snippet.
                this.setState({
                    currentTarget: this.state.currentTarget + 1,
                    code: "",
                });
                this._start();
            }
        } else {
            this.setState({ error: 'Your code is too different from the target to submit.' });
        }
    }

    _pushEvent(event) {
        this.events.push({ 
            ...event, 
            timestamp: new Date().toISOString()
        });
    }

    mountEditor(e) {
        e.focus();
        e.onKeyDown(evt => this.handleKeyEvent(evt));
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
        }[this._currentSetting()];

        const settingHeader = <h2>Setting { this.state.currentSetting + 1 }/{ NUMBER_OF_SETTINGS } - { settingDescription } </h2>;

        if (!this.state.started) {
            return (
                <div>
                    { settingHeader }
                    <p>Press the button below to start the typing task. You'll be shown code on the left, and the goal is to type it on the right.</p>
                    <p>Once you're done typing, press Ctrl+Enter to submit.</p>
                    <button onClick={() => this._start()}>Start</button>
                </div>
            );
        }

        return (
            <div>
                { settingHeader }
                <Progress current={this.state.currentTarget} total={this.state.dataset.length} />
                
                <p>Type the code on the right. Press Ctrl+Enter when you're done.</p>
                <p className="TypingTask-error"> { this.state.error || ' ' } </p>

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
                        <MonacoEditor
                            language="python"
                            theme="vs-dark"
                            value={this.state.code}
                            onChange={(nV, e) => this.onChange(nV, e)}
                            editorDidMount={e => this.mountEditor(e)}
                            automaticLayout={true}
                            options={{
                                tabCompletion: enableCompletion ? "on" : "off",
                                quickSuggestions: enableCompletion,
                                snippetSuggestions: enableCompletion ? true : "none",
                            }}
                        />
                    </div>
                </div>

            </div>
        )
    }
}

export default TypingTask;
