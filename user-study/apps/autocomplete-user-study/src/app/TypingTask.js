import React from 'react';
import Progress from './Progress.js';
import Consent from './Consent';
import MonacoEditor from 'react-monaco-editor';
import CodeEditor, { AutocompleteSetting } from './CodeEditor';
import YouTube from 'react-youtube';

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
            userConsented: false,
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
        if (!this.state.userConsented) {
            return <Consent start={() => this.setState({ userConsented: true })} />;
        }

        if (!this.state.dataset) {
            return <p>Loading dataset...</p>
        }

        if (this.state.currentSetting < NUMBER_OF_SETTINGS) {
            return (
                <p>
                    Thank you for participating!
                    Finally, please fill out
                    <a href="https://forms.gle/XCvFzKt8J7HMek3W9">
                        this one-minute survey about your experience.
                    </a>
                </p>
            );
        }

        const enableCompletion = (this.settingsOrder[this.state.currentSetting] == AutocompleteSetting.DEFAULT);

        const settingDescription = {
            [AutocompleteSetting.NONE]: "No autocomplete",
            [AutocompleteSetting.DEFAULT]: "VSCode's default autocomplete",
            [AutocompleteSetting.PRAGMATIC]: "Pragmatic autocomplete",
            [AutocompleteSetting.BOTH]: "VSCode's + pragmatic autocomplete",
        }[this._currentSetting()];

        const videoId = {
            [AutocompleteSetting.NONE]: 'C_ngwHWUp8s',
            [AutocompleteSetting.DEFAULT]: 'eCyuVOnsn84',
            [AutocompleteSetting.PRAGMATIC]: 'oS3uFM9okfw',
            [AutocompleteSetting.BOTH]: 'Guu5XQmapxg',
        }[this._currentSetting()];

        const ctrlEnterInstructions = (
                <p>
                    Press Ctrl+Enter after typing an entire line of code using abbreviations to expand them.
                    Keywords highlighted in yellow can be abbreviated by just its initials.
                </p>
        );

        const settingQuickInstructions = {
            [AutocompleteSetting.NONE]: null,
            [AutocompleteSetting.DEFAULT]: null,
            [AutocompleteSetting.PRAGMATIC]: ctrlEnterInstructions,
            [AutocompleteSetting.BOTH]: ctrlEnterInstructions,
        }[this._currentSetting()];

        const settingHeader = <h2>Setting { this.state.currentSetting + 1 }/{ NUMBER_OF_SETTINGS } - { settingDescription } </h2>;

        if (!this.state.started) {
            return (
              <div className="instructions-container">
                {settingHeader}
                <p>
                  In this study, we want to evaluate the effectiveness of
                  different autocomplete systems. Please watch the video below
                  for a quick explanation of the settings you will use in this section.
                </p>

                <YouTube videoId={videoId} />

                <p>
                  You'll be shown code on the left, and your goal is to type it
                  on the right correctly as fast as possible.
                  Once you're done typing, press Ctrl+S to submit (Cmd+S on
                  MacOS).
                </p>
                <p>
                  Differences in spaces (including indentation) do not matter. A
                  small number of errors is tolerated.
                </p>
                <p>
                  Feel free to take a break now before the task starts.
                  When you're ready to start the task, which will take
                  approximately 5 minutes, press the button below. 
                </p>
                <button onClick={() => this._start()}>
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
                            onKeyDown={key => this._pushEvent({ 'type': 'key-press', key })}
                        />
                    </div>
                </div>

            </div>
        )
    }
}

export default TypingTask;