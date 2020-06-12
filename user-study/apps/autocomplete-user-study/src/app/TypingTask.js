import React from 'react';

import Progress from './Progress.js';
import MonacoEditor from 'react-monaco-editor';

import leven from 'leven';

const MAX_ERRORS = 5;

function computeErrors(code, target) {
    return leven(code.replace(/\s/g, ''),
                 target.replace(/\s/g, ''));
}

class TypingTask extends React.Component {
    constructor(props) {
        super(props);
        this.events = [];

        this.state = {
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
                this.setState({ dataset })
                this._pushEvent({ 'type': 'start' });
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

            this.setState({
                currentTarget: this.state.currentTarget + 1,
                code: "",
            });
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

    render() {
        if (!this.state.dataset) {
            return <p>Loading dataset...</p>
        }

        return (
            <div>
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
                        />
                    </div>
                </div>

            </div>
        )
    }
}

export default TypingTask;
