import React from 'react';

import CodeEditor, { AutocompleteSetting } from './CodeEditor';

export default () => (
    <div style={{ width: 1024, height: 768, padding: '2em'}}>
        <h1>Playground</h1>
        <CodeEditor setting={AutocompleteSetting.PRAGMATIC} />
    </div>
);